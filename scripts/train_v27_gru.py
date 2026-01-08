"""
MALLORN v27: GRU with Attention for Light Curve Classification

Architecture:
- Multi-band GRU encoder with shared weights
- Attention mechanism to focus on important time points
- Band aggregation layer
- Feature extraction for XGBoost hybrid

Target: Beat v21's 0.6708 OOF / 0.6649 LB
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent
os.chdir(base_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("=" * 60)
print("MALLORN v27: GRU with Attention")
print("=" * 60)

# ====================
# 1. LOAD DATA
# ====================
print("\n[1/6] Loading data...", flush=True)

train_log = pd.read_csv(base_path / 'data/raw/train_log.csv')
test_log = pd.read_csv(base_path / 'data/raw/test_log.csv')

train_ids = train_log['object_id'].tolist()
test_ids = test_log['object_id'].tolist()
y = train_log['target'].values

# Load light curves
train_lcs = []
for i in range(1, 21):
    path = base_path / f'data/raw/split_{i:02d}/train_full_lightcurves.csv'
    if path.exists():
        train_lcs.append(pd.read_csv(path))
train_lc = pd.concat(train_lcs, ignore_index=True)

test_lcs = []
for i in range(1, 21):
    path = base_path / f'data/raw/split_{i:02d}/test_full_lightcurves.csv'
    if path.exists():
        test_lcs.append(pd.read_csv(path))
test_lc = pd.concat(test_lcs, ignore_index=True)

print(f"   Train: {len(train_ids)} objects, Test: {len(test_ids)} objects")

# ====================
# 2. PREPROCESS LIGHT CURVES
# ====================
print("\n[2/6] Preprocessing light curves...", flush=True)

BANDS = ['u', 'g', 'r', 'i', 'z', 'y']
BAND_TO_IDX = {b: i for i, b in enumerate(BANDS)}
MAX_OBS = 100  # Max observations per band

def preprocess_lightcurve(lc_df, obj_id, max_obs=MAX_OBS):
    """Convert light curve to tensor format."""
    obj_lc = lc_df[lc_df['object_id'] == obj_id]

    # Initialize arrays
    flux = np.zeros((len(BANDS), max_obs), dtype=np.float32)
    time = np.zeros((len(BANDS), max_obs), dtype=np.float32)
    flux_err = np.zeros((len(BANDS), max_obs), dtype=np.float32)
    mask = np.ones((len(BANDS), max_obs), dtype=np.float32)  # 1 = padded

    for band in BANDS:
        band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')
        n_obs = min(len(band_data), max_obs)

        if n_obs > 0:
            band_idx = BAND_TO_IDX[band]

            # Extract values
            t = band_data['Time (MJD)'].values[:n_obs]
            f = band_data['Flux'].values[:n_obs]
            f_err = band_data['Flux_err'].values[:n_obs]

            # Normalize time (start from 0, scale to ~1)
            t = (t - t.min()) / 500.0  # Assume ~500 day span

            # Normalize flux (z-score per band)
            f_mean, f_std = f.mean(), f.std()
            if f_std > 0:
                f = (f - f_mean) / f_std
                f_err = f_err / f_std

            # Fill arrays
            flux[band_idx, :n_obs] = f
            time[band_idx, :n_obs] = t
            flux_err[band_idx, :n_obs] = f_err
            mask[band_idx, :n_obs] = 0  # 0 = valid

    return flux, time, flux_err, mask


class LightCurveDataset(Dataset):
    def __init__(self, lc_df, object_ids, labels=None):
        self.lc_df = lc_df
        self.object_ids = object_ids
        self.labels = labels

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, idx):
        obj_id = self.object_ids[idx]
        flux, time, flux_err, mask = preprocess_lightcurve(self.lc_df, obj_id)

        data = {
            'flux': torch.tensor(flux),
            'time': torch.tensor(time),
            'flux_err': torch.tensor(flux_err),
            'mask': torch.tensor(mask),
        }

        if self.labels is not None:
            data['label'] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return data


# ====================
# 3. DEFINE GRU MODEL
# ====================
print("\n[3/6] Building GRU model...", flush=True)

class AttentionGRU(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()

        # Band embedding
        self.band_embed = nn.Embedding(len(BANDS), 16)

        # GRU encoder (shared across bands)
        self.gru = nn.GRU(
            input_size=input_dim + 16,  # flux, time, flux_err + band embedding
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Band aggregation
        self.band_agg = nn.Sequential(
            nn.Linear(hidden_dim * 2 * len(BANDS), hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.hidden_dim = hidden_dim

    def forward(self, flux, time, flux_err, mask, return_features=False):
        batch_size = flux.shape[0]
        band_features = []

        for band_idx in range(len(BANDS)):
            # Get band data: (batch, max_obs)
            band_flux = flux[:, band_idx, :]
            band_time = time[:, band_idx, :]
            band_err = flux_err[:, band_idx, :]
            band_mask = mask[:, band_idx, :]

            # Stack features: (batch, max_obs, 3)
            x = torch.stack([band_flux, band_time, band_err], dim=-1)

            # Add band embedding: (batch, max_obs, 3 + 16)
            band_emb = self.band_embed(torch.tensor(band_idx, device=flux.device))
            band_emb = band_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, MAX_OBS, -1)
            x = torch.cat([x, band_emb], dim=-1)

            # GRU encoding
            gru_out, _ = self.gru(x)  # (batch, max_obs, hidden*2)

            # Attention with masking (handle all-masked case)
            attn_scores = self.attention(gru_out).squeeze(-1)  # (batch, max_obs)

            # Check if any valid positions exist
            valid_count = (1 - band_mask).sum(dim=1, keepdim=True)  # (batch, 1)
            has_valid = valid_count > 0

            # Apply mask
            attn_scores = attn_scores.masked_fill(band_mask.bool(), float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=-1)

            # Replace NaN with uniform weights for empty bands
            attn_weights = torch.where(
                torch.isnan(attn_weights),
                torch.ones_like(attn_weights) / MAX_OBS,
                attn_weights
            )
            attn_weights = attn_weights.unsqueeze(-1)  # (batch, max_obs, 1)

            # Weighted sum
            band_rep = (gru_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

            # Zero out bands with no valid data
            band_rep = band_rep * has_valid.float()

            band_features.append(band_rep)

        # Concatenate all bands
        all_bands = torch.cat(band_features, dim=-1)  # (batch, hidden*2*6)

        # Band aggregation
        features = self.band_agg(all_bands)  # (batch, hidden*2)

        if return_features:
            return features

        # Classification
        logits = self.classifier(features).squeeze(-1)
        return logits


# ====================
# 4. TRAIN GRU MODEL
# ====================
print("\n[4/6] Training GRU model...", flush=True)

def train_gru_fold(train_ids_fold, val_ids_fold, y_train, y_val, epochs=30):
    """Train GRU for one fold."""

    # Create datasets
    train_dataset = LightCurveDataset(train_lc, train_ids_fold, y_train)
    val_dataset = LightCurveDataset(train_lc, val_ids_fold, y_val)

    # Handle class imbalance with weighted sampler
    class_counts = np.bincount(y_train)
    weights = 1.0 / class_counts
    sample_weights = weights[y_train]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model
    model = AttentionGRU(hidden_dim=64, num_layers=2, dropout=0.3).to(device)

    # Loss with class weights
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_f1 = 0
    best_model_state = model.state_dict().copy()  # Initialize with current state
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            flux = batch['flux'].to(device)
            time = batch['time'].to(device)
            flux_err = batch['flux_err'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(flux, time, flux_err, mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                flux = batch['flux'].to(device)
                time = batch['time'].to(device)
                flux_err = batch['flux_err'].to(device)
                mask = batch['mask'].to(device)

                logits = model(flux, time, flux_err, mask)
                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(batch['label'].numpy())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)

        # Find best threshold (use lower range due to class imbalance)
        best_thresh = 0.1
        best_val_f1 = 0
        for thresh in np.arange(0.01, 0.5, 0.02):
            preds = (val_preds > thresh).astype(int)
            if preds.sum() > 0:  # Only if we predict some positives
                f1 = f1_score(val_labels, preds)
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_thresh = thresh

        # Debug: show prediction distribution
        if (epoch + 1) == 1:
            print(f"      Pred stats: min={val_preds.min():.4f}, max={val_preds.max():.4f}, mean={val_preds.mean():.4f}")

        scheduler.step(1 - best_val_f1)

        if best_val_f1 > best_f1:
            best_f1 = best_val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 10:
            break

        # Debug output every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1}: val_f1={best_val_f1:.4f}, best_f1={best_f1:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_f1


# ====================
# 5. EXTRACT GRU FEATURES
# ====================
print("\n[5/6] Cross-validation and feature extraction...", flush=True)

n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Storage for GRU features
train_gru_features = np.zeros((len(train_ids), 128))  # hidden_dim * 2
test_gru_features = np.zeros((len(test_ids), 128))
oof_preds_gru = np.zeros(len(y))

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(train_ids, y)):
    print(f"\n   Fold {fold+1}/{n_splits}:", flush=True)

    train_ids_fold = [train_ids[i] for i in train_idx]
    val_ids_fold = [train_ids[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    # Train GRU
    model, fold_f1 = train_gru_fold(train_ids_fold, val_ids_fold, y_train, y_val, epochs=30)
    fold_scores.append(fold_f1)
    print(f"      GRU F1: {fold_f1:.4f}")

    # Extract features for validation set
    model.eval()
    val_dataset = LightCurveDataset(train_lc, val_ids_fold)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    val_features = []
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            flux = batch['flux'].to(device)
            time = batch['time'].to(device)
            flux_err = batch['flux_err'].to(device)
            mask = batch['mask'].to(device)

            features = model(flux, time, flux_err, mask, return_features=True)
            logits = model.classifier(features).squeeze(-1)

            val_features.append(features.cpu().numpy())
            val_preds.append(torch.sigmoid(logits).cpu().numpy())

    val_features = np.concatenate(val_features, axis=0)
    val_preds = np.concatenate(val_preds, axis=0)

    train_gru_features[val_idx] = val_features
    oof_preds_gru[val_idx] = val_preds

    # Extract features for test set (average across folds)
    test_dataset = LightCurveDataset(test_lc, test_ids)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    test_features_fold = []
    with torch.no_grad():
        for batch in test_loader:
            flux = batch['flux'].to(device)
            time = batch['time'].to(device)
            flux_err = batch['flux_err'].to(device)
            mask = batch['mask'].to(device)

            features = model(flux, time, flux_err, mask, return_features=True)
            test_features_fold.append(features.cpu().numpy())

    test_features_fold = np.concatenate(test_features_fold, axis=0)
    test_gru_features += test_features_fold / n_splits

# GRU-only performance
best_gru_f1 = 0
for thresh in np.arange(0.1, 0.9, 0.05):
    f1 = f1_score(y, (oof_preds_gru > thresh).astype(int))
    if f1 > best_gru_f1:
        best_gru_f1 = f1

print(f"\n   GRU-only OOF F1: {best_gru_f1:.4f}")
print(f"   Mean Fold F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

# ====================
# 6. COMBINE WITH XGBOOST
# ====================
print("\n[6/6] Training XGBoost with GRU features...", flush=True)

# Load v21 features
import pickle

cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_features = cached['train_features']
test_features = cached['test_features']

selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
importance_df = selection['importance_df']
high_corr_df = selection['high_corr_df']

corr_to_drop = set()
for _, row in high_corr_df.iterrows():
    if row['feature_1'] not in corr_to_drop:
        corr_to_drop.add(row['feature_2'])
clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
selected_120 = clean_features.head(120)['feature'].tolist()

# GP2D features
with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

# Merge and select
train_combined = train_features.merge(train_gp2d, on='object_id', how='left')
test_combined = test_features.merge(test_gp2d, on='object_id', how='left')

base_cols = [c for c in selected_120 if c in train_combined.columns]
v21_feature_cols = base_cols + gp2d_cols
v21_feature_cols = list(dict.fromkeys(v21_feature_cols))

train_combined = train_combined.set_index('object_id').loc[train_ids].reset_index()
test_combined = test_combined.set_index('object_id').loc[test_ids].reset_index()

X_v21 = train_combined[v21_feature_cols].values.astype(np.float32)
X_v21 = np.nan_to_num(X_v21, nan=0, posinf=0, neginf=0)
X_test_v21 = test_combined[v21_feature_cols].values.astype(np.float32)
X_test_v21 = np.nan_to_num(X_test_v21, nan=0, posinf=0, neginf=0)

# Combine v21 features with GRU features
X_combined = np.hstack([X_v21, train_gru_features])
X_test_combined = np.hstack([X_test_v21, test_gru_features])

print(f"   V21 features: {X_v21.shape[1]}")
print(f"   GRU features: {train_gru_features.shape[1]}")
print(f"   Combined: {X_combined.shape[1]}")

# Load Optuna params
with open(base_path / 'data/processed/optuna_v20c_results.pkl', 'rb') as f:
    optuna_data = pickle.load(f)
xgb_params = optuna_data['xgb_best_params']

# Train XGBoost
scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
oof_preds_xgb = np.zeros(len(y))
test_preds_xgb = np.zeros((len(test_ids), n_splits))

for fold, (train_idx, val_idx) in enumerate(cv.split(X_combined, y)):
    model = xgb.XGBClassifier(**xgb_params, scale_pos_weight=scale_pos_weight, random_state=42, verbosity=0)
    model.fit(X_combined[train_idx], y[train_idx],
              eval_set=[(X_combined[val_idx], y[val_idx])], verbose=False)

    oof_preds_xgb[val_idx] = model.predict_proba(X_combined[val_idx])[:, 1]
    test_preds_xgb[:, fold] = model.predict_proba(X_test_combined)[:, 1]

# Find best threshold
best_oof_f1 = 0
best_thresh = 0.07
for thresh in np.arange(0.03, 0.20, 0.01):
    f1 = f1_score(y, (oof_preds_xgb > thresh).astype(int))
    if f1 > best_oof_f1:
        best_oof_f1 = f1
        best_thresh = thresh

print(f"\n{'='*60}")
print(f"Combined (v21 + GRU) OOF F1: {best_oof_f1:.4f} (threshold={best_thresh:.2f})")
print(f"GRU-only OOF F1: {best_gru_f1:.4f}")
print(f"{'='*60}")

# Comparison
baseline_f1 = 0.6708
improvement = (best_oof_f1 - baseline_f1) / baseline_f1 * 100
print(f"\nComparison with v21 baseline (OOF F1={baseline_f1:.4f}):")
if best_oof_f1 > baseline_f1:
    print(f"   IMPROVEMENT: +{improvement:.2f}%")
else:
    print(f"   Change: {improvement:+.2f}%")

# Generate submission
test_probs_mean = test_preds_xgb.mean(axis=1)
test_binary = (test_probs_mean > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})
submission_path = base_path / 'submissions/submission_v27_gru.csv'
submission.to_csv(submission_path, index=False)

print(f"\nSaved to {submission_path}")
print(f"Predicted TDEs: {test_binary.sum()} / {len(test_binary)}")

print("\nDone!")
