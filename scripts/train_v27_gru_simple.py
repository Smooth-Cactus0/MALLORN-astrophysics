"""
MALLORN v27: Simple GRU for Light Curve Feature Extraction

Simpler architecture with mean pooling instead of attention.
Uses extracted features with XGBoost.
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
import pickle
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent
os.chdir(base_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

print("=" * 60)
print("MALLORN v27: Simple GRU")
print("=" * 60)

# ====================
# 1. LOAD DATA
# ====================
print("\n[1/5] Loading data...", flush=True)

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

print(f"   Train: {len(train_ids)}, Test: {len(test_ids)}")

# ====================
# 2. PREPROCESS INTO SEQUENCES
# ====================
print("\n[2/5] Preprocessing...", flush=True)

BANDS = ['g', 'r', 'i', 'z']  # Focus on key bands
BAND_TO_IDX = {b: i for i, b in enumerate(BANDS)}
MAX_OBS = 200  # Total observations across all bands

def preprocess_multiband(lc_df, obj_id, max_obs=MAX_OBS):
    """Create single sequence from all bands, sorted by time."""
    obj_lc = lc_df[lc_df['object_id'] == obj_id].copy()

    if len(obj_lc) == 0:
        return np.zeros((max_obs, 5), dtype=np.float32), 0

    # Filter to key bands
    obj_lc = obj_lc[obj_lc['Filter'].isin(BANDS)]

    if len(obj_lc) == 0:
        return np.zeros((max_obs, 5), dtype=np.float32), 0

    # Sort by time
    obj_lc = obj_lc.sort_values('Time (MJD)')

    # Normalize
    times = obj_lc['Time (MJD)'].values
    times = (times - times.min()) / max(times.max() - times.min(), 1)

    fluxes = obj_lc['Flux'].values
    flux_mean, flux_std = fluxes.mean(), fluxes.std()
    if flux_std > 0:
        fluxes = (fluxes - flux_mean) / flux_std

    flux_errs = obj_lc['Flux_err'].values
    if flux_std > 0:
        flux_errs = flux_errs / flux_std

    band_ids = obj_lc['Filter'].map(BAND_TO_IDX).values

    n_obs = min(len(obj_lc), max_obs)

    # Create sequence: (time, flux, flux_err, band_id, band_onehot)
    seq = np.zeros((max_obs, 5), dtype=np.float32)  # time, flux, err, band_sin, band_cos
    seq[:n_obs, 0] = times[:n_obs]
    seq[:n_obs, 1] = fluxes[:n_obs]
    seq[:n_obs, 2] = flux_errs[:n_obs]
    # Encode band as sin/cos
    seq[:n_obs, 3] = np.sin(2 * np.pi * band_ids[:n_obs] / len(BANDS))
    seq[:n_obs, 4] = np.cos(2 * np.pi * band_ids[:n_obs] / len(BANDS))

    return seq, n_obs


class LCDataset(Dataset):
    def __init__(self, lc_df, object_ids, labels=None):
        self.lc_df = lc_df
        self.object_ids = object_ids
        self.labels = labels

        # Preprocess all
        print("      Preprocessing sequences...", flush=True)
        self.sequences = []
        self.lengths = []
        for i, obj_id in enumerate(object_ids):
            seq, length = preprocess_multiband(lc_df, obj_id)
            self.sequences.append(seq)
            self.lengths.append(length)
            if (i + 1) % 1000 == 0:
                print(f"      {i+1}/{len(object_ids)}", flush=True)

        self.sequences = np.array(self.sequences)
        self.lengths = np.array(self.lengths)

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, idx):
        data = {
            'seq': torch.tensor(self.sequences[idx]),
            'length': torch.tensor(self.lengths[idx])
        }
        if self.labels is not None:
            data['label'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return data


# ====================
# 3. SIMPLE GRU MODEL
# ====================
print("\n[3/5] Building model...", flush=True)

class SimpleGRU(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.hidden_dim = hidden_dim

    def forward(self, seq, lengths, return_features=False):
        # Pack padded sequence for efficiency
        batch_size = seq.shape[0]

        # GRU
        gru_out, _ = self.gru(seq)  # (batch, seq_len, hidden*2)

        # Mean pooling over valid positions
        # Create mask based on lengths
        mask = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)

        # Masked mean pooling
        masked_out = gru_out * mask
        sum_out = masked_out.sum(dim=1)  # (batch, hidden*2)
        lengths_clamped = lengths.clamp(min=1).unsqueeze(-1).float()
        features = sum_out / lengths_clamped  # (batch, hidden*2)

        if return_features:
            return features

        logits = self.fc(features).squeeze(-1)
        return logits


# ====================
# 4. TRAIN AND EXTRACT
# ====================
print("\n[4/5] Training and extracting features...", flush=True)

# Prepare datasets
print("   Creating train dataset...")
train_dataset = LCDataset(train_lc, train_ids, y)
print("   Creating test dataset...")
test_dataset = LCDataset(test_lc, test_ids)

n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

train_gru_features = np.zeros((len(train_ids), 128))
test_gru_features = np.zeros((len(test_ids), 128))
oof_preds_gru = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(cv.split(train_ids, y)):
    print(f"\n   Fold {fold+1}/{n_splits}:", flush=True)

    # Create fold datasets
    train_seqs = train_dataset.sequences[train_idx]
    train_lens = train_dataset.lengths[train_idx]
    val_seqs = train_dataset.sequences[val_idx]
    val_lens = train_dataset.lengths[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    # Create loaders
    train_data = torch.utils.data.TensorDataset(
        torch.tensor(train_seqs), torch.tensor(train_lens), torch.tensor(y_train, dtype=torch.float32)
    )
    val_data = torch.utils.data.TensorDataset(
        torch.tensor(val_seqs), torch.tensor(val_lens), torch.tensor(y_val, dtype=torch.float32)
    )

    # Weighted sampler
    class_counts = np.bincount(y_train)
    weights = 1.0 / class_counts
    sample_weights = weights[y_train]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_data, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    # Model
    model = SimpleGRU(hidden_dim=64, num_layers=2, dropout=0.3).to(device)

    # Loss
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_f1 = 0
    best_state = model.state_dict().copy()

    for epoch in range(20):
        # Train
        model.train()
        for batch in train_loader:
            seq, length, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(seq, length)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                seq, length, _ = [b.to(device) for b in batch]
                logits = model(seq, length)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())

        val_preds = np.array(val_preds)

        # Find best threshold
        best_val_f1 = 0
        for thresh in np.arange(0.01, 0.5, 0.02):
            preds = (val_preds > thresh).astype(int)
            if preds.sum() > 0:
                f1 = f1_score(y_val, preds)
                best_val_f1 = max(best_val_f1, f1)

        scheduler.step(1 - best_val_f1)

        if best_val_f1 > best_f1:
            best_f1 = best_val_f1
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1}: val_f1={best_val_f1:.4f}")

    print(f"      Best F1: {best_f1:.4f}")

    # Extract features
    model.load_state_dict(best_state)
    model.eval()

    # Validation features
    with torch.no_grad():
        val_features = model(
            torch.tensor(val_seqs).to(device),
            torch.tensor(val_lens).to(device),
            return_features=True
        ).cpu().numpy()
        val_preds = torch.sigmoid(model(
            torch.tensor(val_seqs).to(device),
            torch.tensor(val_lens).to(device)
        )).cpu().numpy()

    train_gru_features[val_idx] = val_features
    oof_preds_gru[val_idx] = val_preds

    # Test features (average across folds)
    with torch.no_grad():
        test_features = model(
            torch.tensor(test_dataset.sequences).to(device),
            torch.tensor(test_dataset.lengths).to(device),
            return_features=True
        ).cpu().numpy()

    test_gru_features += test_features / n_splits

# GRU-only performance
best_gru_f1 = 0
for thresh in np.arange(0.01, 0.5, 0.02):
    preds = (oof_preds_gru > thresh).astype(int)
    if preds.sum() > 0:
        f1 = f1_score(y, preds)
        best_gru_f1 = max(best_gru_f1, f1)

print(f"\n   GRU-only OOF F1: {best_gru_f1:.4f}")

# ====================
# 5. COMBINE WITH XGBOOST
# ====================
print("\n[5/5] Training XGBoost with GRU features...", flush=True)

# Load v21 features
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

# Combine
X_combined = np.hstack([X_v21, train_gru_features])
X_test_combined = np.hstack([X_test_v21, test_gru_features])

print(f"   V21: {X_v21.shape[1]}, GRU: {train_gru_features.shape[1]}, Total: {X_combined.shape[1]}")

# Optuna params
with open(base_path / 'data/processed/optuna_v20c_results.pkl', 'rb') as f:
    optuna_data = pickle.load(f)
xgb_params = optuna_data['xgb_best_params']

# Train
scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(test_ids), n_splits))

for fold, (train_idx, val_idx) in enumerate(cv.split(X_combined, y)):
    model = xgb.XGBClassifier(**xgb_params, scale_pos_weight=scale_pos_weight, random_state=42, verbosity=0)
    model.fit(X_combined[train_idx], y[train_idx],
              eval_set=[(X_combined[val_idx], y[val_idx])], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_combined[val_idx])[:, 1]
    test_preds[:, fold] = model.predict_proba(X_test_combined)[:, 1]

# Best threshold
best_oof_f1 = 0
best_thresh = 0.07
for thresh in np.arange(0.03, 0.20, 0.01):
    f1 = f1_score(y, (oof_preds > thresh).astype(int))
    if f1 > best_oof_f1:
        best_oof_f1 = f1
        best_thresh = thresh

print(f"\n{'='*60}")
print(f"Combined OOF F1: {best_oof_f1:.4f} (threshold={best_thresh:.2f})")
print(f"GRU-only OOF F1: {best_gru_f1:.4f}")
print(f"{'='*60}")

baseline_f1 = 0.6708
improvement = (best_oof_f1 - baseline_f1) / baseline_f1 * 100
print(f"\nComparison with v21 (OOF F1={baseline_f1:.4f}): {improvement:+.2f}%")

# Submission
test_probs = test_preds.mean(axis=1)
test_binary = (test_probs > best_thresh).astype(int)

submission = pd.DataFrame({'object_id': test_ids, 'target': test_binary})
submission.to_csv(base_path / 'submissions/submission_v27_gru.csv', index=False)

print(f"\nPredicted TDEs: {test_binary.sum()} / {len(test_binary)}")
print("Done!")
