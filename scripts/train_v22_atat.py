"""
MALLORN v22: ATAT (Astronomical Transformer for Time series And Tabular data)

First implementation of ATAT for MALLORN classification.
Based on: https://arxiv.org/abs/2405.03078

This combines:
- Light-curve transformer with time modulation
- Tabular transformer with quantile feature tokenizer
- Multi-modal fusion

Comparing to:
- v19 GBM: OOF F1 = 0.6626
- v11 LSTM: OOF F1 = 0.12
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import QuantileTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

base_path = Path(__file__).parent.parent

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)

print("=" * 60, flush=True)
print("MALLORN v22: ATAT Implementation", flush=True)
print("=" * 60, flush=True)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_lc = data['train_lc']
test_lc = data['test_lc']
train_meta = data['train_meta']
test_meta = data['test_meta']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()

y = train_meta['target'].values

print(f"   Train: {len(train_ids)} objects ({y.sum()} TDE)", flush=True)
print(f"   Test: {len(test_ids)} objects", flush=True)

# ====================
# 2. PREPARE TABULAR FEATURES
# ====================
print("\n2. Preparing tabular features...", flush=True)

# Load cached features (same as v20c)
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
selected_features = clean_features.head(100)['feature'].tolist()  # Use top 100

# Get feature matrices
train_features = train_features.set_index('object_id').loc[train_ids].reset_index()
test_features_df = test_features.set_index('object_id').loc[test_ids].reset_index()

available_features = [f for f in selected_features if f in train_features.columns]
X_tab_train = train_features[available_features].values.astype(np.float32)
X_tab_test = test_features_df[available_features].values.astype(np.float32)

# Handle NaN
X_tab_train = np.nan_to_num(X_tab_train, nan=0, posinf=0, neginf=0)
X_tab_test = np.nan_to_num(X_tab_test, nan=0, posinf=0, neginf=0)

# Quantile transform
qt = QuantileTransformer(output_distribution='normal', random_state=42)
X_tab_train = qt.fit_transform(X_tab_train)
X_tab_test = qt.transform(X_tab_test)

n_features = X_tab_train.shape[1]
print(f"   Tabular features: {n_features}", flush=True)

# ====================
# 3. PREPARE LIGHTCURVE DATA
# ====================
print("\n3. Preparing light curve data...", flush=True)

BAND_MAP = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}
MAX_SEQ_LEN = 300

def prepare_lightcurves(lc_df, object_ids, max_len=MAX_SEQ_LEN):
    """Prepare lightcurves for ATAT."""
    n_objects = len(object_ids)

    flux_arr = np.zeros((n_objects, max_len, 2), dtype=np.float32)
    time_arr = np.zeros((n_objects, max_len), dtype=np.float32)
    band_arr = np.zeros((n_objects, max_len), dtype=np.int64)
    mask_arr = np.zeros((n_objects, max_len), dtype=np.float32)

    grouped = {obj_id: group for obj_id, group in lc_df.groupby('object_id')}

    for i, obj_id in enumerate(object_ids):
        if i % 500 == 0:
            print(f"      Processing {i}/{n_objects}...", flush=True)

        if obj_id not in grouped:
            continue

        obj_lc = grouped[obj_id].sort_values('Time (MJD)')

        times = obj_lc['Time (MJD)'].values
        fluxes = obj_lc['Flux'].values
        flux_errs = obj_lc['Flux_err'].values
        bands = obj_lc['Filter'].values

        # Normalize time
        times = times - times.min()

        # Normalize flux per object
        flux_scale = np.median(np.abs(fluxes[fluxes != 0])) if np.any(fluxes != 0) else 1.0
        if flux_scale == 0:
            flux_scale = 1.0

        n_obs = min(len(times), max_len)

        time_arr[i, :n_obs] = times[:n_obs]
        flux_arr[i, :n_obs, 0] = fluxes[:n_obs] / flux_scale
        flux_arr[i, :n_obs, 1] = flux_errs[:n_obs] / flux_scale
        mask_arr[i, :n_obs] = 1.0

        for j in range(n_obs):
            if bands[j] in BAND_MAP:
                band_arr[i, j] = BAND_MAP[bands[j]]

    # Handle NaN/inf
    flux_arr = np.nan_to_num(flux_arr, nan=0, posinf=0, neginf=0)
    time_arr = np.nan_to_num(time_arr, nan=0, posinf=0, neginf=0)

    return flux_arr, time_arr, band_arr, mask_arr

print("   Preparing train lightcurves...", flush=True)
flux_train, time_train, band_train, mask_train = prepare_lightcurves(train_lc, train_ids)

print("   Preparing test lightcurves...", flush=True)
flux_test, time_test, band_test, mask_test = prepare_lightcurves(test_lc, test_ids)

print(f"   Lightcurve shape: {flux_train.shape}", flush=True)

# ====================
# 4. CREATE DATASET
# ====================
print("\n4. Creating PyTorch dataset...", flush=True)

class MALLORNDataset(Dataset):
    def __init__(self, flux, time, band, mask, features, labels=None):
        self.flux = torch.from_numpy(flux)
        self.time = torch.from_numpy(time)
        self.band = torch.from_numpy(band)
        self.mask = torch.from_numpy(mask)
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels) if labels is not None else None

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        if self.labels is not None:
            return (
                self.flux[idx], self.time[idx], self.band[idx],
                self.mask[idx], self.features[idx], self.labels[idx]
            )
        else:
            return (
                self.flux[idx], self.time[idx], self.band[idx],
                self.mask[idx], self.features[idx]
            )

# ====================
# 5. DEFINE ATAT MODEL
# ====================
print("\n5. Creating ATAT model...", flush=True)

from models.atat import ATAT

model_config = {
    'n_features': n_features,
    'n_classes': 2,
    'lc_embed_dim': 48,  # Smaller for efficiency
    'tab_embed_dim': 32,
    'lc_layers': 2,
    'tab_layers': 2,
    'n_heads': 4,
    'dropout': 0.3,
    'n_bands': 6
}

# Count parameters
test_model = ATAT(**model_config)
n_params = sum(p.numel() for p in test_model.parameters())
print(f"   Model parameters: {n_params:,}", flush=True)

# ====================
# 6. TRAINING LOOP
# ====================
print("\n6. Training ATAT (5-fold CV)...", flush=True)

n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Training parameters
batch_size = 64
n_epochs = 30
lr = 2e-4
weight_decay = 1e-4

# Class weights for imbalanced data
n_pos = y.sum()
n_neg = len(y) - n_pos
pos_weight = torch.tensor([n_neg / n_pos]).to(device)

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(test_ids), n_splits))
models = []

for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
    print(f"\n   Fold {fold+1}/{n_splits}:", flush=True)

    # Create datasets
    train_dataset = MALLORNDataset(
        flux_train[train_idx], time_train[train_idx],
        band_train[train_idx], mask_train[train_idx],
        X_tab_train[train_idx], y[train_idx]
    )
    val_dataset = MALLORNDataset(
        flux_train[val_idx], time_train[val_idx],
        band_train[val_idx], mask_train[val_idx],
        X_tab_train[val_idx], y[val_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = ATAT(**model_config).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Training
    best_val_f1 = 0
    best_model_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0

        for batch in train_loader:
            flux, time, band, mask, features, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            logits = model(flux, time, band, mask, features, mode='both')
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                flux, time, band, mask, features, labels = [b.to(device) for b in batch]
                logits = model(flux, time, band, mask, features, mode='both')
                probs = torch.softmax(logits, dim=1)[:, 1]
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)

        # Find best threshold
        best_f1 = 0
        for t in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(val_labels, (val_preds > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1

        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1}: loss={train_loss/len(train_loader):.4f}, val_F1={best_f1:.4f}", flush=True)

        # Early stopping
        if best_f1 > best_val_f1:
            best_val_f1 = best_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"      Early stopping at epoch {epoch+1}", flush=True)
                break

    # Load best model
    model.load_state_dict(best_model_state)
    models.append(model)

    # Get OOF predictions
    model.eval()
    with torch.no_grad():
        for i in range(0, len(val_idx), batch_size):
            batch_idx = val_idx[i:i+batch_size]

            flux = torch.from_numpy(flux_train[batch_idx]).to(device)
            time = torch.from_numpy(time_train[batch_idx]).to(device)
            band = torch.from_numpy(band_train[batch_idx]).to(device)
            mask = torch.from_numpy(mask_train[batch_idx]).to(device)
            features = torch.from_numpy(X_tab_train[batch_idx]).to(device)

            logits = model(flux, time, band, mask, features, mode='both')
            probs = torch.softmax(logits, dim=1)[:, 1]
            oof_preds[batch_idx] = probs.cpu().numpy()

    # Get test predictions
    test_dataset = MALLORNDataset(
        flux_test, time_test, band_test, mask_test, X_tab_test
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    fold_test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            flux, time, band, mask, features = [b.to(device) for b in batch]
            logits = model(flux, time, band, mask, features, mode='both')
            probs = torch.softmax(logits, dim=1)[:, 1]
            fold_test_preds.extend(probs.cpu().numpy())

    test_preds[:, fold] = np.array(fold_test_preds)

    print(f"      Best val F1: {best_val_f1:.4f}", flush=True)

# ====================
# 7. EVALUATE
# ====================
print("\n7. Evaluating OOF predictions...", flush=True)

best_f1 = 0
best_thresh = 0.5

for t in np.arange(0.05, 0.95, 0.01):
    f1 = f1_score(y, (oof_preds > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}", flush=True)

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))

print(f"   Precision: {tp/(tp+fp):.4f}", flush=True)
print(f"   Recall: {tp/(tp+fn):.4f}", flush=True)

# ====================
# 8. CREATE SUBMISSION
# ====================
print("\n8. Creating submission...", flush=True)

test_avg = test_preds.mean(axis=1)
test_final = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v22_atat.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved to {submission_path}", flush=True)
print(f"   Predictions: {test_final.sum()} TDEs / {len(test_final)} total", flush=True)

# Save model
with open(base_path / 'data/processed/models_v22_atat.pkl', 'wb') as f:
    pickle.dump({
        'model_config': model_config,
        'best_thresh': best_thresh,
        'oof_f1': best_f1,
        'feature_cols': available_features
    }, f)

# ====================
# SUMMARY
# ====================
print("\n" + "=" * 60, flush=True)
print("ATAT Training Complete!", flush=True)
print("=" * 60, flush=True)

print(f"\nVersion Comparison:", flush=True)
print(f"  v11 (LSTM):           OOF F1 = 0.12", flush=True)
print(f"  v19 (GBM+GP2D):       OOF F1 = 0.6626, LB = 0.6649", flush=True)
print(f"  v21 (XGB only):       OOF F1 = 0.6708", flush=True)
print(f"  v22 (ATAT):           OOF F1 = {best_f1:.4f}", flush=True)

if best_f1 > 0.12:
    print(f"\n  ATAT beats homemade LSTM! (+{(best_f1-0.12)*100:.1f}%)", flush=True)
else:
    print(f"\n  ATAT needs more tuning...", flush=True)

print("=" * 60, flush=True)
