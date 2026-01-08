"""
MALLORN v26: XGBoost with ASTROMER Transfer Learning Embeddings

Key Innovation:
- Uses pre-trained ASTROMER transformer to extract embeddings from light curves
- ASTROMER was trained on ~1M MACHO light curves with self-supervised masked prediction
- These embeddings capture learned temporal patterns that may transfer to TDE detection

Target: Beat v21's 0.6708 OOF / 0.6649 LB
"""
import sys
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

# Setup paths
base_path = Path(__file__).parent.parent
os.chdir(base_path)

sys.path.insert(0, str(base_path / 'src'))

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 60, flush=True)
print("MALLORN v26: XGBoost with ASTROMER Embeddings", flush=True)
print("=" * 60, flush=True)

# ====================
# 1. LOAD DATA AND FEATURES (same as v25)
# ====================
print("\n[1/5] Loading data and features...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_lc = data['train_lc']
test_lc = data['test_lc']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

# Load cached features
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

# Merge base features with GP2D
train_combined = train_features.merge(train_gp2d, on='object_id', how='left')
test_combined = test_features.merge(test_gp2d, on='object_id', how='left')

# Select columns (base + GP2D)
base_cols = [c for c in selected_120 if c in train_combined.columns]
base_feature_cols = base_cols + gp2d_cols
base_feature_cols = list(dict.fromkeys(base_feature_cols))
base_feature_cols = [c for c in base_feature_cols if c in train_combined.columns]

print(f"   Base features: {len(base_feature_cols)}", flush=True)
print(f"   Samples: {len(y)} ({np.sum(y==1)} TDE, {np.sum(y==0)} non-TDE)", flush=True)

# ====================
# 2. EXTRACT ASTROMER EMBEDDINGS
# ====================
print("\n[2/5] Extracting ASTROMER embeddings...", flush=True)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Check for cached embeddings
train_emb_path = base_path / 'data/processed/train_astromer_embeddings.pkl'
test_emb_path = base_path / 'data/processed/test_astromer_embeddings.pkl'

if train_emb_path.exists() and test_emb_path.exists():
    print("   Loading cached embeddings...", flush=True)
    train_embeddings = pd.read_pickle(train_emb_path)
    test_embeddings = pd.read_pickle(test_emb_path)
    print(f"   Train embeddings: {train_embeddings.shape}", flush=True)
    print(f"   Test embeddings: {test_embeddings.shape}", flush=True)
else:
    # Import ASTROMER
    from ASTROMER.models import SingleBandEncoder

    print("   Loading pre-trained MACHO encoder...", flush=True)
    encoder = SingleBandEncoder()
    encoder = encoder.from_pretraining('macho')
    encoder_layer = encoder.model.get_layer('encoder')
    print("   ASTROMER loaded!", flush=True)

    def extract_embeddings_batch(lightcurves_df, object_ids, desc=""):
        """Extract ASTROMER embeddings for multiple objects."""
        max_obs = 100
        bands = ['g', 'r', 'i', 'z']  # Focus on key bands with most data

        all_results = []
        total = len(object_ids)

        for idx, obj_id in enumerate(object_ids):
            if (idx + 1) % 200 == 0:
                print(f"   {desc} Progress: {idx+1}/{total} ({100*(idx+1)/total:.1f}%)", flush=True)

            obj_lc = lightcurves_df[lightcurves_df['object_id'] == obj_id]
            result = {'object_id': obj_id}

            for band in bands:
                band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

                # Default values for missing bands
                if len(band_data) < 5:
                    for suffix in ['mean', 'max', 'std']:
                        result[f'{band}_emb_{suffix}'] = np.zeros(256)
                    continue

                # Prepare data
                times = band_data['Time (MJD)'].values
                fluxes = band_data['Flux'].values

                times = times - times.min()
                flux_mean, flux_std = fluxes.mean(), fluxes.std()
                if flux_std > 0:
                    fluxes = (fluxes - flux_mean) / flux_std

                n_obs = min(len(times), max_obs)
                times = times[:n_obs]
                fluxes = fluxes[:n_obs]

                # Pad
                times_padded = np.zeros(max_obs, dtype=np.float32)
                fluxes_padded = np.zeros(max_obs, dtype=np.float32)
                mask = np.ones(max_obs, dtype=np.float32)

                times_padded[:n_obs] = times
                fluxes_padded[:n_obs] = fluxes
                mask[:n_obs] = 0

                # Get embeddings
                input_data = {
                    'input': tf.constant(fluxes_padded.reshape(1, max_obs, 1)),
                    'times': tf.constant(times_padded.reshape(1, max_obs, 1)),
                    'mask_in': tf.constant(mask.reshape(1, max_obs, 1))
                }

                emb = encoder_layer(input_data).numpy()[0, :n_obs, :]  # (n_obs, 256)

                # Aggregate embeddings
                result[f'{band}_emb_mean'] = emb.mean(axis=0)
                result[f'{band}_emb_max'] = emb.max(axis=0)
                result[f'{band}_emb_std'] = emb.std(axis=0)

            all_results.append(result)

        return pd.DataFrame(all_results)

    def expand_embeddings_df(df):
        """Expand array columns into individual features."""
        expanded = {'object_id': df['object_id']}

        for col in df.columns:
            if col == 'object_id':
                continue
            arr = np.vstack(df[col].values)
            # Only keep first 32 components (PCA-like reduction)
            for i in range(min(32, arr.shape[1])):
                expanded[f'{col}_{i}'] = arr[:, i]

        return pd.DataFrame(expanded)

    print("   Extracting training embeddings...", flush=True)
    train_emb_raw = extract_embeddings_batch(train_lc, train_ids, "Train")
    train_embeddings = expand_embeddings_df(train_emb_raw)
    train_embeddings.to_pickle(train_emb_path)

    print("   Extracting test embeddings...", flush=True)
    test_emb_raw = extract_embeddings_batch(test_lc, test_ids, "Test")
    test_embeddings = expand_embeddings_df(test_emb_raw)
    test_embeddings.to_pickle(test_emb_path)

    print(f"   Saved embeddings: train={train_embeddings.shape}, test={test_embeddings.shape}", flush=True)

# ====================
# 3. COMBINE ALL FEATURES
# ====================
print("\n[3/5] Combining features...", flush=True)

# Get embedding columns
emb_cols = [c for c in train_embeddings.columns if c != 'object_id']

# Merge embeddings with existing features
train_combined = train_combined.merge(train_embeddings, on='object_id', how='left')
test_combined = test_combined.merge(test_embeddings, on='object_id', how='left')

# All feature columns
all_feature_cols = base_feature_cols + emb_cols
all_feature_cols = [c for c in all_feature_cols if c in train_combined.columns]

# Align indices
train_combined = train_combined.set_index('object_id').loc[train_ids].reset_index()
test_combined = test_combined.set_index('object_id').loc[test_ids].reset_index()

X = train_combined[all_feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
X_test = test_combined[all_feature_cols].values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

print(f"   Total features: {len(all_feature_cols)} (base={len(base_feature_cols)}, ASTROMER={len(emb_cols)})", flush=True)

# ====================
# 4. LOAD OPTUNA PARAMS
# ====================
print("\n[4/5] Loading Optuna parameters...", flush=True)

with open(base_path / 'data/processed/optuna_v20c_results.pkl', 'rb') as f:
    optuna_data = pickle.load(f)

xgb_params = optuna_data['xgb_best_params']
print(f"   XGB params loaded", flush=True)

# ====================
# 5. TRAIN XGBOOST (5-FOLD CV)
# ====================
print("\n[5/5] Training XGBoost (5-fold CV)...", flush=True)

n_neg, n_pos = np.sum(y == 0), np.sum(y == 1)
scale_pos_weight = n_neg / n_pos

n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(test_ids), n_splits))

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = xgb.XGBClassifier(
        **xgb_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds[:, fold] = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold
    best_f1 = 0
    for thresh in np.arange(0.03, 0.20, 0.01):
        f1 = f1_score(y_val, (oof_preds[val_idx] > thresh).astype(int))
        if f1 > best_f1:
            best_f1 = f1
    print(f"   Fold {fold+1}: F1 = {best_f1:.4f}", flush=True)

# Find optimal global threshold
best_global_thresh = 0.07
best_oof_f1 = 0
for thresh in np.arange(0.03, 0.20, 0.01):
    oof_f1 = f1_score(y, (oof_preds > thresh).astype(int))
    if oof_f1 > best_oof_f1:
        best_oof_f1 = oof_f1
        best_global_thresh = thresh

print(f"\n{'='*60}")
print(f"OOF F1 Score: {best_oof_f1:.4f} (threshold={best_global_thresh:.2f})")
print(f"{'='*60}")

# Comparison
baseline_f1 = 0.6708
improvement = (best_oof_f1 - baseline_f1) / baseline_f1 * 100
print(f"\nComparison with v21 baseline (OOF F1={baseline_f1:.4f}):")
if best_oof_f1 > baseline_f1:
    print(f"   IMPROVEMENT: +{improvement:.2f}%")
else:
    print(f"   Change: {improvement:+.2f}%")

# ====================
# GENERATE SUBMISSION
# ====================
print("\nGenerating submission...", flush=True)

test_probs_mean = test_preds.mean(axis=1)
test_binary = (test_probs_mean > best_global_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})
submission_path = base_path / 'submissions/submission_v26_astromer.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved to {submission_path}")
print(f"   Predicted TDEs: {test_binary.sum()} / {len(test_binary)}")

# Save cache
cache_path = base_path / 'data/processed/v26_astromer_cache.pkl'
with open(cache_path, 'wb') as f:
    pickle.dump({
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'best_threshold': best_global_thresh,
        'oof_f1': best_oof_f1
    }, f)
print(f"   Cache saved to {cache_path}")

print("\nDone!")
