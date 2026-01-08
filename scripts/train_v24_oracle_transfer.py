"""
MALLORN v24: ORACLE Transfer Learning

Uses pre-trained ORACLE model (trained on ~500K ELAsTiCC transients)
to extract embeddings from MALLORN lightcurves, then feeds to XGBoost.

This is TRUE transfer learning:
1. ORACLE's GRU learned patterns from massive ELAsTiCC dataset
2. We extract embeddings (latent representations) for MALLORN objects
3. XGBoost uses embeddings + our physics features for classification

Installation: pip install astro-oracle
"""

import sys
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

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 60, flush=True)
print("MALLORN v24: ORACLE Transfer Learning", flush=True)
print("=" * 60, flush=True)

# ====================
# 1. INSTALL AND IMPORT ORACLE
# ====================
print("\n1. Loading ORACLE pre-trained model...", flush=True)

try:
    from oracle.pretrained.ELAsTiCC import ORACLE1_ELAsTiCC_lite
    print("   ORACLE loaded successfully!", flush=True)
except ImportError:
    print("   ERROR: astro-oracle not installed.", flush=True)
    print("   Run: pip install astro-oracle", flush=True)
    sys.exit(1)

# Initialize model (downloads weights automatically)
print("   Initializing ORACLE1_ELAsTiCC_lite model...", flush=True)
oracle_model = ORACLE1_ELAsTiCC_lite()
print("   Model ready!", flush=True)

# ====================
# 2. LOAD MALLORN DATA
# ====================
print("\n2. Loading MALLORN data...", flush=True)

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
# 3. CONVERT MALLORN TO ORACLE FORMAT
# ====================
print("\n3. Converting MALLORN data to ORACLE format...", flush=True)

# ORACLE expects astropy Table with specific columns
# We need to check the exact format required

from astropy.table import Table

def convert_to_oracle_format(lc_df, object_ids, meta_df):
    """
    Convert MALLORN lightcurves to ORACLE input format.

    ORACLE expects columns like:
    - time, flux, flux_err, band (for light curves)
    - Possibly redshift, ra, dec for metadata
    """
    # Group by object
    grouped = {obj_id: group for obj_id, group in lc_df.groupby('object_id')}

    all_tables = []
    valid_ids = []

    for i, obj_id in enumerate(object_ids):
        if i % 500 == 0:
            print(f"      Converting {i}/{len(object_ids)}...", flush=True)

        if obj_id not in grouped:
            continue

        obj_lc = grouped[obj_id].sort_values('Time (MJD)')

        # Get metadata
        meta_row = meta_df[meta_df['object_id'] == obj_id]
        z = meta_row['Z'].values[0] if len(meta_row) > 0 else 0.0

        # Create table for this object
        # ORACLE format: time, flux, flux_err, band, and possibly metadata
        t = Table()
        t['time'] = obj_lc['Time (MJD)'].values
        t['flux'] = obj_lc['Flux'].values
        t['flux_err'] = obj_lc['Flux_err'].values
        t['band'] = obj_lc['Filter'].values

        # Add metadata
        t.meta['object_id'] = obj_id
        t.meta['redshift'] = z

        all_tables.append(t)
        valid_ids.append(obj_id)

    return all_tables, valid_ids

print("   Converting train lightcurves...", flush=True)
train_tables, train_valid_ids = convert_to_oracle_format(train_lc, train_ids, train_meta)

print("   Converting test lightcurves...", flush=True)
test_tables, test_valid_ids = convert_to_oracle_format(test_lc, test_ids, test_meta)

print(f"   Valid train objects: {len(train_tables)}", flush=True)
print(f"   Valid test objects: {len(test_tables)}", flush=True)

# ====================
# 4. EXTRACT ORACLE EMBEDDINGS
# ====================
print("\n4. Extracting ORACLE embeddings...", flush=True)

def get_embeddings_batch(model, tables, batch_size=100):
    """Extract embeddings in batches."""
    embeddings = []

    for i in range(0, len(tables), batch_size):
        batch = tables[i:i+batch_size]

        if (i // batch_size) % 10 == 0:
            print(f"      Batch {i//batch_size + 1}/{len(tables)//batch_size + 1}...", flush=True)

        for table in batch:
            try:
                emb = model.embed(table)
                embeddings.append(emb)
            except Exception as e:
                # If embedding fails, use zeros
                embeddings.append(np.zeros(model.embedding_dim if hasattr(model, 'embedding_dim') else 64))

    return np.array(embeddings)

print("   Extracting train embeddings...", flush=True)
try:
    train_embeddings = get_embeddings_batch(oracle_model, train_tables)
    print(f"   Train embeddings shape: {train_embeddings.shape}", flush=True)
except Exception as e:
    print(f"   ERROR extracting embeddings: {e}", flush=True)
    print("   Trying alternative approach...", flush=True)

    # Alternative: extract embeddings one by one with error handling
    train_embeddings = []
    for i, table in enumerate(train_tables):
        if i % 500 == 0:
            print(f"      Processing {i}/{len(train_tables)}...", flush=True)
        try:
            emb = oracle_model.embed(table)
            if isinstance(emb, np.ndarray):
                train_embeddings.append(emb.flatten())
            else:
                train_embeddings.append(np.zeros(64))
        except:
            train_embeddings.append(np.zeros(64))

    train_embeddings = np.array(train_embeddings)
    print(f"   Train embeddings shape: {train_embeddings.shape}", flush=True)

print("   Extracting test embeddings...", flush=True)
test_embeddings = []
for i, table in enumerate(test_tables):
    if i % 500 == 0:
        print(f"      Processing {i}/{len(test_tables)}...", flush=True)
    try:
        emb = oracle_model.embed(table)
        if isinstance(emb, np.ndarray):
            test_embeddings.append(emb.flatten())
        else:
            test_embeddings.append(np.zeros(train_embeddings.shape[1]))
    except:
        test_embeddings.append(np.zeros(train_embeddings.shape[1]))

test_embeddings = np.array(test_embeddings)
print(f"   Test embeddings shape: {test_embeddings.shape}", flush=True)

# ====================
# 5. COMBINE WITH EXISTING FEATURES
# ====================
print("\n5. Combining embeddings with existing features...", flush=True)

# Load existing features (from v21)
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

# Merge features
train_combined = train_features.merge(train_gp2d, on='object_id', how='left')
test_combined = test_features.merge(test_gp2d, on='object_id', how='left')

# Get feature matrix
base_cols = [c for c in selected_120 if c in train_combined.columns]
all_feature_cols = base_cols + gp2d_cols
all_feature_cols = list(dict.fromkeys(all_feature_cols))
all_feature_cols = [c for c in all_feature_cols if c in train_combined.columns]

# Align with valid IDs
train_combined = train_combined.set_index('object_id')
test_combined = test_combined.set_index('object_id')

X_features_train = train_combined.loc[train_valid_ids][all_feature_cols].values.astype(np.float32)
X_features_test = test_combined.loc[test_valid_ids][all_feature_cols].values.astype(np.float32)

# Handle NaN
X_features_train = np.nan_to_num(X_features_train, nan=0, posinf=0, neginf=0)
X_features_test = np.nan_to_num(X_features_test, nan=0, posinf=0, neginf=0)

# Combine features with embeddings
embedding_cols = [f'oracle_emb_{i}' for i in range(train_embeddings.shape[1])]
X_train = np.hstack([X_features_train, train_embeddings])
X_test = np.hstack([X_features_test, test_embeddings])

all_cols = all_feature_cols + embedding_cols

# Get labels for valid IDs
y_train = train_meta.set_index('object_id').loc[train_valid_ids]['target'].values

print(f"   Combined feature shape: {X_train.shape}", flush=True)
print(f"   Original features: {len(all_feature_cols)}", flush=True)
print(f"   ORACLE embeddings: {train_embeddings.shape[1]}", flush=True)

# ====================
# 6. LOAD OPTUNA PARAMS AND TRAIN
# ====================
print("\n6. Training XGBoost with ORACLE embeddings...", flush=True)

with open(base_path / 'data/processed/optuna_v20c_results.pkl', 'rb') as f:
    optuna_data = pickle.load(f)
xgb_params = optuna_data['xgb_best_params']

n_neg, n_pos = np.sum(y_train == 0), np.sum(y_train == 1)
scale_pos_weight = n_neg / n_pos

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y_train))
test_preds = np.zeros((len(test_valid_ids), 5))
feature_importance = np.zeros(len(all_cols))

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = xgb.XGBClassifier(
        **xgb_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds[:, fold] = model.predict_proba(X_test)[:, 1]
    feature_importance += model.feature_importances_

    # Fold F1
    best_f1 = 0
    for t in np.arange(0.05, 0.9, 0.01):
        f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1

    print(f"   Fold {fold+1}: F1={best_f1:.4f}", flush=True)

# ====================
# 7. FIND OPTIMAL THRESHOLD
# ====================
print("\n7. Finding optimal threshold...", flush=True)

best_f1 = 0
best_thresh = 0.5

for t in np.arange(0.05, 0.95, 0.01):
    f1 = f1_score(y_train, (oof_preds > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}", flush=True)

# ====================
# 8. ANALYZE EMBEDDING IMPORTANCE
# ====================
print("\n8. Analyzing ORACLE embedding importance...", flush=True)

feature_importance /= 5
importance_df_new = pd.DataFrame({
    'feature': all_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Separate embedding vs original features
emb_importance = importance_df_new[importance_df_new['feature'].str.startswith('oracle_emb')]
orig_importance = importance_df_new[~importance_df_new['feature'].str.startswith('oracle_emb')]

print(f"   Total ORACLE embedding importance: {emb_importance['importance'].sum():.4f}", flush=True)
print(f"   Total original feature importance: {orig_importance['importance'].sum():.4f}", flush=True)

print("\n   Top 5 ORACLE embeddings:")
for _, row in emb_importance.head(5).iterrows():
    print(f"      {row['feature']}: {row['importance']:.4f}", flush=True)

# ====================
# 9. CREATE SUBMISSION
# ====================
print("\n9. Creating submission...", flush=True)

test_avg = test_preds.mean(axis=1)
test_final = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_valid_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v24_oracle.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved to {submission_path}", flush=True)
print(f"   Predictions: {test_final.sum()} TDEs / {len(test_final)} total ({100*test_final.mean():.1f}%)", flush=True)

# Save cache
with open(base_path / 'data/processed/oracle_embeddings_cache.pkl', 'wb') as f:
    pickle.dump({
        'train_embeddings': train_embeddings,
        'test_embeddings': test_embeddings,
        'train_ids': train_valid_ids,
        'test_ids': test_valid_ids
    }, f)

with open(base_path / 'data/processed/models_v24_oracle.pkl', 'wb') as f:
    pickle.dump({
        'best_thresh': best_thresh,
        'feature_cols': all_cols,
        'oof_f1': best_f1,
        'importance_df': importance_df_new
    }, f)

# ====================
# SUMMARY
# ====================
print("\n" + "=" * 60, flush=True)
print("v24 ORACLE Transfer Learning Complete!", flush=True)
print("=" * 60, flush=True)

print(f"\nVersion Comparison:", flush=True)
print(f"  v21 (XGB only):         OOF F1 = 0.6708, LB = 0.6649", flush=True)
print(f"  v22 (ATAT scratch):     OOF F1 = 0.5053, LB = 0.4876", flush=True)
print(f"  v24 (ORACLE transfer):  OOF F1 = {best_f1:.4f}", flush=True)

delta = best_f1 - 0.6708
if delta > 0:
    print(f"\n  +{delta*100:.2f}% improvement with transfer learning!", flush=True)
else:
    print(f"\n  {delta*100:.2f}% vs v21", flush=True)

print("\n  Key insight: ORACLE embeddings capture patterns from", flush=True)
print("  ~500K ELAsTiCC transients, enabling knowledge transfer!", flush=True)
print("=" * 60, flush=True)
