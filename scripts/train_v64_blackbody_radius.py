"""
MALLORN v64: Blackbody Radius Evolution Features

Adding R_bb evolution features to v34a Bazin baseline.

PHYSICS BASIS:
    Stefan-Boltzmann: L = 4πR²σT⁴
    Therefore: R_bb = sqrt(L / (4πσT⁴)) ∝ sqrt(L) / T²

KEY DISCRIMINATOR:
    - SUPERNOVAE: R_bb INCREASES initially (expanding ejecta)
      "Photospheric radius ALWAYS increases early on" (Piro & Nakar 2013)

    - TDEs: R_bb DECREASES from start or stays constant
      "Photospheric radius decays from the very beginning - essentially
       IMPOSSIBLE for it to be a supernova" (Perley et al. 2019)

    - AGN: Stochastic R_bb evolution

Features added (~50):
- R_bb at 6 epochs (peak, 10d, 20d, 30d, 50d, 100d)
- dR/dt rates (early, late, overall)
- Direction indicators (R_increasing_early, R_monotonic_decrease)
- R ratios (peak/50d, peak/100d)
- Temperature evolution (T_variance, T_constancy)
- Combined discriminating scores

Target: Beat v34a OOF F1=0.6667, LB F1=0.6907
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v64: Blackbody Radius Evolution Features", flush=True)
print("=" * 80, flush=True)
print("Adding R_bb physics to v34a Bazin baseline", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD DATA & v34a FEATURES
# ====================
print("\n1. Loading data and v34a features...", flush=True)

from utils.data_loader import load_all_data

data = load_all_data()
train_meta = data['train_meta']
test_meta = data['test_meta']
train_lc = data['train_lc']
test_lc = data['test_lc']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDEs)", flush=True)
print(f"   Test: {len(test_ids)} objects", flush=True)

# Load v34a feature pipeline (same as v34a_bazin.py)
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']

# Load feature selection
selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
importance_df = selection['importance_df']
high_corr_df = selection['high_corr_df']

corr_to_drop = set()
for _, row in high_corr_df.iterrows():
    if row['feature_1'] not in corr_to_drop:
        corr_to_drop.add(row['feature_2'])
clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
selected_120 = clean_features.head(120)['feature'].tolist()

# TDE physics features
tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']
tde_cols = [c for c in train_tde.columns if c != 'object_id']

train_base = train_base.merge(train_tde, on='object_id', how='left')
test_base = test_base.merge(test_tde, on='object_id', how='left')

# GP2D features
with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

# Combine base features (v21 feature set)
train_v21 = train_base[['object_id'] + selected_120].copy()
train_v21 = train_v21.merge(train_tde, on='object_id', how='left')
train_v21 = train_v21.merge(train_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

test_v21 = test_base[['object_id'] + selected_120].copy()
test_v21 = test_v21.merge(test_tde, on='object_id', how='left')
test_v21 = test_v21.merge(test_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

print(f"   v21 base features: {len(train_v21.columns) - 1}", flush=True)

# Load Bazin features
from features.bazin_fitting import extract_bazin_features

# Check if cached
bazin_cache_path = base_path / 'data/processed/bazin_features_cache.pkl'
if bazin_cache_path.exists():
    print("   Loading cached Bazin features...", flush=True)
    with open(bazin_cache_path, 'rb') as f:
        bazin_cache = pickle.load(f)
    train_bazin = bazin_cache['train']
    test_bazin = bazin_cache['test']
else:
    print("   Extracting Bazin features (this takes a few minutes)...", flush=True)
    train_bazin = extract_bazin_features(train_lc, train_ids)
    test_bazin = extract_bazin_features(test_lc, test_ids)
    with open(bazin_cache_path, 'wb') as f:
        pickle.dump({'train': train_bazin, 'test': test_bazin}, f)

print(f"   Bazin features: {len(train_bazin.columns) - 1}", flush=True)

# ====================
# 2. EXTRACT R_bb FEATURES
# ====================
print("\n2. Extracting Blackbody Radius (R_bb) features...", flush=True)

from features.blackbody_radius import extract_radius_features

# Check for cache
rbb_cache_path = base_path / 'data/processed/rbb_features_cache.pkl'
if rbb_cache_path.exists():
    print("   Loading cached R_bb features...", flush=True)
    with open(rbb_cache_path, 'rb') as f:
        rbb_cache = pickle.load(f)
    train_rbb = rbb_cache['train']
    test_rbb = rbb_cache['test']
else:
    print("   Training set R_bb extraction...", flush=True)
    train_rbb = extract_radius_features(train_lc, train_ids)

    print("   Test set R_bb extraction...", flush=True)
    test_rbb = extract_radius_features(test_lc, test_ids)

    # Cache for future use
    with open(rbb_cache_path, 'wb') as f:
        pickle.dump({'train': train_rbb, 'test': test_rbb}, f)
    print("   R_bb features cached.", flush=True)

rbb_cols = [c for c in train_rbb.columns if c != 'object_id']
print(f"   Extracted {len(rbb_cols)} R_bb features", flush=True)

# Check coverage
rbb_peak_coverage = train_rbb['R_bb_peak'].notna().sum() / len(train_rbb)
print(f"   R_bb_peak coverage: {100*rbb_peak_coverage:.1f}%", flush=True)

# Show key features
print("\n   Key R_bb features:", flush=True)
key_features = ['R_bb_peak', 'dRdt_early', 'R_increasing_early',
                'R_monotonic_decrease', 'T_constancy']
for feat in key_features:
    if feat in train_rbb.columns:
        coverage = train_rbb[feat].notna().sum() / len(train_rbb)
        print(f"      {feat}: {100*coverage:.1f}% coverage", flush=True)

# ====================
# 3. COMBINE ALL FEATURES
# ====================
print("\n3. Combining v21 + Bazin + R_bb features...", flush=True)

# Start with v21 + Bazin (v34a baseline)
train_v34a = train_v21.merge(train_bazin, on='object_id', how='left')
test_v34a = test_v21.merge(test_bazin, on='object_id', how='left')

# Add R_bb features
train_combined = train_v34a.merge(train_rbb, on='object_id', how='left')
test_combined = test_v34a.merge(test_rbb, on='object_id', how='left')

# Get feature columns
feature_names = [c for c in train_combined.columns if c != 'object_id']

print(f"   v21 features: {len(train_v21.columns) - 1}", flush=True)
print(f"   + Bazin: {len(train_bazin.columns) - 1}", flush=True)
print(f"   + R_bb: {len(rbb_cols)}", flush=True)
print(f"   = Total: {len(feature_names)} features", flush=True)

# Prepare numpy arrays
X_train = train_combined.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values

print(f"\n   Train shape: {X_train.shape}", flush=True)
print(f"   Test shape: {X_test.shape}", flush=True)

# Handle any remaining inf/nan
X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# ====================
# 4. TRAIN XGBOOST (v34a parameters)
# ====================
print("\n4. Training XGBoost with 5-fold CV...", flush=True)

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.025,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros((len(X_test), n_folds))
feature_importance = np.zeros(len(feature_names))
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    oof_preds[val_idx] = model.predict(dval)
    test_preds[:, fold-1] = model.predict(dtest)

    # Feature importance
    importance = model.get_score(importance_type='gain')
    for feat, gain in importance.items():
        if feat in feature_names:
            idx = feature_names.index(feat)
            feature_importance[idx] += gain

    # Fold metrics
    best_f1 = 0
    best_thresh = 0.5
    for t in np.linspace(0.03, 0.3, 50):
        preds_binary = (oof_preds[val_idx] > t).astype(int)
        f1 = f1_score(y_val, preds_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    fold_f1s.append(best_f1)
    print(f"      Fold F1: {best_f1:.4f} @ threshold={best_thresh:.3f}", flush=True)

# ====================
# 5. OVERALL RESULTS
# ====================
print("\n" + "=" * 80, flush=True)
print("CROSS-VALIDATION RESULTS", flush=True)
print("=" * 80, flush=True)

# Find optimal threshold
best_f1 = 0
best_thresh = 0.1
for t in np.linspace(0.03, 0.3, 200):
    preds_binary = (oof_preds > t).astype(int)
    f1 = f1_score(y, preds_binary)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}", flush=True)
print(f"   Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}", flush=True)
print(f"   Mean Fold F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}", flush=True)

# Confusion matrix
final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
tn = np.sum((final_preds == 0) & (y == 0))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\n   Confusion Matrix:", flush=True)
print(f"      TP={tp}, FP={fp}", flush=True)
print(f"      FN={fn}, TN={tn}", flush=True)
print(f"   Precision: {precision:.4f}", flush=True)
print(f"   Recall: {recall:.4f}", flush=True)

# ====================
# 6. FEATURE IMPORTANCE ANALYSIS
# ====================
print("\n" + "=" * 80, flush=True)
print("FEATURE IMPORTANCE ANALYSIS", flush=True)
print("=" * 80, flush=True)

feature_importance = feature_importance / n_folds
importance_df_result = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n   Top 30 Features:", flush=True)
print(importance_df_result.head(30).to_string(index=False), flush=True)

# Analyze R_bb feature contributions
print("\n   R_bb Feature Analysis:", flush=True)
rbb_importance = importance_df_result[importance_df_result['feature'].isin(rbb_cols)]
total_importance = importance_df_result['importance'].sum()

if len(rbb_importance) > 0 and total_importance > 0:
    rbb_total = rbb_importance['importance'].sum()
    print(f"      R_bb features used: {len(rbb_importance)} / {len(rbb_cols)}", flush=True)
    print(f"      R_bb importance share: {100*rbb_total/total_importance:.1f}%", flush=True)

    print("\n   Top 10 R_bb Features:", flush=True)
    for idx, row in rbb_importance.head(10).iterrows():
        rank = list(importance_df_result['feature']).index(row['feature']) + 1
        print(f"      {rank:3d}. {row['feature']:35s} {row['importance']:10.1f}", flush=True)
else:
    print("      No R_bb features selected by model", flush=True)

# ====================
# 7. CREATE SUBMISSION
# ====================
print("\n" + "=" * 80, flush=True)
print("GENERATING SUBMISSION", flush=True)
print("=" * 80, flush=True)

test_avg = test_preds.mean(axis=1)
test_final = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v64_blackbody_radius.csv'
submission.to_csv(submission_path, index=False)

print(f"   Submission saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_final.sum()} / {len(test_final)}", flush=True)

# Save artifacts
artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'feature_importance': importance_df_result,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'feature_names': feature_names,
    'rbb_features': rbb_cols,
    'fold_f1s': fold_f1s
}

with open(base_path / 'data/processed/v64_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# ====================
# 8. COMPARISON WITH v34a
# ====================
print("\n" + "=" * 80, flush=True)
print("COMPARISON WITH v34a BASELINE", flush=True)
print("=" * 80, flush=True)

v34a_oof_f1 = 0.6667  # From v34a results
v34a_lb_f1 = 0.6907   # From Kaggle LB

improvement_oof = (best_f1 - v34a_oof_f1) / v34a_oof_f1 * 100
improvement_abs = best_f1 - v34a_oof_f1

print(f"\n   v34a Baseline:", flush=True)
print(f"      OOF F1: {v34a_oof_f1:.4f}", flush=True)
print(f"      LB F1:  {v34a_lb_f1:.4f}", flush=True)

print(f"\n   v64 (+ R_bb):", flush=True)
print(f"      OOF F1: {best_f1:.4f}", flush=True)
print(f"      Change: {improvement_oof:+.2f}% ({improvement_abs:+.4f})", flush=True)

if best_f1 > v34a_oof_f1:
    expected_lb = v34a_lb_f1 * (best_f1 / v34a_oof_f1)
    print(f"\n   Expected LB F1: ~{expected_lb:.4f}", flush=True)
    print("\n   SUCCESS: R_bb features improved performance!", flush=True)
    print("   Physics hypothesis validated - R_bb evolution discriminates TDEs!", flush=True)
else:
    print("\n   R_bb features did not improve OOF F1", flush=True)
    print("   Possible reasons:", flush=True)
    print("     - Insufficient coverage (need more multi-band observations)", flush=True)
    print("     - Temperature fitting unreliable with sparse data", flush=True)
    print("     - Information already captured by existing color features", flush=True)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v64 Complete", flush=True)
print("=" * 80, flush=True)
