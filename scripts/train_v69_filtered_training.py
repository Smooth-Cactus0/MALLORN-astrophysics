"""
MALLORN v69: AGN-Filtered Training

New strategy: Use AGN filter to CLEAN training data, not as a stage.

Problem with v68: MaxVar can't distinguish TDE from SN (both high MaxVar)
Solution: Use AGN filter only to identify "easy" AGN for exclusion

Approach:
1. Use MaxVar to identify high-confidence AGN in TRAINING data
2. Train v34a on transients only (exclude AGN)
3. At test time:
   - High-confidence AGN -> predict 0 (not TDE)
   - Others -> use v34a trained on cleaner data

This gives v34a a cleaner signal (TDE vs SN only, no AGN noise).
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

print("=" * 80, flush=True)
print("MALLORN v69: AGN-Filtered Training", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

spec_types = train_meta['SpecType'].values
is_agn_true = (spec_types == 'AGN').astype(int)

print(f"   Training: {len(train_ids)} total", flush=True)
print(f"   AGN: {np.sum(is_agn_true)}, Transients: {len(train_ids) - np.sum(is_agn_true)}", flush=True)
print(f"   TDEs: {np.sum(y)}", flush=True)

# ====================
# 2. LOAD ALL FEATURES
# ====================
print("\n2. Loading features...", flush=True)

# v34a features
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']

selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
importance_df = selection['importance_df']
high_corr_df = selection['high_corr_df']

corr_to_drop = set()
for _, row in high_corr_df.iterrows():
    if row['feature_1'] not in corr_to_drop:
        corr_to_drop.add(row['feature_2'])
clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
selected_120 = clean_features.head(120)['feature'].tolist()

tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']

train_base = train_base.merge(train_tde, on='object_id', how='left')
test_base = test_base.merge(test_tde, on='object_id', how='left')

with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

train_v21 = train_base[['object_id'] + selected_120].copy()
train_v21 = train_v21.merge(train_tde, on='object_id', how='left')
train_v21 = train_v21.merge(train_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

test_v21 = test_base[['object_id'] + selected_120].copy()
test_v21 = test_v21.merge(test_tde, on='object_id', how='left')
test_v21 = test_v21.merge(test_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

with open(base_path / 'data/processed/bazin_features_cache.pkl', 'rb') as f:
    bazin_cache = pickle.load(f)
train_bazin = bazin_cache['train']
test_bazin = bazin_cache['test']

train_v34a = train_v21.merge(train_bazin, on='object_id', how='left')
test_v34a = test_v21.merge(test_bazin, on='object_id', how='left')

feature_names = [c for c in train_v34a.columns if c != 'object_id']
print(f"   v34a features: {len(feature_names)}", flush=True)

X_train_full = train_v34a.drop(columns=['object_id']).values
X_test = test_v34a.drop(columns=['object_id']).values

# MaxVar for AGN detection
with open(base_path / 'data/processed/powerlaw_features_cache.pkl', 'rb') as f:
    pl_cache = pickle.load(f)
train_pl = pl_cache['train']
test_pl = pl_cache['test']

maxvar_cols = ['r_maxvar', 'maxvar_mean', 'maxvar_max', 'g_maxvar']
maxvar_cols = [c for c in maxvar_cols if c in train_pl.columns]

X_maxvar_train = train_pl[maxvar_cols].values
X_maxvar_test = test_pl[maxvar_cols].values
X_maxvar_train = np.nan_to_num(X_maxvar_train, nan=0)
X_maxvar_test = np.nan_to_num(X_maxvar_test, nan=0)

# ====================
# 3. TRAIN AGN DETECTOR
# ====================
print("\n3. Training AGN detector...", flush=True)

xgb_agn_params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'learning_rate': 0.05,
    'scale_pos_weight': len(is_agn_true[is_agn_true==0]) / len(is_agn_true[is_agn_true==1]),
    'random_state': 42
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_agn = np.zeros(len(X_maxvar_train))
test_agn = np.zeros(len(X_maxvar_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_maxvar_train, is_agn_true), 1):
    X_tr, X_val = X_maxvar_train[train_idx], X_maxvar_train[val_idx]
    y_tr, y_val = is_agn_true[train_idx], is_agn_true[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_maxvar_test)

    model = xgb.train(xgb_agn_params, dtrain, num_boost_round=100,
                      evals=[(dval, 'val')], early_stopping_rounds=20, verbose_eval=False)

    oof_agn[val_idx] = model.predict(dval)
    test_agn += model.predict(dtest) / n_folds

agn_acc = np.mean((oof_agn > 0.5).astype(int) == is_agn_true)
print(f"   AGN detection accuracy: {agn_acc:.3f}", flush=True)

# ====================
# 4. STRATEGY: TRAIN SEPARATE MODELS
# ====================
print("\n" + "=" * 80, flush=True)
print("TRAINING STRATEGIES", flush=True)
print("=" * 80, flush=True)

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
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

results = {}

# Strategy 1: v34a baseline (for comparison)
print("\n   Strategy 1: v34a baseline (all data)...", flush=True)
oof_base = np.zeros(len(y))
test_base_pred = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y), 1):
    X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    scale_pw = len(y_tr[y_tr==0]) / max(1, len(y_tr[y_tr==1]))

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)

    params = xgb_params.copy()
    params['scale_pos_weight'] = scale_pw

    model = xgb.train(params, dtrain, num_boost_round=500,
                      evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)

    oof_base[val_idx] = model.predict(dval)
    test_base_pred += model.predict(dtest) / n_folds

best_f1_base = 0
best_t_base = 0.1
for t in np.linspace(0.03, 0.3, 100):
    f1 = f1_score(y, (oof_base > t).astype(int))
    if f1 > best_f1_base:
        best_f1_base = f1
        best_t_base = t

results['baseline'] = {'oof': oof_base, 'test': test_base_pred, 'f1': best_f1_base, 'thresh': best_t_base}
print(f"      OOF F1: {best_f1_base:.4f}", flush=True)

# Strategy 2: Train on transients only
print("\n   Strategy 2: Train on transients only (exclude AGN)...", flush=True)

transient_mask = is_agn_true == 0
X_transient = X_train_full[transient_mask]
y_transient = y[transient_mask]

print(f"      Training on {len(y_transient)} transients ({np.sum(y_transient)} TDE)", flush=True)

oof_trans = np.zeros(len(y))
test_trans = np.zeros(len(X_test))

# Train on transients, but evaluate on all
for fold, (train_idx, val_idx) in enumerate(skf.split(X_transient, y_transient), 1):
    X_tr = X_transient[train_idx]
    y_tr = y_transient[train_idx]

    scale_pw = len(y_tr[y_tr==0]) / max(1, len(y_tr[y_tr==1]))

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    dfull = xgb.DMatrix(X_train_full, feature_names=feature_names)

    params = xgb_params.copy()
    params['scale_pos_weight'] = scale_pw

    model = xgb.train(params, dtrain, num_boost_round=300, verbose_eval=False)

    oof_trans += model.predict(dfull) / n_folds
    test_trans += model.predict(dtest) / n_folds

best_f1_trans = 0
best_t_trans = 0.1
for t in np.linspace(0.03, 0.5, 100):
    f1 = f1_score(y, (oof_trans > t).astype(int))
    if f1 > best_f1_trans:
        best_f1_trans = f1
        best_t_trans = t

results['transient_only'] = {'oof': oof_trans, 'test': test_trans, 'f1': best_f1_trans, 'thresh': best_t_trans}
print(f"      OOF F1: {best_f1_trans:.4f}", flush=True)

# Strategy 3: Hybrid - use AGN prob to gate
print("\n   Strategy 3: Baseline + AGN gating...", flush=True)

best_f1_hybrid = 0
best_params_hybrid = {}

for agn_thresh in [0.7, 0.8, 0.9]:
    for agn_action in ['zero', 'reduce']:
        if agn_action == 'zero':
            oof_hybrid = oof_base.copy()
            oof_hybrid[oof_agn > agn_thresh] = 0
        else:
            oof_hybrid = oof_base * (1 - 0.5 * (oof_agn > agn_thresh).astype(float))

        for t in np.linspace(0.03, 0.3, 50):
            f1 = f1_score(y, (oof_hybrid > t).astype(int))
            if f1 > best_f1_hybrid:
                best_f1_hybrid = f1
                best_params_hybrid = {'agn_thresh': agn_thresh, 'action': agn_action, 'tde_thresh': t}
                best_oof_hybrid = oof_hybrid.copy()

print(f"      OOF F1: {best_f1_hybrid:.4f}", flush=True)
print(f"      Params: {best_params_hybrid}", flush=True)

# Apply to test
if best_params_hybrid['action'] == 'zero':
    test_hybrid = test_base_pred.copy()
    test_hybrid[test_agn > best_params_hybrid['agn_thresh']] = 0
else:
    test_hybrid = test_base_pred * (1 - 0.5 * (test_agn > best_params_hybrid['agn_thresh']).astype(float))

results['hybrid'] = {'oof': best_oof_hybrid, 'test': test_hybrid, 'f1': best_f1_hybrid,
                     'thresh': best_params_hybrid['tde_thresh'], 'params': best_params_hybrid}

# ====================
# 5. BEST RESULT
# ====================
print("\n" + "=" * 80, flush=True)
print("RESULTS COMPARISON", flush=True)
print("=" * 80, flush=True)

best_strategy = max(results.keys(), key=lambda k: results[k]['f1'])
best = results[best_strategy]

print(f"\n   Strategy          OOF F1    Notes", flush=True)
print(f"   --------          ------    -----", flush=True)
for name, res in results.items():
    marker = " <-- BEST" if name == best_strategy else ""
    print(f"   {name:18s} {res['f1']:.4f}{marker}", flush=True)

print(f"\n   Best: {best_strategy} with F1={best['f1']:.4f}", flush=True)

# Confusion matrix
final_pred = (best['oof'] > best['thresh']).astype(int)
tp = np.sum((final_pred == 1) & (y == 1))
fp = np.sum((final_pred == 1) & (y == 0))
fn = np.sum((final_pred == 0) & (y == 1))
print(f"\n   TP={tp}, FP={fp}, FN={fn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}", flush=True)

# ====================
# 6. SUBMISSION
# ====================
print("\n" + "=" * 80, flush=True)
print("SUBMISSION", flush=True)
print("=" * 80, flush=True)

test_binary = (best['test'] > best['thresh']).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

submission_path = base_path / 'submissions/submission_v69_filtered.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_binary.sum()}", flush=True)

# Also save baseline submission for comparison
test_binary_base = (results['baseline']['test'] > results['baseline']['thresh']).astype(int)
sub_base = pd.DataFrame({'object_id': test_ids, 'target': test_binary_base})
sub_base.to_csv(base_path / 'submissions/submission_v69_baseline.csv', index=False)
print(f"   Also saved: submission_v69_baseline.csv (pure v34a retrain)", flush=True)

# Artifacts
with open(base_path / 'data/processed/v69_artifacts.pkl', 'wb') as f:
    pickle.dump({'results': results, 'best_strategy': best_strategy, 'oof_agn': oof_agn, 'test_agn': test_agn}, f)

print("\n" + "=" * 80, flush=True)
print(f"v69 Complete - Best: {best_strategy} (OOF F1={best['f1']:.4f})", flush=True)
print("=" * 80, flush=True)
