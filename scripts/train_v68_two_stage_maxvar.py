"""
MALLORN v68: Two-Stage Classification with MaxVar

Strategy:
  Stage 1: AGN Filter using MaxVar features
           - AGN: Stochastic variability -> Lower MaxVar
           - Transients (TDE/SN): Dramatic brightening -> Higher MaxVar

  Stage 2: TDE Classification on non-AGN candidates
           - Blend v34a predictions with MaxVar-only model
           - v34a excels at TDE vs SN (Bazin, colors, decay)
           - MaxVar adds transient amplitude signal

This separates concerns:
- MaxVar doesn't need to distinguish TDE from SN (hard)
- v34a doesn't need to handle AGN noise (confusing)
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v68: Two-Stage MaxVar Classification", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_lc = data['train_lc']
test_lc = data['test_lc']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values  # Binary: TDE or not

# Get SpecType for analysis
spec_types = train_meta['SpecType'].values
print(f"\n   Training distribution:", flush=True)
for st in np.unique(spec_types):
    count = np.sum(spec_types == st)
    tde_count = np.sum((spec_types == st) & (y == 1))
    print(f"      {st:15s}: {count:4d} ({tde_count} TDE)", flush=True)

# Create AGN label (for Stage 1)
is_agn = (spec_types == 'AGN').astype(int)
print(f"\n   AGN count: {np.sum(is_agn)} / {len(is_agn)}", flush=True)
print(f"   Transients: {len(is_agn) - np.sum(is_agn)}", flush=True)

# ====================
# 2. LOAD FEATURES
# ====================
print("\n2. Loading features...", flush=True)

# v34a features (full pipeline)
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

# Bazin
with open(base_path / 'data/processed/bazin_features_cache.pkl', 'rb') as f:
    bazin_cache = pickle.load(f)
train_bazin = bazin_cache['train']
test_bazin = bazin_cache['test']

train_v34a = train_v21.merge(train_bazin, on='object_id', how='left')
test_v34a = test_v21.merge(test_bazin, on='object_id', how='left')

v34a_features = [c for c in train_v34a.columns if c != 'object_id']
print(f"   v34a features: {len(v34a_features)}", flush=True)

# MaxVar features
with open(base_path / 'data/processed/powerlaw_features_cache.pkl', 'rb') as f:
    pl_cache = pickle.load(f)
train_pl = pl_cache['train']
test_pl = pl_cache['test']

maxvar_cols = ['r_maxvar', 'maxvar_mean', 'maxvar_max', 'g_maxvar', 'i_maxvar',
               'r_late_frac', 'r_power_exponent', 'tde_decay_score']
maxvar_cols = [c for c in maxvar_cols if c in train_pl.columns]
print(f"   MaxVar features: {len(maxvar_cols)}", flush=True)

# ====================
# 3. STAGE 1: AGN FILTER
# ====================
print("\n" + "=" * 80, flush=True)
print("STAGE 1: AGN FILTER (using MaxVar)", flush=True)
print("=" * 80, flush=True)

# Use MaxVar features to identify AGN
# AGN have lower MaxVar (stochastic) vs transients (dramatic brightening)
train_stage1 = train_pl[['object_id'] + maxvar_cols].copy()
test_stage1 = test_pl[['object_id'] + maxvar_cols].copy()

X_stage1 = train_stage1.drop(columns=['object_id']).values
X_stage1_test = test_stage1.drop(columns=['object_id']).values

X_stage1 = np.nan_to_num(X_stage1, nan=0, posinf=1e10, neginf=-1e10)
X_stage1_test = np.nan_to_num(X_stage1_test, nan=0, posinf=1e10, neginf=-1e10)

# Train AGN classifier
xgb_agn_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': len(is_agn[is_agn==0]) / len(is_agn[is_agn==1]),
    'random_state': 42,
    'n_jobs': -1
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_agn_prob = np.zeros(len(X_stage1))
test_agn_prob = np.zeros(len(X_stage1_test))

print("\n   Training AGN classifier...", flush=True)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_stage1, is_agn), 1):
    X_tr, X_val = X_stage1[train_idx], X_stage1[val_idx]
    y_tr, y_val = is_agn[train_idx], is_agn[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=maxvar_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=maxvar_cols)
    dtest = xgb.DMatrix(X_stage1_test, feature_names=maxvar_cols)

    model = xgb.train(
        xgb_agn_params,
        dtrain,
        num_boost_round=200,
        evals=[(dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    oof_agn_prob[val_idx] = model.predict(dval)
    test_agn_prob += model.predict(dtest) / n_folds

    # Per-fold metrics
    agn_pred = (oof_agn_prob[val_idx] > 0.5).astype(int)
    acc = np.mean(agn_pred == y_val)
    print(f"      Fold {fold}: AGN detection accuracy = {acc:.3f}", flush=True)

# Overall AGN detection performance
print(f"\n   AGN Detection Results:", flush=True)
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    agn_pred = (oof_agn_prob > thresh).astype(int)
    tp_agn = np.sum((agn_pred == 1) & (is_agn == 1))
    fp_agn = np.sum((agn_pred == 1) & (is_agn == 0))
    fn_agn = np.sum((agn_pred == 0) & (is_agn == 1))
    prec = tp_agn / (tp_agn + fp_agn) if (tp_agn + fp_agn) > 0 else 0
    rec = tp_agn / (tp_agn + fn_agn) if (tp_agn + fn_agn) > 0 else 0
    print(f"      thresh={thresh}: Precision={prec:.3f}, Recall={rec:.3f}", flush=True)

# ====================
# 4. STAGE 2: TDE CLASSIFICATION
# ====================
print("\n" + "=" * 80, flush=True)
print("STAGE 2: TDE CLASSIFICATION (v34a + MaxVar ensemble)", flush=True)
print("=" * 80, flush=True)

# Load v34a OOF predictions
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a_artifacts = pickle.load(f)

v34a_oof = v34a_artifacts['oof_preds']
v34a_test = v34a_artifacts['test_preds']
print(f"   v34a OOF F1: {v34a_artifacts['oof_f1']:.4f}", flush=True)

# Train MaxVar-only TDE classifier
print("\n   Training MaxVar-only TDE classifier...", flush=True)

xgb_maxvar_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 4,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'random_state': 42,
    'n_jobs': -1
}

oof_maxvar_tde = np.zeros(len(X_stage1))
test_maxvar_tde = np.zeros(len(X_stage1_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_stage1, y), 1):
    X_tr, X_val = X_stage1[train_idx], X_stage1[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=maxvar_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=maxvar_cols)
    dtest = xgb.DMatrix(X_stage1_test, feature_names=maxvar_cols)

    model = xgb.train(
        xgb_maxvar_params,
        dtrain,
        num_boost_round=200,
        evals=[(dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    oof_maxvar_tde[val_idx] = model.predict(dval)
    test_maxvar_tde += model.predict(dtest) / n_folds

# MaxVar-only performance
best_f1_maxvar = 0
for t in np.linspace(0.05, 0.5, 50):
    pred = (oof_maxvar_tde > t).astype(int)
    f1 = f1_score(y, pred)
    if f1 > best_f1_maxvar:
        best_f1_maxvar = f1
print(f"   MaxVar-only OOF F1: {best_f1_maxvar:.4f}", flush=True)

# ====================
# 5. TWO-STAGE ENSEMBLE
# ====================
print("\n" + "=" * 80, flush=True)
print("TWO-STAGE ENSEMBLE", flush=True)
print("=" * 80, flush=True)

# Strategy:
# - If AGN probability is high, predict NOT TDE
# - Otherwise, blend v34a and MaxVar predictions

print("\n   Testing ensemble strategies...", flush=True)

best_result = {'f1': 0, 'strategy': '', 'params': {}}

# Strategy A: Soft AGN gating
print("\n   Strategy A: Soft AGN gating", flush=True)
for agn_weight in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for v34a_weight in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        maxvar_weight = 1 - v34a_weight

        # Blend TDE predictions
        blended_tde = v34a_weight * v34a_oof + maxvar_weight * oof_maxvar_tde

        # Apply AGN penalty
        final_prob = blended_tde * (1 - agn_weight * oof_agn_prob)

        for thresh in np.linspace(0.05, 0.3, 30):
            pred = (final_prob > thresh).astype(int)
            f1 = f1_score(y, pred)

            if f1 > best_result['f1']:
                best_result['f1'] = f1
                best_result['strategy'] = 'soft_agn_gate'
                best_result['params'] = {
                    'agn_weight': agn_weight,
                    'v34a_weight': v34a_weight,
                    'threshold': thresh
                }
                best_result['oof_prob'] = final_prob.copy()

print(f"      Best: F1={best_result['f1']:.4f}", flush=True)
print(f"      Params: {best_result['params']}", flush=True)

# Strategy B: Hard AGN filter then blend
print("\n   Strategy B: Hard AGN filter then blend", flush=True)
for agn_thresh in [0.5, 0.6, 0.7, 0.8]:
    is_agn_pred = oof_agn_prob > agn_thresh

    for v34a_weight in [0.6, 0.7, 0.8, 0.9, 1.0]:
        maxvar_weight = 1 - v34a_weight

        # Blend only for non-AGN
        blended_tde = v34a_weight * v34a_oof + maxvar_weight * oof_maxvar_tde

        # Set AGN to 0 probability
        final_prob = blended_tde.copy()
        final_prob[is_agn_pred] = 0

        for thresh in np.linspace(0.05, 0.3, 30):
            pred = (final_prob > thresh).astype(int)
            f1 = f1_score(y, pred)

            if f1 > best_result['f1']:
                best_result['f1'] = f1
                best_result['strategy'] = 'hard_agn_filter'
                best_result['params'] = {
                    'agn_thresh': agn_thresh,
                    'v34a_weight': v34a_weight,
                    'threshold': thresh
                }
                best_result['oof_prob'] = final_prob.copy()

print(f"      Best: F1={best_result['f1']:.4f}", flush=True)
print(f"      Params: {best_result['params']}", flush=True)

# Strategy C: v34a only with AGN penalty
print("\n   Strategy C: v34a only + AGN penalty", flush=True)
for agn_weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    final_prob = v34a_oof * (1 - agn_weight * oof_agn_prob)

    for thresh in np.linspace(0.05, 0.3, 30):
        pred = (final_prob > thresh).astype(int)
        f1 = f1_score(y, pred)

        if f1 > best_result['f1']:
            best_result['f1'] = f1
            best_result['strategy'] = 'v34a_agn_penalty'
            best_result['params'] = {
                'agn_weight': agn_weight,
                'threshold': thresh
            }
            best_result['oof_prob'] = final_prob.copy()

print(f"      Best: F1={best_result['f1']:.4f}", flush=True)
print(f"      Params: {best_result['params']}", flush=True)

# ====================
# 6. FINAL RESULTS
# ====================
print("\n" + "=" * 80, flush=True)
print("FINAL RESULTS", flush=True)
print("=" * 80, flush=True)

print(f"\n   Best Strategy: {best_result['strategy']}", flush=True)
print(f"   Best Params: {best_result['params']}", flush=True)
print(f"   OOF F1: {best_result['f1']:.4f}", flush=True)

# Apply best strategy to test
params = best_result['params']

if best_result['strategy'] == 'soft_agn_gate':
    agn_w = params['agn_weight']
    v34a_w = params['v34a_weight']
    maxvar_w = 1 - v34a_w
    blended = v34a_w * v34a_test + maxvar_w * test_maxvar_tde
    test_final_prob = blended * (1 - agn_w * test_agn_prob)

elif best_result['strategy'] == 'hard_agn_filter':
    agn_t = params['agn_thresh']
    v34a_w = params['v34a_weight']
    maxvar_w = 1 - v34a_w
    is_agn_test = test_agn_prob > agn_t
    blended = v34a_w * v34a_test + maxvar_w * test_maxvar_tde
    test_final_prob = blended.copy()
    test_final_prob[is_agn_test] = 0

elif best_result['strategy'] == 'v34a_agn_penalty':
    agn_w = params['agn_weight']
    test_final_prob = v34a_test * (1 - agn_w * test_agn_prob)

thresh = params['threshold']
test_binary = (test_final_prob > thresh).astype(int)

# Confusion matrix on OOF
final_oof = (best_result['oof_prob'] > thresh).astype(int)
tp = np.sum((final_oof == 1) & (y == 1))
fp = np.sum((final_oof == 1) & (y == 0))
fn = np.sum((final_oof == 0) & (y == 1))
tn = np.sum((final_oof == 0) & (y == 0))

print(f"\n   Confusion Matrix:", flush=True)
print(f"      TP={tp}, FP={fp}", flush=True)
print(f"      FN={fn}, TN={tn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}", flush=True)
print(f"   Recall: {tp/(tp+fn):.4f}", flush=True)

# ====================
# 7. SUBMISSION
# ====================
print("\n" + "=" * 80, flush=True)
print("SUBMISSION", flush=True)
print("=" * 80, flush=True)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

submission_path = base_path / 'submissions/submission_v68_two_stage.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_binary.sum()}", flush=True)

# Save artifacts
artifacts = {
    'oof_agn_prob': oof_agn_prob,
    'test_agn_prob': test_agn_prob,
    'oof_maxvar_tde': oof_maxvar_tde,
    'test_maxvar_tde': test_maxvar_tde,
    'best_result': best_result,
    'test_final_prob': test_final_prob
}

with open(base_path / 'data/processed/v68_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# Comparison
print("\n" + "=" * 80, flush=True)
print("COMPARISON", flush=True)
print("=" * 80, flush=True)

print(f"""
   Model                    OOF F1   LB F1
   -----                    ------   -----
   v34a (baseline)          0.6667   0.6907
   v65 (+ all MaxVar)       0.6780   0.6344 (overfit)
   v68 (two-stage)          {best_result['f1']:.4f}   ???

   v68 uses AGN filter to reduce noise before TDE classification.
   If OOF improvement holds on LB, we should see better generalization.
""", flush=True)

print("=" * 80, flush=True)
print("v68 Complete", flush=True)
print("=" * 80, flush=True)
