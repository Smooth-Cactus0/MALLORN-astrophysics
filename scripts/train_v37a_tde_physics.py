"""
MALLORN v37a: XGBoost + TDE Physics Model (Technique #9)

REPLACING Bazin (SN-optimized) with custom TDE physics model.

TDE hybrid model:
- Sigmoid rise (circularization timescale)
- Exponential decay (early-time)
- Power law decay with α parameter (late-time t^(-5/3) fallback)

f(t) = A * [1/(1+exp(-(t-t0)/τ_rise))] * [exp(-(t-t0)/τ_fall)] * [(1+(t-t0)/τ_fall)^(-α)] + B

Physics basis:
- Ṁ(t) ∝ t^(-5/3) fallback rate (Guillochon & Ramirez-Ruiz 2013)
- Super-Eddington accretion (Lodato & Rossi 2011)
- α ~ 1.67 for TDEs (general relativity prediction)

Features added:
- 6 bands × 9 parameters = 54 features
- 7 cross-band consistency features (incl. alpha deviation from theory)
- Total: 61 new features

Hypothesis: TDE-specific physics should outperform SN-optimized Bazin
v34a (Bazin): OOF F1=0.6667, LB F1=0.6907
Target: LB F1 > 0.70 (+1-4% gain)
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
print("MALLORN v37a: TDE Physics Model (Technique #9/9)", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD v21 FEATURES
# ====================
print("\n1. Loading v21 baseline features...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_lc = data['train_lc']
test_lc = data['test_lc']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

# Load v21 feature set
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
tde_cols = [c for c in train_tde.columns if c != 'object_id']

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

print(f"   v21 baseline: {len(train_v21.columns)-1} features", flush=True)

# ====================
# 2. EXTRACT TDE PHYSICS FEATURES
# ====================
print("\n2. Extracting TDE physics model features...", flush=True)

from features.tde_physics_model import extract_tde_features

print("   Training set...", flush=True)
train_tde = extract_tde_features(train_lc, train_ids, model_type='hybrid')
print(f"   Extracted {len(train_tde.columns)-1} TDE features for {len(train_tde)} objects", flush=True)

print("   Test set...", flush=True)
test_tde = extract_tde_features(test_lc, test_ids, model_type='hybrid')
print(f"   Extracted {len(test_tde.columns)-1} TDE features for {len(test_tde)} objects", flush=True)

# Check fit success rate
r_success = train_tde['r_tde_A'].notna().sum()
print(f"   Fit success rate (r-band): {100*r_success/len(train_tde):.1f}%", flush=True)

# Check mean alpha (should be ~1.67 for TDEs)
alpha_col = 'r_tde_alpha'
if alpha_col in train_tde.columns:
    alpha_values = train_tde[alpha_col].dropna()
    if len(alpha_values) > 0:
        print(f"   Mean alpha (r-band): {alpha_values.mean():.2f} (theory: 1.67 for TDEs)", flush=True)

# ====================
# 3. COMBINE FEATURES
# ====================
print("\n3. Combining v21 + TDE features...", flush=True)

train_combined = train_v21.merge(train_tde, on='object_id', how='left')
test_combined = test_v21.merge(test_tde, on='object_id', how='left')

X_train = train_combined.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values
feature_names = [c for c in train_combined.columns if c != 'object_id']

print(f"   Total features: {len(feature_names)}", flush=True)
print(f"   ({len(train_v21.columns)-1} v21 + {len(train_tde.columns)-1} TDE)", flush=True)
print(f"   Training shape: {X_train.shape}", flush=True)

# ====================
# 4. TRAIN XGBOOST (v21 parameters)
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

    importance = model.get_score(importance_type='gain')
    for feat, gain in importance.items():
        if feat in feature_names:
            idx = feature_names.index(feat)
            feature_importance[idx] += gain

    best_f1 = 0
    best_thresh = 0.5
    for t in np.linspace(0.05, 0.5, 50):
        preds_binary = (oof_preds[val_idx] > t).astype(int)
        f1 = f1_score(y_val, preds_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"      Best threshold: {best_thresh:.3f}, F1: {best_f1:.4f}", flush=True)

print("\n" + "=" * 80, flush=True)
print("CROSS-VALIDATION RESULTS", flush=True)
print("=" * 80, flush=True)

best_f1 = 0
best_thresh = 0.5
for t in np.linspace(0.05, 0.5, 100):
    preds_binary = (oof_preds > t).astype(int)
    f1 = f1_score(y, preds_binary)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}", flush=True)

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
tn = np.sum((final_preds == 0) & (y == 0))

print(f"   Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}", flush=True)
print(f"   Recall: {tp/(tp+fn):.4f}", flush=True)

# ====================
# 5. FEATURE IMPORTANCE
# ====================
print("\n5. Top 30 Features by Importance:", flush=True)

feature_importance = feature_importance / n_folds
importance_df_result = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df_result.head(30).to_string(index=False), flush=True)

# Highlight TDE features
tde_cols = [c for c in feature_names if 'tde' in c]
tde_importance = importance_df_result[importance_df_result['feature'].isin(tde_cols)]
if len(tde_importance) > 0:
    print(f"\n   {len(tde_importance)} TDE features in model", flush=True)
    print(f"   Top TDE feature: {tde_importance.iloc[0]['feature']} (rank {list(importance_df_result.index).index(tde_importance.index[0])+1})", flush=True)
    total_tde_importance = tde_importance['importance'].sum()
    total_importance = importance_df_result['importance'].sum()
    print(f"   TDE features account for {100*total_tde_importance/total_importance:.1f}% of model importance", flush=True)

    print("\n   Top 10 TDE features:")
    for idx, row in tde_importance.head(10).iterrows():
        rank = list(importance_df_result.index).index(idx) + 1
        print(f"      {rank:3d}. {row['feature']:35s} {row['importance']:8.1f}", flush=True)

    # Highlight alpha features specifically (key physics parameter)
    alpha_features = tde_importance[tde_importance['feature'].str.contains('alpha')]
    if len(alpha_features) > 0:
        print(f"\n   Alpha (power law) features:")
        for idx, row in alpha_features.head(5).iterrows():
            rank = list(importance_df_result.index).index(idx) + 1
            print(f"      {rank:3d}. {row['feature']:35s} {row['importance']:8.1f}", flush=True)

# ====================
# 6. CREATE SUBMISSION
# ====================
print("\n6. Creating submission...", flush=True)

test_avg = test_preds.mean(axis=1)
test_final = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v37a_tde_physics.csv'
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
    'feature_names': feature_names
}

with open(base_path / 'data/processed/v37a_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v37a (TDE Physics Model) Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"v34a (Bazin): OOF F1 = 0.6667, LB F1 = 0.6907", flush=True)
print(f"v21 (baseline): OOF F1 = 0.6708, LB F1 = 0.6649", flush=True)

change_vs_v34a_oof = (best_f1 - 0.6667) * 100 / 0.6667
change_vs_v21_oof = (best_f1 - 0.6708) * 100 / 0.6708

print(f"Change vs v34a (OOF): {change_vs_v34a_oof:+.2f}% ({best_f1 - 0.6667:+.4f})", flush=True)
print(f"Change vs v21 (OOF): {change_vs_v21_oof:+.2f}% ({best_f1 - 0.6708:+.4f})", flush=True)

if best_f1 > 0.6667:
    print("SUCCESS: TDE physics model improved over Bazin!", flush=True)
elif best_f1 > 0.6708:
    print("IMPROVEMENT: Beat v21 baseline!", flush=True)
else:
    print("TDE model did not improve - analyze alpha distribution and fit quality", flush=True)

print("\nNEXT STEP: Test on leaderboard to see if TDE physics generalizes better", flush=True)
print("=" * 80, flush=True)
