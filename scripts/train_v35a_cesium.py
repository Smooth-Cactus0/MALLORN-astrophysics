"""
MALLORN v35a: Bazin + Cesium Astronomy Features (Techniques #1 + #4)

Building on v34a (LB F1=0.6907) with Cesium domain-specific features.

Features added to v34a:
- v34a: 224 features (v21 baseline + Bazin parametric)
- Cesium: ~80 new features (astronomy-specific variability metrics)
- Total: ~300 features

Cesium Features:
- Stetson J, K: Correlated variability indices
- Beyond N-std: Outlier detection (1σ, 2σ)
- Flux percentile ratios: Shape characterization (mid20, mid35, mid50, mid65, mid80)
- Maximum slope: Fastest rise/fall rate
- Linear trend: Overall flux evolution
- Percent amplitude: Peak-to-median ratio
- Anderson-Darling: Normality test

Expected gain: +1-3% (PLAsTiCC all top 3 used Cesium)
Target: LB F1 > 0.71 (from v34a 0.6907)
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
print("MALLORN v35a: Bazin + Cesium Features (Techniques #1+4 / 9)", flush=True)
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
# 2. EXTRACT BAZIN FEATURES
# ====================
print("\n2. Extracting Bazin parametric features...", flush=True)

from features.bazin_fitting import extract_bazin_features

print("   Training set...", flush=True)
train_bazin = extract_bazin_features(train_lc, train_ids)
print(f"   Extracted {len(train_bazin.columns)-1} Bazin features for {len(train_bazin)} objects", flush=True)

print("   Test set...", flush=True)
test_bazin = extract_bazin_features(test_lc, test_ids)
print(f"   Extracted {len(test_bazin.columns)-1} Bazin features for {len(test_bazin)} objects", flush=True)

# Check fit success rate
r_success = train_bazin['r_bazin_A'].notna().sum()
print(f"   Fit success rate (r-band): {100*r_success/len(train_bazin):.1f}%", flush=True)

# ====================
# 3. EXTRACT CESIUM FEATURES
# ====================
print("\n3. Extracting Cesium astronomy features...", flush=True)

from features.cesium_features import extract_cesium_features

print("   Training set...", flush=True)
train_cesium = extract_cesium_features(train_lc, train_ids)
print(f"   Extracted {len(train_cesium.columns)-1} Cesium features for {len(train_cesium)} objects", flush=True)

print("   Test set...", flush=True)
test_cesium = extract_cesium_features(test_lc, test_ids)
print(f"   Extracted {len(test_cesium.columns)-1} Cesium features for {len(test_cesium)} objects", flush=True)

# Check feature coverage
stetson_j_coverage = train_cesium['r_cesium_stetson_j'].notna().sum() / len(train_cesium)
print(f"   Feature coverage (r-band Stetson J): {100*stetson_j_coverage:.1f}%", flush=True)

# ====================
# 4. COMBINE ALL FEATURES
# ====================
print("\n4. Combining v21 + Bazin + Cesium features...", flush=True)

train_combined = train_v21.merge(train_bazin, on='object_id', how='left')
train_combined = train_combined.merge(train_cesium, on='object_id', how='left')

test_combined = test_v21.merge(test_bazin, on='object_id', how='left')
test_combined = test_combined.merge(test_cesium, on='object_id', how='left')

X_train = train_combined.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values
feature_names = [c for c in train_combined.columns if c != 'object_id']

print(f"   Total features: {len(feature_names)}", flush=True)
print(f"   ({len(train_v21.columns)-1} v21 + {len(train_bazin.columns)-1} Bazin + {len(train_cesium.columns)-1} Cesium)", flush=True)
print(f"   Training shape: {X_train.shape}", flush=True)

# ====================
# 5. TRAIN XGBOOST
# ====================
print("\n5. Training XGBoost with 5-fold CV...", flush=True)

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
# 6. FEATURE IMPORTANCE
# ====================
print("\n6. Top 30 Features by Importance:", flush=True)

feature_importance = feature_importance / n_folds
importance_df_result = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df_result.head(30).to_string(index=False), flush=True)

# Highlight Bazin features
bazin_cols = [c for c in feature_names if 'bazin' in c]
bazin_importance = importance_df_result[importance_df_result['feature'].isin(bazin_cols)]
if len(bazin_importance) > 0:
    total_bazin_importance = bazin_importance['importance'].sum()
    total_importance = importance_df_result['importance'].sum()
    print(f"\n   Bazin: {len(bazin_importance)} features, {100*total_bazin_importance/total_importance:.1f}% importance", flush=True)
    print(f"   Top Bazin: {bazin_importance.iloc[0]['feature']} (rank {list(importance_df_result.index).index(bazin_importance.index[0])+1})", flush=True)

# Highlight Cesium features
cesium_cols = [c for c in feature_names if 'cesium' in c]
cesium_importance = importance_df_result[importance_df_result['feature'].isin(cesium_cols)]
if len(cesium_importance) > 0:
    total_cesium_importance = cesium_importance['importance'].sum()
    print(f"\n   Cesium: {len(cesium_importance)} features, {100*total_cesium_importance/total_importance:.1f}% importance", flush=True)
    print(f"   Top Cesium: {cesium_importance.iloc[0]['feature']} (rank {list(importance_df_result.index).index(cesium_importance.index[0])+1})", flush=True)

    print("\n   Top 10 Cesium features:")
    for idx, row in cesium_importance.head(10).iterrows():
        rank = list(importance_df_result.index).index(idx) + 1
        print(f"      {rank:3d}. {row['feature']:35s} {row['importance']:8.1f}", flush=True)

# ====================
# 7. CREATE SUBMISSION
# ====================
print("\n7. Creating submission...", flush=True)

test_avg = test_preds.mean(axis=1)
test_final = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v35a_cesium.csv'
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

with open(base_path / 'data/processed/v35a_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v35a (Bazin + Cesium) Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"v34a (Bazin): OOF F1 = 0.6667, LB F1 = 0.6907", flush=True)
print(f"v21 (baseline): OOF F1 = 0.6708, LB F1 = 0.6649", flush=True)
change_vs_v34a_oof = (best_f1 - 0.6667) * 100 / 0.6667
change_vs_v21_oof = (best_f1 - 0.6708) * 100 / 0.6708
print(f"Change vs v34a (OOF): {change_vs_v34a_oof:+.2f}% ({best_f1 - 0.6667:+.4f})", flush=True)
print(f"Change vs v21 (OOF): {change_vs_v21_oof:+.2f}% ({best_f1 - 0.6708:+.4f})", flush=True)

if best_f1 > 0.70:
    print("SUCCESS: Achieved Week 1 target (OOF > 0.70)!", flush=True)
elif best_f1 > 0.6667:
    print("IMPROVEMENT: Beat v34a OOF!", flush=True)
else:
    print("Cesium features did not improve - analyze and adjust", flush=True)

print("=" * 80, flush=True)
