"""
MALLORN v30: Advanced Physics Features

Building on v21 (OOF F1=0.6708, LB=0.6649) by adding:
1. Multi-epoch blackbody cooling curves (7 temperature measurements)
2. Late-time color evolution (100-200 days post-peak)
3. SED fitting quality metrics
4. Cross-band temporal asymmetry comparisons

Hypothesis: TDE-specific physics signatures (slow cooling, persistent blue
colors, thermal SED, achromatic evolution) will improve discrimination.
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
print("MALLORN v30: Advanced Physics Features", flush=True)
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

y = train_meta['target'].values

# ====================
# 2. EXTRACT ADVANCED PHYSICS FEATURES
# ====================
print("\n2. Extracting advanced physics features...", flush=True)

from features.advanced_physics import extract_advanced_physics_features

print("   Training set...", flush=True)
train_adv = extract_advanced_physics_features(train_lc, train_ids)
print(f"   Extracted {len(train_adv.columns)-1} features for {len(train_adv)} objects", flush=True)

print("   Test set...", flush=True)
test_adv = extract_advanced_physics_features(test_lc, test_ids)
print(f"   Extracted {len(test_adv.columns)-1} features for {len(test_adv)} objects", flush=True)

# ====================
# 3. LOAD v21 BASE FEATURES
# ====================
print("\n3. Loading v21 base features...", flush=True)

# Load v21 features (147 features from features_v4_cache.pkl)
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']

print(f"   v21 base features: {len(train_base.columns)-1}", flush=True)
print(f"   Advanced physics features: {len(train_adv.columns)-1}", flush=True)

# ====================
# 4. COMBINE FEATURES
# ====================
print("\n4. Combining feature sets...", flush=True)

# Merge on object_id
train_combined = train_base.merge(train_adv, on='object_id', how='left')
test_combined = test_base.merge(test_adv, on='object_id', how='left')

# Remove object_id column
X_train = train_combined.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values

feature_names = [c for c in train_combined.columns if c != 'object_id']

print(f"   Total features: {len(feature_names)}", flush=True)
print(f"   Training shape: {X_train.shape}", flush=True)
print(f"   Test shape: {X_test.shape}", flush=True)

# ====================
# 5. TRAIN XGBoost (v21 parameters)
# ====================
print("\n5. Training XGBoost with 5-fold CV...", flush=True)

# Use v21's proven hyperparameters
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
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),  # ~20
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

    # Train
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # Predictions
    oof_preds[val_idx] = model.predict(dval)
    test_preds[:, fold-1] = model.predict(dtest)

    # Feature importance
    importance = model.get_score(importance_type='gain')
    for feat, gain in importance.items():
        if feat in feature_names:
            idx = feature_names.index(feat)
            feature_importance[idx] += gain

    # Fold F1
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

# Find optimal threshold on full OOF
best_f1 = 0
best_thresh = 0.5
for t in np.linspace(0.05, 0.5, 100):
    preds_binary = (oof_preds > t).astype(int)
    f1 = f1_score(y, preds_binary)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}", flush=True)

# Confusion matrix
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
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(30).to_string(index=False), flush=True)

# Highlight new advanced physics features
adv_features_list = [c for c in feature_names if any(x in c for x in [
    'temp_epoch', 'cooling_rate', 'sed_quality', 'late_', 'asymmetry', 'peak_lag'
])]
adv_importance = importance_df[importance_df['feature'].isin(adv_features_list)]
if len(adv_importance) > 0:
    print(f"\n   {len(adv_importance)} advanced physics features in model", flush=True)
    print(f"   Top advanced feature: {adv_importance.iloc[0]['feature']} (rank {importance_df.index.get_loc(adv_importance.index[0])+1})", flush=True)
    total_adv_importance = adv_importance['importance'].sum()
    total_importance = importance_df['importance'].sum()
    print(f"   Advanced features account for {100*total_adv_importance/total_importance:.1f}% of model importance", flush=True)

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

submission_path = base_path / 'submissions/submission_v30_advanced_physics.csv'
submission.to_csv(submission_path, index=False)

print(f"   Submission saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_final.sum()} / {len(test_final)}", flush=True)

# Save OOF predictions and model artifacts
print("\n8. Saving model artifacts...", flush=True)

artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'feature_importance': importance_df,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'xgb_params': xgb_params,
    'feature_names': feature_names,
    'train_combined': train_combined,
    'test_combined': test_combined
}

artifacts_path = base_path / 'data/processed/v30_artifacts.pkl'
with open(artifacts_path, 'wb') as f:
    pickle.dump(artifacts, f)

print(f"   Artifacts saved: {artifacts_path.name}", flush=True)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v30 Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"Baseline v21: OOF F1 = 0.6708", flush=True)
print(f"Change: {(best_f1 - 0.6708)*100:+.2f}% ({best_f1 - 0.6708:+.4f})", flush=True)
print("=" * 80, flush=True)
