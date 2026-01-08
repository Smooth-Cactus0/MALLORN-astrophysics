"""
MALLORN v32: XGBoost with Feature Interactions (Experiment 2)

Physics-motivated feature interactions capture non-linear relationships:
- Color × Redshift (rest-frame evolution)
- Temperature × Time (cooling dynamics)
- Amplitude × Duration (luminosity-timescale relation)
- GP scales × Amplitude
- Asymmetry × Color

Expected improvement: +2-4% over v21
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
print("MALLORN v32: XGBoost with Feature Interactions", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD v21 FEATURES
# ====================
print("\n1. Loading v21 feature set...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

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

train_combined = train_base[['object_id'] + selected_120].copy()
train_combined = train_combined.merge(train_tde, on='object_id', how='left')
train_combined = train_combined.merge(train_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

test_combined = test_base[['object_id'] + selected_120].copy()
test_combined = test_combined.merge(test_tde, on='object_id', how='left')
test_combined = test_combined.merge(test_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

print(f"   v21 features: {len(train_combined.columns)-1}", flush=True)

# ====================
# 2. CREATE INTERACTIONS
# ====================
print("\n2. Creating physics-motivated interactions...", flush=True)

from features.interactions import create_physics_interactions, select_top_interactions

# Get feature names (exclude object_id)
original_features = [c for c in train_combined.columns if c != 'object_id']

# Create interactions
print("   Generating interactions...", flush=True)
train_with_int = create_physics_interactions(
    train_combined.drop(columns=['object_id']),
    original_features
)
test_with_int = create_physics_interactions(
    test_combined.drop(columns=['object_id']),
    original_features
)

print(f"   Total features: {len(train_with_int.columns)} (was {len(original_features)})", flush=True)

# Select top interactions
print("   Selecting top 30 interactions by correlation...", flush=True)
top_interactions = select_top_interactions(
    train_with_int,
    y,
    original_features,
    top_k=30
)

# Keep original + top interactions
selected_features = original_features + top_interactions
X_train = train_with_int[selected_features].values
X_test = test_with_int[selected_features].values
feature_names = selected_features

print(f"   Final feature count: {len(feature_names)}", flush=True)
print(f"   ({len(original_features)} original + {len(top_interactions)} interactions)", flush=True)

# ====================
# 3. TRAIN XGBOOST
# ====================
print("\n3. Training XGBoost with 5-fold CV...", flush=True)

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
# 4. FEATURE IMPORTANCE
# ====================
print("\n4. Top 30 Features by Importance:", flush=True)

feature_importance = feature_importance / n_folds
importance_df_result = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df_result.head(30).to_string(index=False), flush=True)

# Highlight interaction features
interaction_importance = importance_df_result[importance_df_result['feature'].isin(top_interactions)]
if len(interaction_importance) > 0:
    print(f"\n   {len(interaction_importance)} interaction features used in model", flush=True)
    top_int = interaction_importance.head(10)
    print("\n   Top 10 interaction features:")
    for idx, row in top_int.iterrows():
        rank = list(importance_df_result.index).index(idx) + 1
        print(f"      {rank:3d}. {row['feature']:35s} {row['importance']:8.1f}", flush=True)

# ====================
# 5. CREATE SUBMISSION
# ====================
print("\n5. Creating submission...", flush=True)

test_avg = test_preds.mean(axis=1)
test_final = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v32_interactions.csv'
submission.to_csv(submission_path, index=False)

print(f"   Submission saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_final.sum()} / {len(test_final)}", flush=True)

artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'feature_importance': importance_df_result,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'feature_names': feature_names,
    'interaction_features': top_interactions
}

with open(base_path / 'data/processed/v32_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v32 (Interactions) Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"Baseline v21 (XGBoost): OOF F1 = 0.6708", flush=True)
change_pct = (best_f1 - 0.6708) * 100
change_abs = best_f1 - 0.6708
print(f"Change: {change_pct:+.2f}% ({change_abs:+.4f})", flush=True)

if best_f1 > 0.6708:
    print("SUCCESS: Feature interactions improved performance!", flush=True)
else:
    print("Interactions did not improve beyond baseline", flush=True)
print("=" * 80, flush=True)
