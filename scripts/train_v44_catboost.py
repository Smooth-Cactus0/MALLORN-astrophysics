"""
MALLORN v44: CatBoost on v34a (Bazin) Features - Model Diversity Strategy

Testing if CatBoost can capture different patterns than XGBoost on the same proven features.

Key differences from XGBoost:
- Symmetric trees (more balanced, less prone to overfitting)
- Ordered boosting (reduces target leakage during training)
- Different splitting criteria (may find different decision boundaries)

Same features as v34a (224):
- 120 selected v4 baseline features
- 52 Bazin parametric features
- TDE physics features
- Multi-band GP features

Strategy: Model diversity for ensembling
Expected: Different errors than XGBoost → ensemble potential
Target: LB F1 > 0.69 (standalone) or good ensemble candidate
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier, Pool
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v44: CatBoost on v34a (Bazin) Features", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. RECONSTRUCT v34a FEATURES
# ====================
print("\n1. Loading v34a (Bazin) feature set...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_lc = data['train_lc']
test_lc = data['test_lc']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

# Load v34a artifacts for reference
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

print(f"   v34a baseline: OOF F1={v34a['oof_f1']:.4f}, LB F1=0.6907 (XGBoost)", flush=True)
print(f"   Features: {len(v34a['feature_names'])}", flush=True)

# Reconstruct v34a features
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

# Load TDE and GP features
tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']

with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

# Build v21 baseline
train_v21 = train_base[['object_id'] + selected_120].copy()
train_v21 = train_v21.merge(train_tde, on='object_id', how='left')
train_v21 = train_v21.merge(train_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

test_v21 = test_base[['object_id'] + selected_120].copy()
test_v21 = test_v21.merge(test_tde, on='object_id', how='left')
test_v21 = test_v21.merge(test_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

print(f"   v21 baseline: {len(train_v21.columns)-1} features", flush=True)

# Extract Bazin features
print("   Extracting Bazin features...", flush=True)
from features.bazin_fitting import extract_bazin_features

print("      Training set...", flush=True)
train_bazin = extract_bazin_features(train_lc, train_ids)
print(f"      Extracted {len(train_bazin.columns)-1} Bazin features", flush=True)

print("      Test set...", flush=True)
test_bazin = extract_bazin_features(test_lc, test_ids)
print(f"      Extracted {len(test_bazin.columns)-1} Bazin features", flush=True)

# Combine v21 + Bazin (= v34a features)
train_combined = train_v21.merge(train_bazin, on='object_id', how='left')
test_combined = test_v21.merge(test_bazin, on='object_id', how='left')

X_train = train_combined.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values
feature_names = [c for c in train_combined.columns if c != 'object_id']

print(f"   v34a features reconstructed: {len(feature_names)}", flush=True)
print(f"   Training shape: {X_train.shape}", flush=True)

# ====================
# 2. TRAIN CATBOOST
# ====================
print("\n2. Training CatBoost with 5-fold CV...", flush=True)

# CatBoost parameters (tuned for similar complexity to XGBoost v34a)
catboost_params = {
    'iterations': 500,
    'learning_rate': 0.025,
    'depth': 5,
    'l2_leaf_reg': 1.5,  # Similar to reg_lambda
    'subsample': 0.8,
    'colsample_bylevel': 0.8,  # Similar to colsample_bytree
    'min_data_in_leaf': 3,  # Similar to min_child_weight
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 50,
    'task_type': 'CPU',
    'thread_count': -1
}

print(f"   scale_pos_weight: {catboost_params['scale_pos_weight']:.2f}", flush=True)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros((len(X_test), n_folds))
feature_importance = np.zeros(len(feature_names))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Create CatBoost pools
    train_pool = Pool(X_tr, y_tr, feature_names=feature_names)
    val_pool = Pool(X_val, y_val, feature_names=feature_names)
    test_pool = Pool(X_test, feature_names=feature_names)

    # Train CatBoost
    model = CatBoostClassifier(**catboost_params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        verbose=False
    )

    # Predictions (probabilities)
    oof_preds[val_idx] = model.predict_proba(val_pool)[:, 1]
    test_preds[:, fold-1] = model.predict_proba(test_pool)[:, 1]

    # Feature importance
    importance = model.get_feature_importance()
    feature_importance += importance

    # Find best threshold
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

# Find global best threshold
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
# 3. FEATURE IMPORTANCE
# ====================
print("\n3. Top 30 Features by Importance:", flush=True)

feature_importance = feature_importance / n_folds
importance_df_result = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df_result.head(30).to_string(index=False), flush=True)

# ====================
# 4. COMPARE TO XGBoost v34a
# ====================
print("\n4. Comparison to XGBoost v34a:", flush=True)

from scipy.stats import spearmanr

# Compare predictions
correlation = spearmanr(oof_preds, v34a['oof_preds'])[0]
print(f"   Prediction correlation (Spearman): {correlation:.4f}", flush=True)

if correlation < 0.90:
    print("   LOW correlation = different patterns captured!", flush=True)
    print("   Good ensemble candidate", flush=True)
elif correlation < 0.95:
    print("   Moderate correlation = some different patterns", flush=True)
    print("   May benefit from ensemble", flush=True)
else:
    print("   HIGH correlation = very similar predictions", flush=True)
    print("   Limited ensemble benefit", flush=True)

# Compare errors
xgb_errors = (v34a['oof_preds'] > v34a['best_threshold']).astype(int) != y
catboost_errors = (oof_preds > best_thresh).astype(int) != y

xgb_only_errors = xgb_errors & ~catboost_errors
catboost_only_errors = catboost_errors & ~xgb_errors
both_errors = xgb_errors & catboost_errors

print(f"   XGBoost-only errors: {xgb_only_errors.sum()}", flush=True)
print(f"   CatBoost-only errors: {catboost_only_errors.sum()}", flush=True)
print(f"   Both wrong: {both_errors.sum()}", flush=True)
print(f"   Complementary errors: {xgb_only_errors.sum() + catboost_only_errors.sum()}", flush=True)

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

submission_path = base_path / 'submissions/submission_v44_catboost.csv'
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
    'model_type': 'CatBoost'
}

with open(base_path / 'data/processed/v44_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v44 (CatBoost on Bazin) Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"v34a (XGBoost on Bazin): OOF F1 = {v34a['oof_f1']:.4f}, LB F1 = 0.6907", flush=True)

change_vs_v34a_oof = (best_f1 - v34a['oof_f1']) * 100 / v34a['oof_f1']
print(f"Change vs v34a (OOF): {change_vs_v34a_oof:+.2f}% ({best_f1 - v34a['oof_f1']:+.4f})", flush=True)

if best_f1 > v34a['oof_f1']:
    print("SUCCESS: CatBoost improved over XGBoost!", flush=True)
    print(f"Expected LB: 0.69-0.71", flush=True)
elif abs(best_f1 - v34a['oof_f1']) < 0.005:
    print("NEUTRAL: Similar OOF to XGBoost", flush=True)
    print("Check prediction correlation for ensemble potential", flush=True)
else:
    print("CatBoost did not improve on OOF", flush=True)
    print("Test on LB to check if architecture helps", flush=True)

print("\nKey Insight: CatBoost uses symmetric trees and ordered boosting", flush=True)
print("  - More regularized than XGBoost depth-wise trees", flush=True)
print("  - May capture different decision boundaries", flush=True)
print("  - Low correlation with XGBoost = good ensemble candidate", flush=True)

print("\nNEXT STEP: Test v44 on leaderboard", flush=True)
print("If competitive, ensemble XGBoost (v34a) + CatBoost (v44)", flush=True)
print("=" * 80, flush=True)
