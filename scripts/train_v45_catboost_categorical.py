"""
MALLORN v45: CatBoost with Categorical Features

Leveraging CatBoost's native categorical feature handling.

Key improvements over v44:
1. Added ~50-80 categorical features (binned continuous features)
2. Physics-meaningful bins: redshift, colors, timescales, brightness, asymmetry
3. CatBoost's ordered target encoding handles these optimally

Strategy: Get features right FIRST, then tune hyperparameters
v44 result: OOF 0.6015, LB 0.5760 (failed)
v45 goal: Improve by giving CatBoost the right feature types

Expected: +5-10% over v44 from categorical features alone
Target: OOF > 0.65, LB > 0.65
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
print("MALLORN v45: CatBoost with Categorical Features", flush=True)
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

# Load v34a and v44 for comparison
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

with open(base_path / 'data/processed/v44_artifacts.pkl', 'rb') as f:
    v44 = pickle.load(f)

print(f"   v34a (XGBoost): OOF F1=0.6667, LB F1=0.6907", flush=True)
print(f"   v44 (CatBoost): OOF F1={v44['oof_f1']:.4f}, LB F1=0.5760", flush=True)
print(f"   v34a features: {len(v34a['feature_names'])}", flush=True)

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
train_v34a = train_v21.merge(train_bazin, on='object_id', how='left')
test_v34a = test_v21.merge(test_bazin, on='object_id', how='left')

print(f"   v34a features reconstructed: {len(train_v34a.columns)-1}", flush=True)

# ====================
# 2. ADD CATEGORICAL FEATURES
# ====================
print("\n2. Adding categorical features for CatBoost...", flush=True)

from features.catboost_categorical import add_categorical_features

train_enhanced, train_cat_indices = add_categorical_features(train_v34a)
test_enhanced, test_cat_indices = add_categorical_features(test_v34a)

print(f"   Total features: {len(train_enhanced.columns)-1}", flush=True)
print(f"   Categorical features: {len(train_cat_indices)}", flush=True)
print(f"   Continuous features: {len(train_enhanced.columns)-1-len(train_cat_indices)}", flush=True)

# Keep as DataFrames for CatBoost (it handles categorical features better this way)
X_train_df = train_enhanced.drop(columns=['object_id'])
X_test_df = test_enhanced.drop(columns=['object_id'])
feature_names = list(X_train_df.columns)

# Get categorical feature names (not indices, since we're using DataFrames)
cat_feature_names = [feature_names[i] for i in train_cat_indices]

print(f"   Training shape: {X_train_df.shape}", flush=True)
print(f"   Categorical feature names: {len(cat_feature_names)}", flush=True)

# ====================
# 3. TRAIN CATBOOST WITH CATEGORICAL FEATURES
# ====================
print("\n3. Training CatBoost with categorical features...", flush=True)

# Same hyperparameters as v44 to isolate categorical feature impact
catboost_params = {
    'iterations': 500,
    'learning_rate': 0.025,
    'depth': 5,
    'l2_leaf_reg': 1.5,
    'subsample': 0.8,
    'colsample_bylevel': 0.8,
    'min_data_in_leaf': 3,
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 50,
    'task_type': 'CPU',
    'thread_count': -1
}

print(f"   scale_pos_weight: {catboost_params['scale_pos_weight']:.2f}", flush=True)
print(f"   Using {len(cat_feature_names)} categorical features", flush=True)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train_df))
test_preds = np.zeros((len(X_test_df), n_folds))
feature_importance = np.zeros(len(feature_names))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_df, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr = X_train_df.iloc[train_idx]
    X_val = X_train_df.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Create CatBoost pools with categorical features specified by name
    train_pool = Pool(X_tr, y_tr, cat_features=cat_feature_names)
    val_pool = Pool(X_val, y_val, cat_features=cat_feature_names)
    test_pool = Pool(X_test_df, cat_features=cat_feature_names)

    # Train CatBoost
    model = CatBoostClassifier(**catboost_params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        verbose=False
    )

    # Predictions
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
# 4. FEATURE IMPORTANCE
# ====================
print("\n4. Top 30 Features by Importance:", flush=True)

feature_importance = feature_importance / n_folds
importance_df_result = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df_result.head(30).to_string(index=False), flush=True)

# Analyze categorical feature importance
cat_feature_names = [feature_names[i] for i in train_cat_indices]
cat_importance = importance_df_result[importance_df_result['feature'].isin(cat_feature_names)]

if len(cat_importance) > 0:
    total_cat_importance = cat_importance['importance'].sum()
    total_importance = importance_df_result['importance'].sum()
    print(f"\n   Categorical features: {100*total_cat_importance/total_importance:.1f}% of model importance", flush=True)

    print("\n   Top 10 Categorical features:")
    for idx, row in cat_importance.head(10).iterrows():
        rank = list(importance_df_result.index).index(idx) + 1
        print(f"      {rank:3d}. {row['feature']:40s} {row['importance']:8.2f}", flush=True)

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

submission_path = base_path / 'submissions/submission_v45_catboost_categorical.csv'
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
    'categorical_indices': train_cat_indices,
    'model_type': 'CatBoost_Categorical'
}

with open(base_path / 'data/processed/v45_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v45 (CatBoost + Categorical) Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"v44 (CatBoost baseline): OOF F1 = {v44['oof_f1']:.4f}, LB F1 = 0.5760", flush=True)
print(f"v34a (XGBoost): OOF F1 = {v34a['oof_f1']:.4f}, LB F1 = 0.6907 (best)", flush=True)

change_vs_v44 = (best_f1 - v44['oof_f1']) * 100 / v44['oof_f1']
change_vs_v34a = (best_f1 - v34a['oof_f1']) * 100 / v34a['oof_f1']

print(f"\nChange vs v44 (OOF): {change_vs_v44:+.2f}% ({best_f1 - v44['oof_f1']:+.4f})", flush=True)
print(f"Change vs v34a (OOF): {change_vs_v34a:+.2f}% ({best_f1 - v34a['oof_f1']:+.4f})", flush=True)

if best_f1 > v44['oof_f1']:
    print("\nSUCCESS: Categorical features improved CatBoost!", flush=True)
    if best_f1 > v34a['oof_f1']:
        print("BREAKTHROUGH: Beat XGBoost v34a!", flush=True)
        print(f"Expected LB: 0.69-0.72", flush=True)
    else:
        print(f"Still below XGBoost, but categorical features helped", flush=True)
        print("Next step: Hyperparameter tuning", flush=True)
else:
    print("\nCategorical features did not help on OOF", flush=True)
    print("May still help generalization - test on LB", flush=True)

print("\nKey Insight: CatBoost categorical features", flush=True)
print("  - Physics-meaningful bins: redshift, colors, timescales", flush=True)
print("  - Ordered target encoding captures complex interactions", flush=True)
print("  - Better than one-hot encoding for high-cardinality features", flush=True)

print("\nNEXT STEP: Test v45 on leaderboard", flush=True)
print("If improved over v44, proceed to hyperparameter tuning", flush=True)
print("=" * 80, flush=True)
