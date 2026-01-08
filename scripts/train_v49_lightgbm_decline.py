"""
MALLORN v49: LightGBM + Time-to-Decline Features

Same features as v48, but with LightGBM (leaf-wise growth).

v46 (LightGBM baseline): OOF 0.6120, LB 0.60
v48 (XGBoost + decline): OOF pending

Features:
- v34a baseline (224 Bazin features)
- Time-to-decline (36 features)
- Total: 260 features

LightGBM advantages:
- Leaf-wise tree growth (more aggressive)
- Different splitting criteria than XGBoost
- Good ensemble diversity (correlation 0.68 with XGBoost)

Expected: Improved over v46, potential ensemble with v48
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v49: LightGBM + Time-to-Decline", flush=True)
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

# Load references
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

with open(base_path / 'data/processed/v46_artifacts.pkl', 'rb') as f:
    v46 = pickle.load(f)

print(f"   v34a (XGBoost): OOF F1=0.6667, LB F1=0.6907", flush=True)
print(f"   v46 (LightGBM baseline): OOF F1={v46['oof_f1']:.4f}, LB F1=0.60", flush=True)

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

# Extract Bazin features
print("   Extracting Bazin features...", flush=True)
from features.bazin_fitting import extract_bazin_features

print("      Training set...", flush=True)
train_bazin = extract_bazin_features(train_lc, train_ids)

print("      Test set...", flush=True)
test_bazin = extract_bazin_features(test_lc, test_ids)

# Combine v21 + Bazin (= v34a features)
train_v34a = train_v21.merge(train_bazin, on='object_id', how='left')
test_v34a = test_v21.merge(test_bazin, on='object_id', how='left')

print(f"   v34a features reconstructed: {len(train_v34a.columns)-1}", flush=True)

# ====================
# 2. EXTRACT TIME-TO-DECLINE FEATURES
# ====================
print("\n2. Extracting time-to-decline features...", flush=True)

from features.time_to_decline import extract_time_to_decline

print("   Training set...", flush=True)
train_decline = extract_time_to_decline(train_lc, train_ids)
print(f"   Extracted {len(train_decline.columns)-1} time-to-decline features", flush=True)

print("   Test set...", flush=True)
test_decline = extract_time_to_decline(test_lc, test_ids)
print(f"   Extracted {len(test_decline.columns)-1} time-to-decline features", flush=True)

# ====================
# 3. COMBINE FEATURES
# ====================
print("\n3. Combining v34a + time-to-decline...", flush=True)

train_combined = train_v34a.merge(train_decline, on='object_id', how='left')
test_combined = test_v34a.merge(test_decline, on='object_id', how='left')

X_train = train_combined.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values
feature_names = [c for c in train_combined.columns if c != 'object_id']

print(f"   Total features: {len(feature_names)}", flush=True)
print(f"   Training shape: {X_train.shape}", flush=True)

# ====================
# 4. TRAIN LIGHTGBM
# ====================
print("\n4. Training LightGBM with 5-fold CV...", flush=True)

# Same hyperparameters as v46
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.025,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_child_samples': 3,
    'lambda_l1': 0.2,
    'lambda_l2': 1.5,
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
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

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds[:, fold-1] = model.predict(X_test, num_iteration=model.best_iteration)

    importance = model.feature_importance(importance_type='gain')
    feature_importance += importance

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

# Analyze time-to-decline importance
decline_cols = [c for c in feature_names if 'decline' in c]
decline_importance = importance_df_result[importance_df_result['feature'].isin(decline_cols)]

if len(decline_importance) > 0:
    total_decline_importance = decline_importance['importance'].sum()
    total_importance = importance_df_result['importance'].sum()
    print(f"\n   Time-to-decline features: {100*total_decline_importance/total_importance:.1f}% of model importance", flush=True)

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

submission_path = base_path / 'submissions/submission_v49_lightgbm_decline.csv'
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
    'model_type': 'LightGBM'
}

with open(base_path / 'data/processed/v49_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v49 (LightGBM + Time-to-Decline) Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"v46 (LightGBM baseline): OOF F1 = {v46['oof_f1']:.4f}, LB F1 = 0.60", flush=True)
print(f"v34a (XGBoost): OOF F1 = {v34a['oof_f1']:.4f}, LB F1 = 0.6907", flush=True)

change_vs_v46 = (best_f1 - v46['oof_f1']) * 100 / v46['oof_f1']
change_vs_v34a = (best_f1 - v34a['oof_f1']) * 100 / v34a['oof_f1']

print(f"\nChange vs v46 (OOF): {change_vs_v46:+.2f}% ({best_f1 - v46['oof_f1']:+.4f})", flush=True)
print(f"Change vs v34a (OOF): {change_vs_v34a:+.2f}% ({best_f1 - v34a['oof_f1']:+.4f})", flush=True)

if best_f1 > v46['oof_f1']:
    print("\nSUCCESS: Time-to-decline improved LightGBM!", flush=True)
    if best_f1 > v34a['oof_f1']:
        print("BREAKTHROUGH: Beat XGBoost v34a!", flush=True)
else:
    print("\nTime-to-decline did not improve LightGBM on OOF", flush=True)
    print("Test on LB to check generalization", flush=True)

print("\nKey Insight: LightGBM + Time-to-Decline", flush=True)
print("  - Leaf-wise growth captures different patterns than XGBoost", flush=True)
print("  - Time-to-decline = Boone's key winning feature", flush=True)
print("  - Good ensemble potential with v48 (XGBoost)", flush=True)

print("\nNEXT STEP: Test v49 on leaderboard", flush=True)
print("Then proceed to v50/v51 (GP-based data augmentation)", flush=True)
print("=" * 80, flush=True)
