"""
MALLORN v48: v34a (Bazin) + Time-to-Decline Features

Adding the KEY winning feature from PLAsTiCC 1st place (Kyle Boone).

Research basis: arxiv.org/abs/1907.04690 (avocado paper)
- "computed the time to decline to 20% of the maximum light"
- This was THE critical feature for winning PLAsTiCC

Features added:
- Time from peak to [80%, 60%, 40%, 20%, 10%] of peak flux
- Per band (u, g, r, i, z, y) = 30 features
- Decline velocity (flux drop per day) = 6 features
- Total: 36 new features

Expected gain: +1-2% (Boone's key winning feature)
Target: Push toward LB 0.70+
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
print("MALLORN v48: Bazin + Time-to-Decline (Boone's Key Feature)", flush=True)
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

# Load v34a for reference
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

print(f"   v34a baseline: OOF F1=0.6667, LB F1=0.6907 (best)", flush=True)
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
train_v34a = train_v21.merge(train_bazin, on='object_id', how='left')
test_v34a = test_v21.merge(test_bazin, on='object_id', how='left')

print(f"   v34a features reconstructed: {len(train_v34a.columns)-1}", flush=True)

# ====================
# 2. EXTRACT TIME-TO-DECLINE FEATURES
# ====================
print("\n2. Extracting time-to-decline features (Boone's key feature)...", flush=True)

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
print(f"   ({len(train_v34a.columns)-1} v34a + {len(train_decline.columns)-1} time-to-decline)", flush=True)
print(f"   Training shape: {X_train.shape}", flush=True)

# ====================
# 4. TRAIN XGBOOST
# ====================
print("\n4. Training XGBoost with 5-fold CV...", flush=True)

# Same hyperparameters as v34a
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

# Analyze time-to-decline importance
decline_cols = [c for c in feature_names if 'decline' in c]
decline_importance = importance_df_result[importance_df_result['feature'].isin(decline_cols)]

if len(decline_importance) > 0:
    total_decline_importance = decline_importance['importance'].sum()
    total_importance = importance_df_result['importance'].sum()
    print(f"\n   Time-to-decline features: {100*total_decline_importance/total_importance:.1f}% of model importance", flush=True)

    print("\n   Top 10 Time-to-Decline Features:")
    for idx, row in decline_importance.head(10).iterrows():
        rank = list(importance_df_result.index).index(idx) + 1
        print(f"      {rank:3d}. {row['feature']:40s} {row['importance']:8.1f}", flush=True)

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

submission_path = base_path / 'submissions/submission_v48_time_to_decline.csv'
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

with open(base_path / 'data/processed/v48_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# Cache time-to-decline for reuse
decline_cache = {
    'train': train_decline,
    'test': test_decline
}
with open(base_path / 'data/processed/time_to_decline_cache.pkl', 'wb') as f:
    pickle.dump(decline_cache, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v48 (Bazin + Time-to-Decline) Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"v34a (Bazin only): OOF F1 = {v34a['oof_f1']:.4f}, LB F1 = 0.6907", flush=True)

change_vs_v34a_oof = (best_f1 - v34a['oof_f1']) * 100 / v34a['oof_f1']
print(f"Change vs v34a (OOF): {change_vs_v34a_oof:+.2f}% ({best_f1 - v34a['oof_f1']:+.4f})", flush=True)

if best_f1 > v34a['oof_f1']:
    print("\nSUCCESS: Time-to-decline improved over v34a!", flush=True)
    expected_lb = 0.6907 * (best_f1 / v34a['oof_f1'])
    print(f"Expected LB: {expected_lb:.4f} (if OOF gain transfers)", flush=True)
else:
    print("\nTime-to-decline did not improve on OOF", flush=True)
    print("Test on LB to check if features help generalization", flush=True)

print("\nKey Insight: Time-to-decline (Boone's winning feature)", flush=True)
print("  - Time from peak to [80%, 60%, 40%, 20%, 10%] of peak flux", flush=True)
print("  - Decline velocity (flux drop per day)", flush=True)
print("  - TDEs decline slowly (months), SNe decline fast (weeks)", flush=True)
print("  - This was THE key feature for PLAsTiCC 1st place", flush=True)

print("\nNEXT STEP: Test v48 on leaderboard", flush=True)
print("Then proceed to v49 (Time-to-Decline for LightGBM)", flush=True)
print("Then v50/v51 (GP-based data augmentation)", flush=True)
print("Target: LB 1.0 (perfect classification)", flush=True)
print("=" * 80, flush=True)
