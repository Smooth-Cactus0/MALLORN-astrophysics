"""
MALLORN v39b: v34a (Bazin) + Adversarial Sample Weights (Technique #5)

Testing adversarial validation reweighting in isolation.

Key finding from adversarial validation:
- AUC = 0.6097 (significant distribution shift detected)
- Test set has +66% longer rise times, +80% higher asymmetry
- Sample weights range: [1.34, 1.92], mean = 1.52

Expected gain: +1-2% from better generalization to test distribution
Target: LB F1 > 0.70
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
print("MALLORN v39b: Bazin + Adversarial Weights (Technique #5)", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD v34a FEATURES
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

# Load v34a artifacts
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

print(f"   v34a baseline: OOF F1={v34a['oof_f1']:.4f}, LB F1=0.6907", flush=True)
print(f"   Features: {len(v34a['feature_names'])}", flush=True)

# Reconstruct v34a features (same as v34a training)
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

# Load TDE and GP features (part of v21 baseline)
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
print(f"   ({len(train_v21.columns)-1} v21 + {len(train_bazin.columns)-1} Bazin)", flush=True)
print(f"   Training shape: {X_train.shape}", flush=True)

# ====================
# 2. LOAD ADVERSARIAL WEIGHTS
# ====================
print("\n2. Loading adversarial sample weights...", flush=True)

with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)

sample_weights = adv_results['sample_weights']
adversarial_auc = adv_results['auc']

print(f"   Adversarial AUC: {adversarial_auc:.4f} (distribution shift detected)", flush=True)
print(f"   Sample weights: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]", flush=True)
print(f"   Mean weight: {sample_weights.mean():.2f}", flush=True)

# Sanity check
assert len(sample_weights) == len(X_train), "Sample weights length mismatch!"

# ====================
# 3. TRAIN XGBOOST WITH SAMPLE WEIGHTS
# ====================
print("\n3. Training XGBoost with adversarial weights...", flush=True)

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

    # Get sample weights for this fold's training data
    weights_tr = sample_weights[train_idx]

    # Create DMatrix with sample weights
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=weights_tr, feature_names=feature_names)
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

# Check if adversarial discriminative features changed in importance
print("\n   Adversarial discriminative features:", flush=True)
top_adv_features = ['all_rise_time', 'all_asymmetry', 'r_rest_fade', 'i_rest_fade', 'i_rest_rise']
for feat in top_adv_features:
    if feat in feature_names:
        feat_importance = importance_df_result[importance_df_result['feature'] == feat]
        if len(feat_importance) > 0:
            rank = list(importance_df_result.index).index(feat_importance.index[0]) + 1
            print(f"      {feat}: rank {rank}, importance {feat_importance['importance'].iloc[0]:.1f}", flush=True)

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

submission_path = base_path / 'submissions/submission_v39b_adversarial.csv'
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
    'sample_weights_used': sample_weights,
    'adversarial_auc': adversarial_auc
}

with open(base_path / 'data/processed/v39b_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v39b (Bazin + Adversarial Weights) Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"v34a (Bazin only): OOF F1 = {v34a['oof_f1']:.4f}, LB F1 = 0.6907", flush=True)

change_vs_v34a_oof = (best_f1 - v34a['oof_f1']) * 100 / v34a['oof_f1']
print(f"Change vs v34a (OOF): {change_vs_v34a_oof:+.2f}% ({best_f1 - v34a['oof_f1']:+.4f})", flush=True)

if best_f1 > v34a['oof_f1']:
    print("SUCCESS: Adversarial weighting improved over v34a!", flush=True)
    print(f"Expected LB: 0.69-0.71 (if OOF gain transfers)", flush=True)
else:
    print("Adversarial weighting did not improve on OOF", flush=True)
    print("Test on LB to check if it helps generalization anyway", flush=True)

print("\nKey Insight: Adversarial weights target distribution shift", flush=True)
print("  - Test set has +66% longer rise times, +80% higher asymmetry", flush=True)
print("  - Weights up-weight these patterns by 1.3-1.9x during training", flush=True)
print("  - Should reduce OOF-LB gap even if OOF score stays similar", flush=True)

print("\nNEXT STEP: Test v39b on leaderboard", flush=True)
print("If successful, combine with Fourier features in v40", flush=True)
print("=" * 80, flush=True)
