"""
MALLORN v41: v34a (Bazin) + Focal Loss (Technique #6)

Testing Focal Loss to better handle class imbalance.

Focal Loss: FL(p_t) = -(1 - p_t)^γ * log(p_t)
- Down-weights easy examples (high confidence correct predictions)
- Up-weights hard examples (low confidence or misclassified)
- γ=2 (standard value from Lin et al. 2017)

Class imbalance: 148 TDEs vs 2895 non-TDEs (5.1% positive)

Expected gain: +1-3% from better handling of hard examples
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
print("MALLORN v41: Bazin + Focal Loss (Technique #6)", flush=True)
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
# 2. SETUP FOCAL LOSS
# ====================
print("\n2. Setting up Focal Loss...", flush=True)

from models.focal_loss import focal_loss_objective, focal_loss_eval

gamma = 2.0  # Standard value from Lin et al. 2017
focal_obj = focal_loss_objective(gamma=gamma)
focal_eval = focal_loss_eval(gamma=gamma)

print(f"   Focal Loss gamma: {gamma}", flush=True)
print(f"   Class balance: {np.sum(y==1)} TDEs vs {np.sum(y==0)} non-TDEs", flush=True)
print(f"   Positive rate: {100*np.mean(y):.1f}%", flush=True)

# ====================
# 3. TRAIN XGBOOST WITH FOCAL LOSS
# ====================
print("\n3. Training XGBoost with Focal Loss...", flush=True)

xgb_params = {
    # Note: No 'objective' or 'eval_metric' - we provide custom ones
    'max_depth': 5,
    'learning_rate': 0.025,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    # No scale_pos_weight - Focal Loss handles imbalance
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
    'disable_default_eval_metric': 1  # Required for custom eval
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
        obj=focal_obj,  # Custom focal loss objective
        custom_metric=focal_eval,  # Custom focal loss eval
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # Get predictions (raw logits)
    oof_logits = model.predict(dval)
    test_logits = model.predict(dtest)

    # Convert to probabilities
    oof_preds[val_idx] = 1.0 / (1.0 + np.exp(-oof_logits))
    test_preds[:, fold-1] = 1.0 / (1.0 + np.exp(-test_logits))

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

submission_path = base_path / 'submissions/submission_v41_focal_loss.csv'
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
    'focal_loss_gamma': gamma
}

with open(base_path / 'data/processed/v41_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v41 (Bazin + Focal Loss) Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"v34a (Bazin + log loss): OOF F1 = {v34a['oof_f1']:.4f}, LB F1 = 0.6907 (best)", flush=True)

change_vs_v34a_oof = (best_f1 - v34a['oof_f1']) * 100 / v34a['oof_f1']
print(f"Change vs v34a (OOF): {change_vs_v34a_oof:+.2f}% ({best_f1 - v34a['oof_f1']:+.4f})", flush=True)

if best_f1 > v34a['oof_f1']:
    print("SUCCESS: Focal Loss improved over standard log loss!", flush=True)
    print(f"Expected LB: 0.69-0.71 (if OOF gain transfers)", flush=True)
else:
    print("Focal Loss did not improve on OOF", flush=True)
    print("Test on LB to check if it helps with hard examples anyway", flush=True)

print("\nKey Insight: Focal Loss targets hard examples", flush=True)
print("  - Down-weights easy correct predictions", flush=True)
print("  - Up-weights misclassified or low-confidence examples", flush=True)
print("  - May improve even if OOF stays similar", flush=True)

print("\nNEXT STEP: Test v41 on leaderboard", flush=True)
print("Target: Beat v34a (LB 0.6907)", flush=True)
print("=" * 80, flush=True)
