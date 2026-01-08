"""
MALLORN v42: v34a (Bazin) + Conservative Pseudo-Labeling (Technique #7)

Testing pseudo-labeling with >0.99 confidence threshold.

Key insight from PLAsTiCC 1st place:
- Use VERY conservative threshold (>0.99) to avoid noisy labels
- Only add high-confidence predictions to training set
- Retrain with expanded dataset
- Can iterate 2-3 times for progressive improvement

Previous attempt (v28b) failed with 0.85 threshold (too aggressive)
Expected gain: +1-2% from additional high-quality training samples
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
print("MALLORN v42: Bazin + Conservative Pseudo-Labeling (Technique #7)", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD v34a FEATURES AND PREDICTIONS
# ====================
print("\n1. Loading v34a (Bazin) artifacts...", flush=True)

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

# Get v34a test predictions
test_preds_v34a = v34a['test_preds']
print(f"   v34a test predictions loaded: {len(test_preds_v34a)} samples", flush=True)

# ====================
# 2. SELECT HIGH-CONFIDENCE PSEUDO-LABELS
# ====================
print("\n2. Selecting high-confidence pseudo-labels...", flush=True)

confidence_threshold = 0.99
print(f"   Confidence threshold: {confidence_threshold}", flush=True)

# High-confidence TDEs (prob > 0.99)
high_conf_tde = test_preds_v34a > confidence_threshold
high_conf_tde_ids = [test_ids[i] for i in range(len(test_ids)) if high_conf_tde[i]]
high_conf_tde_probs = test_preds_v34a[high_conf_tde]

# High-confidence non-TDEs (prob < 0.01)
high_conf_nontde = test_preds_v34a < (1 - confidence_threshold)
high_conf_nontde_ids = [test_ids[i] for i in range(len(test_ids)) if high_conf_nontde[i]]
high_conf_nontde_probs = test_preds_v34a[high_conf_nontde]

n_pseudo_tde = len(high_conf_tde_ids)
n_pseudo_nontde = len(high_conf_nontde_ids)
n_pseudo_total = n_pseudo_tde + n_pseudo_nontde

print(f"   High-confidence TDEs: {n_pseudo_tde}", flush=True)
print(f"   High-confidence non-TDEs: {n_pseudo_nontde}", flush=True)
print(f"   Total pseudo-labels: {n_pseudo_total} ({100*n_pseudo_total/len(test_ids):.1f}% of test set)", flush=True)

if n_pseudo_total == 0:
    print("\n   ERROR: No high-confidence predictions found!", flush=True)
    print("   Try lowering confidence threshold (e.g., 0.95 or 0.90)", flush=True)
    sys.exit(1)

if n_pseudo_total < 20:
    print(f"\n   WARNING: Only {n_pseudo_total} pseudo-labels found", flush=True)
    print("   This may not provide significant benefit", flush=True)

# ====================
# 3. RECONSTRUCT v34a FEATURES
# ====================
print("\n3. Reconstructing v34a feature set...", flush=True)

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

print(f"   v34a features reconstructed: {len(train_combined.columns)-1}", flush=True)

# ====================
# 4. ADD PSEUDO-LABELS TO TRAINING SET
# ====================
print("\n4. Adding pseudo-labels to training set...", flush=True)

# Create pseudo-labeled dataset from test set
pseudo_ids = high_conf_tde_ids + high_conf_nontde_ids
pseudo_labels = [1] * n_pseudo_tde + [0] * n_pseudo_nontde

# Extract pseudo-labeled samples from test set
test_combined_pseudo = test_combined[test_combined['object_id'].isin(pseudo_ids)].copy()
test_combined_pseudo['target'] = test_combined_pseudo['object_id'].map(dict(zip(pseudo_ids, pseudo_labels)))

# Combine original training + pseudo-labels
train_original_ids = train_combined['object_id'].tolist()
train_original_labels = y.tolist()

train_combined_with_target = train_combined.copy()
train_combined_with_target['target'] = train_original_labels

# Concatenate
train_expanded = pd.concat([train_combined_with_target, test_combined_pseudo], ignore_index=True)

print(f"   Original training samples: {len(train_combined)} (TDEs: {np.sum(y==1)}, non-TDEs: {np.sum(y==0)})", flush=True)
print(f"   Pseudo-labeled samples: {n_pseudo_total} (TDEs: {n_pseudo_tde}, non-TDEs: {n_pseudo_nontde})", flush=True)
print(f"   Expanded training samples: {len(train_expanded)} (TDEs: {np.sum(train_expanded['target']==1)}, non-TDEs: {np.sum(train_expanded['target']==0)})", flush=True)

# Prepare for training
X_train_expanded = train_expanded.drop(columns=['object_id', 'target']).values
y_train_expanded = train_expanded['target'].values
feature_names = [c for c in train_expanded.columns if c not in ['object_id', 'target']]

X_test = test_combined.drop(columns=['object_id']).values

print(f"   Training shape: {X_train_expanded.shape}", flush=True)

# ====================
# 5. TRAIN XGBOOST WITH EXPANDED DATASET
# ====================
print("\n5. Training XGBoost with pseudo-labeled data...", flush=True)

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
    'scale_pos_weight': len(y_train_expanded[y_train_expanded==0]) / len(y_train_expanded[y_train_expanded==1]),
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

print(f"   scale_pos_weight: {xgb_params['scale_pos_weight']:.2f}", flush=True)

# Important: Only evaluate on ORIGINAL training set, not pseudo-labeled samples
# This ensures we measure true generalization, not memorization of pseudo-labels
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_ids))  # Only original training samples
test_preds = np.zeros((len(X_test), n_folds))
feature_importance = np.zeros(len(feature_names))

for fold, (train_idx, val_idx) in enumerate(skf.split(train_combined.drop(columns=['object_id']).values, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    # Training: Original training fold + ALL pseudo-labels
    X_tr_original = train_combined.drop(columns=['object_id']).values[train_idx]
    y_tr_original = y[train_idx]

    # Add pseudo-labeled samples to training fold
    X_tr_pseudo = test_combined_pseudo.drop(columns=['object_id', 'target']).values
    y_tr_pseudo = test_combined_pseudo['target'].values

    X_tr_combined = np.vstack([X_tr_original, X_tr_pseudo])
    y_tr_combined = np.concatenate([y_tr_original, y_tr_pseudo])

    # Validation: Only original validation fold (NO pseudo-labels)
    X_val = train_combined.drop(columns=['object_id']).values[val_idx]
    y_val = y[val_idx]

    print(f"      Training samples: {len(X_tr_combined)} ({len(X_tr_original)} original + {len(X_tr_pseudo)} pseudo)", flush=True)
    print(f"      Validation samples: {len(X_val)} (original only)", flush=True)

    dtrain = xgb.DMatrix(X_tr_combined, label=y_tr_combined, feature_names=feature_names)
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

submission_path = base_path / 'submissions/submission_v42_pseudolabel.csv'
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
    'pseudo_label_config': {
        'confidence_threshold': confidence_threshold,
        'n_pseudo_tde': n_pseudo_tde,
        'n_pseudo_nontde': n_pseudo_nontde,
        'pseudo_ids': pseudo_ids,
        'pseudo_labels': pseudo_labels
    }
}

with open(base_path / 'data/processed/v42_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v42 (Bazin + Pseudo-Labeling) Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"v34a (Bazin only): OOF F1 = {v34a['oof_f1']:.4f}, LB F1 = 0.6907 (best)", flush=True)

change_vs_v34a_oof = (best_f1 - v34a['oof_f1']) * 100 / v34a['oof_f1']
print(f"Change vs v34a (OOF): {change_vs_v34a_oof:+.2f}% ({best_f1 - v34a['oof_f1']:+.4f})", flush=True)

if best_f1 > v34a['oof_f1']:
    print("SUCCESS: Pseudo-labeling improved over v34a!", flush=True)
    print(f"Expected LB: 0.69-0.71 (if OOF gain transfers)", flush=True)
else:
    print("Pseudo-labeling did not improve on OOF", flush=True)
    print("Test on LB to check if it helps generalization anyway", flush=True)

print(f"\nKey Insight: Pseudo-labeling adds {n_pseudo_total} high-confidence samples", flush=True)
print(f"  - Expanded training: {len(train_ids)} → {len(train_expanded)} (+{n_pseudo_total})", flush=True)
print(f"  - TDE ratio: {100*np.sum(y==1)/len(y):.1f}% → {100*np.sum(y_train_expanded==1)/len(y_train_expanded):.1f}%", flush=True)
print(f"  - Conservative threshold (0.99) ensures high-quality pseudo-labels", flush=True)

print("\nNEXT STEP: Test v42 on leaderboard", flush=True)
print("Target: Beat v34a (LB 0.6907)", flush=True)
print("=" * 80, flush=True)
