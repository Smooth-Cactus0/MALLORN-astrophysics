"""
MALLORN v72: Feature Subtraction Experiment

Hypothesis: v34a's 224 features may contain noise that hurts generalization.
Simpler models often generalize better.

Experiment:
- Test top 50, 100, 150, and all (224) features
- Use same v34a parameters and CV setup
- Compare OOF F1 scores
- Generate submission for best performing count

Expected: Fewer features = lower OOF but better LB generalization
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
print("MALLORN v72: Feature Subtraction Experiment", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD DATA & v34a FEATURE IMPORTANCE
# ====================
print("\n1. Loading data and v34a artifacts...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDE)", flush=True)

# Load v34a feature importance
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a_artifacts = pickle.load(f)

v34a_importance = v34a_artifacts['feature_importance']
v34a_features = v34a_artifacts['feature_names']
print(f"   v34a total features: {len(v34a_features)}", flush=True)
print(f"   v34a OOF F1: {v34a_artifacts['oof_f1']:.4f}", flush=True)

# Sort by importance
top_features = v34a_importance.sort_values('importance', ascending=False)['feature'].tolist()

# ====================
# 2. LOAD ALL FEATURE DATA
# ====================
print("\n2. Loading feature data...", flush=True)

# Load all feature sources (same as v34a)
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']

tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']

with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']

with open(base_path / 'data/processed/bazin_features_cache.pkl', 'rb') as f:
    bazin_cache = pickle.load(f)
train_bazin = bazin_cache['train']
test_bazin = bazin_cache['test']

# Merge all features
train_all = train_base.merge(train_tde, on='object_id', how='left')
train_all = train_all.merge(train_gp2d, on='object_id', how='left')
train_all = train_all.merge(train_bazin, on='object_id', how='left')

test_all = test_base.merge(test_tde, on='object_id', how='left')
test_all = test_all.merge(test_gp2d, on='object_id', how='left')
test_all = test_all.merge(test_bazin, on='object_id', how='left')

# Filter to features that exist in v34a
available_features = [f for f in top_features if f in train_all.columns and f in test_all.columns]
print(f"   Available features: {len(available_features)}", flush=True)

# ====================
# 3. FEATURE COUNT EXPERIMENT
# ====================
print("\n" + "=" * 80, flush=True)
print("FEATURE COUNT EXPERIMENT", flush=True)
print("=" * 80, flush=True)

feature_counts = [50, 100, 150, len(available_features)]
results = {}

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

for n_features in feature_counts:
    print(f"\n--- Testing top {n_features} features ---", flush=True)

    # Select top N features
    selected = available_features[:n_features]

    X_train = train_all[selected].values
    X_test = test_all[selected].values

    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros((len(X_test), n_folds))
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=selected)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=selected)
        dtest = xgb.DMatrix(X_test, feature_names=selected)

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        oof_preds[val_idx] = model.predict(dval)
        test_preds[:, fold-1] = model.predict(dtest)

        # Fold F1
        best_f1 = 0
        for t in np.linspace(0.03, 0.3, 50):
            f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)

    # Overall OOF F1
    best_f1 = 0
    best_thresh = 0.1
    for t in np.linspace(0.03, 0.3, 200):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # Confusion matrix
    final_preds = (oof_preds > best_thresh).astype(int)
    tp = np.sum((final_preds == 1) & (y == 1))
    fp = np.sum((final_preds == 1) & (y == 0))
    fn = np.sum((final_preds == 0) & (y == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    results[n_features] = {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'fold_f1s': fold_f1s,
        'fold_std': np.std(fold_f1s),
        'precision': precision,
        'recall': recall,
        'tp': tp, 'fp': fp, 'fn': fn,
        'oof_preds': oof_preds,
        'test_preds': test_preds.mean(axis=1),
        'features': selected
    }

    print(f"   OOF F1: {best_f1:.4f} @ thresh={best_thresh:.3f}", flush=True)
    print(f"   Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}", flush=True)
    print(f"   Fold Std: {np.std(fold_f1s):.4f}", flush=True)
    print(f"   TP={tp}, FP={fp}, FN={fn}", flush=True)
    print(f"   Precision={precision:.4f}, Recall={recall:.4f}", flush=True)

# ====================
# 4. RESULTS SUMMARY
# ====================
print("\n" + "=" * 80, flush=True)
print("RESULTS SUMMARY", flush=True)
print("=" * 80, flush=True)

print(f"\n   {'Features':<12} {'OOF F1':<10} {'Fold Std':<10} {'Precision':<10} {'Recall':<10}", flush=True)
print(f"   {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}", flush=True)

for n_feat in feature_counts:
    r = results[n_feat]
    print(f"   {n_feat:<12} {r['oof_f1']:<10.4f} {r['fold_std']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f}", flush=True)

# Find best by OOF F1
best_by_oof = max(results.keys(), key=lambda k: results[k]['oof_f1'])
# Find best by fold stability (lowest std)
best_by_stability = min(results.keys(), key=lambda k: results[k]['fold_std'])

print(f"\n   Best by OOF F1: {best_by_oof} features (F1={results[best_by_oof]['oof_f1']:.4f})", flush=True)
print(f"   Best by stability: {best_by_stability} features (std={results[best_by_stability]['fold_std']:.4f})", flush=True)

# ====================
# 5. GENERATE SUBMISSIONS
# ====================
print("\n" + "=" * 80, flush=True)
print("SUBMISSIONS", flush=True)
print("=" * 80, flush=True)

for n_feat in feature_counts:
    r = results[n_feat]
    test_binary = (r['test_preds'] > r['threshold']).astype(int)

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': test_binary
    })

    submission_path = base_path / f'submissions/submission_v72_top{n_feat}.csv'
    submission.to_csv(submission_path, index=False)

    print(f"   Top {n_feat}: {submission_path.name} (TDEs: {test_binary.sum()})", flush=True)

# Save artifacts
with open(base_path / 'data/processed/v72_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

# ====================
# 6. RECOMMENDATION
# ====================
print("\n" + "=" * 80, flush=True)
print("RECOMMENDATION", flush=True)
print("=" * 80, flush=True)

print(f"""
   Based on the results:

   1. If OOF F1 correlates with LB (as expected):
      -> Use top {best_by_oof} features

   2. If fold stability matters for generalization:
      -> Use top {best_by_stability} features (lowest variance)

   3. Given v34a's OOF (0.6667) -> LB (0.6907) pattern:
      -> Lower OOF might actually mean BETTER LB generalization
      -> Consider testing the SMALLEST feature set first

   v34a reference: 224 features, OOF=0.6667, LB=0.6907
""", flush=True)

print("=" * 80, flush=True)
print("v72 Complete", flush=True)
print("=" * 80, flush=True)
