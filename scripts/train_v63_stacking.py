"""
MALLORN v63: Stacking Meta-Learner

Combines multiple OOF predictions as meta-features for a second-level classifier.

Layer 1 (Base models):
- v34a Bazin XGBoost (OOF F1=0.6667, LB=0.6907)
- Multi-class probabilities (TDE/AGN/SN_Ia/SN_CC)
- v62 predictions

Layer 2 (Meta-learner):
- Logistic Regression or XGBoost on stacked OOF predictions

Key insight: Different models capture different patterns.
- v34a is best at Bazin parametric features
- Multi-class captures class structure
- Combining them may extract more signal

Target: OOF F1 > 0.68, LB F1 > 0.70
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v63: Stacking Meta-Learner", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD BASE MODEL PREDICTIONS
# ====================
print("\n1. Loading base model OOF predictions...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

print(f"   Training samples: {len(y)}", flush=True)
print(f"   TDEs: {np.sum(y)} ({100*np.sum(y)/len(y):.1f}%)", flush=True)

# Load v34a (Bazin) - our best single model
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

v34a_oof = v34a['oof_preds']
v34a_test = v34a['test_preds']
print(f"   v34a (Bazin): OOF F1={v34a['oof_f1']:.4f}, {len(v34a_oof)} samples", flush=True)

# Load v62 (multi-class)
try:
    with open(base_path / 'data/processed/v62_artifacts.pkl', 'rb') as f:
        v62 = pickle.load(f)
    v62_oof = v62['oof_preds']
    v62_test = v62['test_preds']
    v62_mc_oof = v62['oof_multiclass']
    v62_mc_test = v62['test_multiclass']
    print(f"   v62 (multi-class): OOF F1={v62['oof_f1']:.4f}", flush=True)
    has_v62 = True
except:
    print("   v62 artifacts not found, skipping", flush=True)
    has_v62 = False

# Load additional model predictions if available
additional_models = []
for version in ['v39b', 'v60a', 'v41', 'v40']:
    try:
        with open(base_path / f'data/processed/{version}_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        if 'oof_preds' in artifacts and len(artifacts['oof_preds']) == len(y):
            additional_models.append({
                'name': version,
                'oof': artifacts['oof_preds'],
                'test': artifacts['test_preds'],
                'f1': artifacts.get('oof_f1', 0)
            })
            print(f"   {version}: OOF F1={artifacts.get('oof_f1', 'N/A')}", flush=True)
    except:
        pass

print(f"\n   Total base models: {2 + len(additional_models) if has_v62 else 1 + len(additional_models)}", flush=True)

# ====================
# 2. CREATE META-FEATURES
# ====================
print("\n2. Creating meta-features...", flush=True)

# Start with v34a
meta_train = [v34a_oof]
meta_test = [v34a_test]
meta_names = ['v34a_prob']

# Add multi-class probabilities from v62
if has_v62:
    meta_train.append(v62_oof)
    meta_test.append(v62_test)
    meta_names.append('v62_prob')

    # Add individual class probabilities
    for i, class_name in enumerate(['AGN', 'SN_CC', 'SN_Ia', 'TDE']):
        meta_train.append(v62_mc_oof[:, i])
        meta_test.append(v62_mc_test[:, i])
        meta_names.append(f'mc_prob_{class_name.lower()}')

# Add additional models
for model in additional_models:
    meta_train.append(model['oof'])
    meta_test.append(model['test'])
    meta_names.append(f"{model['name']}_prob")

X_meta_train = np.column_stack(meta_train)
X_meta_test = np.column_stack(meta_test)

print(f"   Meta-features: {X_meta_train.shape[1]}", flush=True)
print(f"   Features: {meta_names}", flush=True)

# ====================
# 3. TRAIN META-LEARNER
# ====================
print("\n3. Training meta-learner...", flush=True)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Try multiple meta-learners
results = {}

# Method 1: Logistic Regression
print("\n   Method 1: Logistic Regression...", flush=True)
oof_lr = np.zeros(len(y))
test_lr = np.zeros(len(test_ids))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_meta_train)
X_test_scaled = scaler.transform(X_meta_test)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    lr = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.5, random_state=42)
    lr.fit(X_tr, y_tr)

    oof_lr[val_idx] = lr.predict_proba(X_val)[:, 1]
    test_lr += lr.predict_proba(X_test_scaled)[:, 1] / n_folds

best_f1_lr = 0
best_thresh_lr = 0.5
for t in np.linspace(0.1, 0.9, 100):
    preds = (oof_lr > t).astype(int)
    f1 = f1_score(y, preds)
    if f1 > best_f1_lr:
        best_f1_lr = f1
        best_thresh_lr = t

print(f"      OOF F1: {best_f1_lr:.4f} @ threshold={best_thresh_lr:.3f}", flush=True)
results['LogisticRegression'] = {'oof': oof_lr, 'test': test_lr, 'f1': best_f1_lr, 'thresh': best_thresh_lr}

# Method 2: XGBoost Meta-Learner
print("\n   Method 2: XGBoost...", flush=True)
oof_xgb = np.zeros(len(y))
test_xgb = np.zeros((len(test_ids), n_folds))

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,  # Shallow for meta-learner
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'random_state': 42,
    'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta_train, y), 1):
    X_tr, X_val = X_meta_train[train_idx], X_meta_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=meta_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=meta_names)
    dtest = xgb.DMatrix(X_meta_test, feature_names=meta_names)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=200,
        evals=[(dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    oof_xgb[val_idx] = model.predict(dval)
    test_xgb[:, fold-1] = model.predict(dtest)

best_f1_xgb = 0
best_thresh_xgb = 0.5
for t in np.linspace(0.1, 0.9, 100):
    preds = (oof_xgb > t).astype(int)
    f1 = f1_score(y, preds)
    if f1 > best_f1_xgb:
        best_f1_xgb = f1
        best_thresh_xgb = t

print(f"      OOF F1: {best_f1_xgb:.4f} @ threshold={best_thresh_xgb:.3f}", flush=True)
results['XGBoost'] = {'oof': oof_xgb, 'test': test_xgb.mean(axis=1), 'f1': best_f1_xgb, 'thresh': best_thresh_xgb}

# Method 3: Simple Average
print("\n   Method 3: Simple Average of base models...", flush=True)
oof_avg = np.mean([meta_train[i] for i in range(min(3, len(meta_train)))], axis=0)
test_avg = np.mean([meta_test[i] for i in range(min(3, len(meta_test)))], axis=0)

best_f1_avg = 0
best_thresh_avg = 0.5
for t in np.linspace(0.1, 0.9, 100):
    preds = (oof_avg > t).astype(int)
    f1 = f1_score(y, preds)
    if f1 > best_f1_avg:
        best_f1_avg = f1
        best_thresh_avg = t

print(f"      OOF F1: {best_f1_avg:.4f} @ threshold={best_thresh_avg:.3f}", flush=True)
results['SimpleAverage'] = {'oof': oof_avg, 'test': test_avg, 'f1': best_f1_avg, 'thresh': best_thresh_avg}

# Method 4: Rank Average
print("\n   Method 4: Rank Average...", flush=True)
from scipy.stats import rankdata

rank_oof = np.zeros(len(y))
rank_test = np.zeros(len(test_ids))

for i in range(len(meta_train)):
    rank_oof += rankdata(meta_train[i]) / len(meta_train[i])
    rank_test += rankdata(meta_test[i]) / len(meta_test[i])

rank_oof /= len(meta_train)
rank_test /= len(meta_train)

best_f1_rank = 0
best_thresh_rank = 0.5
for t in np.linspace(0.3, 0.7, 100):
    preds = (rank_oof > t).astype(int)
    f1 = f1_score(y, preds)
    if f1 > best_f1_rank:
        best_f1_rank = f1
        best_thresh_rank = t

print(f"      OOF F1: {best_f1_rank:.4f} @ threshold={best_thresh_rank:.3f}", flush=True)
results['RankAverage'] = {'oof': rank_oof, 'test': rank_test, 'f1': best_f1_rank, 'thresh': best_thresh_rank}

# ====================
# 4. SELECT BEST METHOD
# ====================
print("\n" + "=" * 80, flush=True)
print("RESULTS SUMMARY", flush=True)
print("=" * 80, flush=True)

print(f"\n   v34a baseline: OOF F1 = {v34a['oof_f1']:.4f}", flush=True)
print(f"\n   Stacking methods:", flush=True)

best_method = None
best_f1 = 0
for method, result in results.items():
    improvement = 100 * (result['f1'] - v34a['oof_f1']) / v34a['oof_f1']
    status = "+" if improvement > 0 else ""
    print(f"      {method}: OOF F1 = {result['f1']:.4f} ({status}{improvement:.2f}%)", flush=True)
    if result['f1'] > best_f1:
        best_f1 = result['f1']
        best_method = method

print(f"\n   Best method: {best_method} (OOF F1 = {best_f1:.4f})", flush=True)

# ====================
# 5. GENERATE SUBMISSION
# ====================
print("\n" + "=" * 80, flush=True)
print("GENERATING SUBMISSION", flush=True)
print("=" * 80, flush=True)

best_result = results[best_method]
test_probs = best_result['test']
test_binary = (test_probs > best_result['thresh']).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

submission_path = base_path / 'submissions' / 'submission_v63_stacking.csv'
submission.to_csv(submission_path, index=False)

print(f"   Method: {best_method}", flush=True)
print(f"   Threshold: {best_result['thresh']:.3f}", flush=True)
print(f"   Predicted TDEs: {test_binary.sum()}", flush=True)
print(f"   Submission saved: {submission_path}", flush=True)

# Also save best individual submissions
for method, result in results.items():
    test_binary = (result['test'] > result['thresh']).astype(int)
    sub = pd.DataFrame({'object_id': test_ids, 'target': test_binary})
    sub.to_csv(base_path / 'submissions' / f'submission_v63_{method.lower()}.csv', index=False)

# Save artifacts
artifacts = {
    'results': results,
    'meta_names': meta_names,
    'best_method': best_method,
    'best_f1': best_f1
}

with open(base_path / 'data/processed/v63_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v63 Complete", flush=True)
print(f"Best stacking: {best_method} with OOF F1 = {best_f1:.4f}", flush=True)
print(f"v34a baseline: OOF F1 = 0.6667, LB F1 = 0.6907", flush=True)
if best_f1 > v34a['oof_f1']:
    print(f"IMPROVEMENT: +{100*(best_f1 - v34a['oof_f1'])/v34a['oof_f1']:.2f}%", flush=True)
print("=" * 80, flush=True)
