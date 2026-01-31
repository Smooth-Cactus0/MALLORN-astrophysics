# LightGBM Generalization for Ensemble Diversity

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a LightGBM model that achieves ~0.68+ LB F1 (close to XGBoost's 0.6907) to provide architectural diversity for ensembling.

**Architecture:** Use v34a's proven feature set with aggressive regularization strategies (DART boosting, heavy L1/L2, aggressive subsampling) to force LightGBM to generalize instead of overfitting. The key insight is that v77 achieved higher OOF (0.6886) but lower LB (0.6714) than v34a, indicating overfitting to training patterns.

**Tech Stack:** LightGBM, Optuna, scikit-learn, pandas, numpy

---

## Context Summary

| Model | OOF F1 | LB F1 | Gap | Notes |
|-------|--------|-------|-----|-------|
| v34a XGBoost | 0.6667 | **0.6907** | +0.024 | Best LB, our target |
| v77 LightGBM | 0.6886 | 0.6714 | -0.017 | Overfits |
| v79c XGBoost | 0.6834 | 0.6891 | +0.006 | 70 features, 2nd best LB |

**Key Insight:** Lower OOF often means better LB. We need LightGBM to "underfit" more.

---

### Task 1: Create Base LightGBM Script with Heavy Regularization

**Files:**
- Create: `scripts/train_v110_lgbm_regularized.py`

**Step 1: Create the training script**

```python
"""
MALLORN v110: LightGBM with Heavy Regularization for Generalization

Goal: Match v34a's LB performance (0.6907) with LightGBM for ensemble diversity.

Strategy:
- Use v34a's exact feature set (proven to generalize)
- Heavy L1/L2 regularization (5-10x higher than v77)
- Low num_leaves (15-31 vs v77's 15-63)
- Aggressive feature/bagging fraction (0.5 vs 0.8)
- DART boosting for additional regularization
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

print("=" * 80)
print("MALLORN v110: LightGBM Heavy Regularization")
print("=" * 80)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...")

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDE)")

# ====================
# 2. LOAD v34a FEATURES (proven to generalize)
# ====================
print("\n2. Loading v34a features...")

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a_artifacts = pickle.load(f)

feature_names = v34a_artifacts['feature_names']
print(f"   v34a features: {len(feature_names)}")

# Load all feature data
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

# Merge all
train_all = train_base.merge(train_tde, on='object_id', how='left')
train_all = train_all.merge(train_gp2d, on='object_id', how='left')
train_all = train_all.merge(train_bazin, on='object_id', how='left')

test_all = test_base.merge(test_tde, on='object_id', how='left')
test_all = test_all.merge(test_gp2d, on='object_id', how='left')
test_all = test_all.merge(test_bazin, on='object_id', how='left')

# Select v34a features
available_features = [f for f in feature_names if f in train_all.columns]
print(f"   Available features: {len(available_features)}")

X_train = train_all[available_features].values
X_test = test_all[available_features].values

# Handle infinities
X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# Calculate class weight
n_pos = np.sum(y)
n_neg = len(y) - n_pos
scale_pos_weight = n_neg / n_pos

# ====================
# 3. TRAIN WITH HEAVY REGULARIZATION
# ====================
print("\n3. Training LightGBM with heavy regularization...")

# Key differences from v77:
# - reg_alpha: 0.01-1.0 -> 1.0-10.0 (10x higher)
# - reg_lambda: 0.1-3.0 -> 2.0-15.0 (5x higher)
# - num_leaves: 15-63 -> 7-23 (much smaller)
# - feature_fraction: 0.6-0.95 -> 0.3-0.6 (more aggressive)
# - bagging_fraction: 0.6-0.95 -> 0.4-0.7 (more aggressive)

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',  # Will test DART in v111
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,

    # Heavy regularization
    'num_leaves': 15,           # v77 had up to 63
    'max_depth': 4,             # v77 had 4-8
    'learning_rate': 0.02,      # Lower than v77's 0.01-0.1
    'n_estimators': 600,

    # Aggressive subsampling
    'feature_fraction': 0.4,    # v77 had 0.6-0.95
    'bagging_fraction': 0.5,    # v77 had 0.6-0.95
    'bagging_freq': 5,

    # Heavy L1/L2
    'reg_alpha': 5.0,           # v77 had 0.01-1.0
    'reg_lambda': 10.0,         # v77 had 0.1-3.0

    # Leaf constraints
    'min_child_samples': 30,    # v77 had 10-50
    'min_child_weight': 0.01,

    # Class imbalance
    'scale_pos_weight': scale_pos_weight,
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(X_test), n_folds))
feature_importance = np.zeros(len(available_features))
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:")

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=available_features)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=available_features, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )

    oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds[:, fold-1] = model.predict(X_test, num_iteration=model.best_iteration)

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance += importance

    # Fold F1
    best_f1 = 0
    best_t = 0.1
    for t in np.linspace(0.03, 0.5, 50):
        f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    fold_f1s.append(best_f1)
    print(f"      Fold F1: {best_f1:.4f} @ threshold {best_t:.3f}")

# ====================
# 4. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

best_f1 = 0
best_thresh = 0.1
for t in np.linspace(0.03, 0.5, 200):
    f1 = f1_score(y, (oof_preds > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
print(f"   Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}")
print(f"   Fold Std: {np.std(fold_f1s):.4f}")

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
print(f"\n   TP={tp}, FP={fp}, FN={fn}")
print(f"   Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}")

# Feature importance analysis
feature_importance = feature_importance / n_folds
importance_df = pd.DataFrame({
    'feature': available_features,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n   Top 15 Features:")
print(importance_df.head(15).to_string(index=False))

# ====================
# 5. SUBMISSION
# ====================
print("\n" + "=" * 80)
print("SUBMISSION")
print("=" * 80)

test_avg = test_preds.mean(axis=1)
test_binary = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

submission_path = base_path / 'submissions/submission_v110_lgbm_reg.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name}")
print(f"   Predicted TDEs: {test_binary.sum()}")

# Save artifacts
artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'feature_importance': importance_df,
    'feature_names': available_features,
    'fold_f1s': fold_f1s,
    'params': lgb_params
}

with open(base_path / 'data/processed/v110_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# ====================
# 6. COMPARISON
# ====================
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(f"""
   Model                       OOF F1   LB F1    Notes
   -----                       ------   -----    -----
   v34a XGBoost (Optuna)       0.6667   0.6907   Best LB (TARGET)
   v77 LightGBM (Optuna)       0.6886   0.6714   Overfit
   v110 LightGBM (Heavy Reg)   {best_f1:.4f}   ???      This run

   Key: Lower OOF than v77 is GOOD if it means better LB!
""")

print("=" * 80)
print("v110 Complete - Submit to Kaggle to check LB!")
print("=" * 80)
```

**Step 2: Run the training script**

Run: `cd "C:\Users\alexy\Documents\Claude_projects\Kaggle competition\MALLORN astrophysics" && python scripts/train_v110_lgbm_regularized.py`

Expected: OOF F1 between 0.64-0.68 (deliberately lower than v77's 0.6886)

**Step 3: Commit**

```bash
git add scripts/train_v110_lgbm_regularized.py
git commit -m "feat(v110): LightGBM with heavy regularization for generalization"
```

---

### Task 2: Create DART Boosting Variant

**Files:**
- Create: `scripts/train_v111_lgbm_dart.py`

**Step 1: Create DART variant script**

```python
"""
MALLORN v111: LightGBM DART Boosting

DART = Dropouts meet Multiple Additive Regression Trees
- Randomly drops trees during boosting
- Reduces overfitting by preventing individual trees from dominating
- Often generalizes better than standard GBDT

Using v110's regularization + DART for maximum generalization.
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

print("=" * 80)
print("MALLORN v111: LightGBM DART Boosting")
print("=" * 80)

# ====================
# 1. LOAD DATA (same as v110)
# ====================
print("\n1. Loading data...")

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDE)")

# ====================
# 2. LOAD v34a FEATURES
# ====================
print("\n2. Loading v34a features...")

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a_artifacts = pickle.load(f)

feature_names = v34a_artifacts['feature_names']

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

train_all = train_base.merge(train_tde, on='object_id', how='left')
train_all = train_all.merge(train_gp2d, on='object_id', how='left')
train_all = train_all.merge(train_bazin, on='object_id', how='left')

test_all = test_base.merge(test_tde, on='object_id', how='left')
test_all = test_all.merge(test_gp2d, on='object_id', how='left')
test_all = test_all.merge(test_bazin, on='object_id', how='left')

available_features = [f for f in feature_names if f in train_all.columns]
print(f"   Features: {len(available_features)}")

X_train = train_all[available_features].values
X_test = test_all[available_features].values

X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

n_pos = np.sum(y)
n_neg = len(y) - n_pos
scale_pos_weight = n_neg / n_pos

# ====================
# 3. DART PARAMETERS
# ====================
print("\n3. Training LightGBM with DART boosting...")

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'dart',    # KEY CHANGE: DART instead of GBDT
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,

    # DART-specific parameters
    'drop_rate': 0.15,          # Fraction of trees to drop
    'max_drop': 50,             # Max trees to drop per iteration
    'skip_drop': 0.5,           # Probability of skipping dropout
    'uniform_drop': False,      # Weight by contribution

    # Same regularization as v110
    'num_leaves': 15,
    'max_depth': 4,
    'learning_rate': 0.02,
    'n_estimators': 600,

    'feature_fraction': 0.4,
    'bagging_fraction': 0.5,
    'bagging_freq': 5,

    'reg_alpha': 5.0,
    'reg_lambda': 10.0,

    'min_child_samples': 30,

    'scale_pos_weight': scale_pos_weight,
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(X_test), n_folds))
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:")

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=available_features)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=available_features, reference=train_data)

    # Note: DART doesn't support early stopping well, so we use fixed iterations
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=lgb_params['n_estimators'],
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(period=0)]
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds[:, fold-1] = model.predict(X_test)

    best_f1 = 0
    best_t = 0.1
    for t in np.linspace(0.03, 0.5, 50):
        f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    fold_f1s.append(best_f1)
    print(f"      Fold F1: {best_f1:.4f} @ threshold {best_t:.3f}")

# ====================
# 4. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

best_f1 = 0
best_thresh = 0.1
for t in np.linspace(0.03, 0.5, 200):
    f1 = f1_score(y, (oof_preds > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
print(f"   Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}")
print(f"   Fold Std: {np.std(fold_f1s):.4f}")

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
print(f"\n   TP={tp}, FP={fp}, FN={fn}")
print(f"   Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}")

# ====================
# 5. SUBMISSION
# ====================
print("\n" + "=" * 80)
print("SUBMISSION")
print("=" * 80)

test_avg = test_preds.mean(axis=1)
test_binary = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

submission_path = base_path / 'submissions/submission_v111_lgbm_dart.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name}")
print(f"   Predicted TDEs: {test_binary.sum()}")

artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'feature_names': available_features,
    'fold_f1s': fold_f1s,
    'params': lgb_params
}

with open(base_path / 'data/processed/v111_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80)
print(f"v111 DART Complete: OOF F1 = {best_f1:.4f}")
print("=" * 80)
```

**Step 2: Run the DART script**

Run: `cd "C:\Users\alexy\Documents\Claude_projects\Kaggle competition\MALLORN astrophysics" && python scripts/train_v111_lgbm_dart.py`

Expected: OOF F1 similar or slightly lower than v110

**Step 3: Commit**

```bash
git add scripts/train_v111_lgbm_dart.py
git commit -m "feat(v111): LightGBM DART boosting variant"
```

---

### Task 3: Optuna Search for Optimal Regularization

**Files:**
- Create: `scripts/train_v112_lgbm_optuna_reg.py`

**Step 1: Create Optuna script with constrained search space**

```python
"""
MALLORN v112: Optuna-Tuned LightGBM with Regularization Constraints

Key insight: v77's Optuna found params that overfit.
This version constrains the search space to favor regularization:
- Lower num_leaves bounds
- Higher reg_alpha/reg_lambda bounds
- Lower feature_fraction bounds

Goal: Find the sweet spot that maximizes GENERALIZATION, not OOF.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v112: Optuna LightGBM with Regularization Constraints")
print("=" * 80)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...")

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDE)")

# ====================
# 2. LOAD v34a FEATURES
# ====================
print("\n2. Loading v34a features...")

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a_artifacts = pickle.load(f)

feature_names = v34a_artifacts['feature_names']

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

train_all = train_base.merge(train_tde, on='object_id', how='left')
train_all = train_all.merge(train_gp2d, on='object_id', how='left')
train_all = train_all.merge(train_bazin, on='object_id', how='left')

test_all = test_base.merge(test_tde, on='object_id', how='left')
test_all = test_all.merge(test_gp2d, on='object_id', how='left')
test_all = test_all.merge(test_bazin, on='object_id', how='left')

available_features = [f for f in feature_names if f in train_all.columns]
print(f"   Features: {len(available_features)}")

X_train = train_all[available_features].values
X_test = test_all[available_features].values

X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

n_pos = np.sum(y)
n_neg = len(y) - n_pos
scale_pos_weight = n_neg / n_pos

# ====================
# 3. OPTUNA WITH CONSTRAINED SEARCH
# ====================
print("\n3. Running Optuna with regularization-focused search space...")

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
folds = list(skf.split(X_train, y))

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,

        # CONSTRAINED: Lower tree complexity
        'num_leaves': trial.suggest_int('num_leaves', 7, 23),      # v77: 15-63
        'max_depth': trial.suggest_int('max_depth', 3, 5),         # v77: 4-8

        # Learning
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),  # v77: 0.01-0.1
        'n_estimators': trial.suggest_int('n_estimators', 400, 800),

        # CONSTRAINED: More aggressive subsampling
        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 0.6),  # v77: 0.6-0.95
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.7),  # v77: 0.6-0.95
        'bagging_freq': trial.suggest_int('bagging_freq', 3, 7),

        # CONSTRAINED: Much higher regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 2.0, 15.0),   # v77: 0.01-1.0
        'reg_lambda': trial.suggest_float('reg_lambda', 5.0, 20.0), # v77: 0.1-3.0

        # Leaf constraints
        'min_child_samples': trial.suggest_int('min_child_samples', 25, 60),  # v77: 10-50

        # Class imbalance
        'scale_pos_weight': scale_pos_weight,
    }

    # DART-specific
    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_float('drop_rate', 0.1, 0.3)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.4, 0.6)

    oof_preds = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        if params['boosting_type'] == 'dart':
            model = lgb.train(
                params,
                train_data,
                num_boost_round=params['n_estimators'],
                valid_sets=[val_data],
                callbacks=[lgb.log_evaluation(period=0)]
            )
            oof_preds[val_idx] = model.predict(X_val)
        else:
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0)
                ]
            )
            oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

    # Find best threshold for F1
    best_f1 = 0
    for t in np.linspace(0.05, 0.5, 50):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1

    return best_f1

# Run optimization
sampler = TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)

print("   Running 80 trials...")
study.optimize(
    objective,
    n_trials=80,
    show_progress_bar=False,
    callbacks=[lambda study, trial: print(f"      Trial {trial.number}: F1={trial.value:.4f}") if trial.number % 10 == 0 else None]
)

print(f"\n   Best trial: {study.best_trial.number}")
print(f"   Best OOF F1: {study.best_value:.4f}")
print(f"\n   Best parameters:")
for k, v in study.best_params.items():
    print(f"      {k}: {v}")

# ====================
# 4. TRAIN FINAL MODEL
# ====================
print("\n4. Training final model with best parameters...")

best_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
    'scale_pos_weight': scale_pos_weight,
    **study.best_params
}

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(X_test), n_folds))
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(folds, 1):
    print(f"\n   Fold {fold}/{n_folds}:")

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=available_features)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=available_features, reference=train_data)

    is_dart = best_params.get('boosting_type') == 'dart'

    if is_dart:
        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=best_params.get('n_estimators', 600),
            valid_sets=[val_data],
            callbacks=[lgb.log_evaluation(period=0)]
        )
        oof_preds[val_idx] = model.predict(X_val)
        test_preds[:, fold-1] = model.predict(X_test)
    else:
        model = lgb.train(
            best_params,
            train_data,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds[:, fold-1] = model.predict(X_test, num_iteration=model.best_iteration)

    best_f1 = 0
    for t in np.linspace(0.03, 0.5, 50):
        f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
    fold_f1s.append(best_f1)
    print(f"      Fold F1: {best_f1:.4f}")

# ====================
# 5. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

best_f1 = 0
best_thresh = 0.1
for t in np.linspace(0.03, 0.5, 200):
    f1 = f1_score(y, (oof_preds > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
print(f"   Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}")
print(f"   Fold Std: {np.std(fold_f1s):.4f}")

# ====================
# 6. SUBMISSION
# ====================
print("\n" + "=" * 80)
print("SUBMISSION")
print("=" * 80)

test_avg = test_preds.mean(axis=1)
test_binary = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

submission_path = base_path / 'submissions/submission_v112_lgbm_optuna_reg.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name}")
print(f"   Predicted TDEs: {test_binary.sum()}")

artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'feature_names': available_features,
    'fold_f1s': fold_f1s,
    'best_params': best_params,
    'study_best_value': study.best_value
}

with open(base_path / 'data/processed/v112_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(f"""
   Model                          OOF F1   LB F1    Notes
   -----                          ------   -----    -----
   v34a XGBoost                   0.6667   0.6907   Best LB (TARGET)
   v77 LightGBM (Optuna)          0.6886   0.6714   Overfits
   v112 LightGBM (Constrained)    {best_f1:.4f}   ???      This run
""")

print("=" * 80)
print("v112 Complete")
print("=" * 80)
```

**Step 2: Run the Optuna script**

Run: `cd "C:\Users\alexy\Documents\Claude_projects\Kaggle competition\MALLORN astrophysics" && python scripts/train_v112_lgbm_optuna_reg.py`

Expected: 80 trials, best F1 likely between 0.65-0.68

**Step 3: Commit**

```bash
git add scripts/train_v112_lgbm_optuna_reg.py
git commit -m "feat(v112): Optuna LightGBM with constrained regularization search"
```

---

### Task 4: Submit to Kaggle and Compare Results

**Step 1: Submit all three variants to Kaggle**

After each script completes, submit to Kaggle:
```bash
# Submit v110
kaggle competitions submit -c mallorn-astronomical-classification-challenge -f submissions/submission_v110_lgbm_reg.csv -m "v110: LightGBM heavy regularization"

# Submit v111
kaggle competitions submit -c mallorn-astronomical-classification-challenge -f submissions/submission_v111_lgbm_dart.csv -m "v111: LightGBM DART boosting"

# Submit v112
kaggle competitions submit -c mallorn-astronomical-classification-challenge -f submissions/submission_v112_lgbm_optuna_reg.csv -m "v112: LightGBM Optuna constrained"
```

**Step 2: Record LB results in RESULTS_SUMMARY.md**

Update the results table with actual LB scores once available.

**Step 3: Commit results update**

```bash
git add RESULTS_SUMMARY.md CLAUDE.md
git commit -m "docs: update results with v110-v112 LightGBM experiments"
```

---

### Task 5: Create Ensemble of Best XGB + Best LGB

**Files:**
- Create: `scripts/train_v113_xgb_lgb_ensemble.py`

**Prerequisite:** Complete Tasks 1-4 and identify which LightGBM variant has best LB score.

**Step 1: Create ensemble script**

```python
"""
MALLORN v113: XGBoost + LightGBM Ensemble

Combines v34a (best XGB) with best regularized LightGBM (v110/v111/v112).

Strategy:
- Use OOF predictions from both models
- Weighted average with weights tuned on OOF
- Key: Different architectures should make different errors
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v113: XGBoost + LightGBM Ensemble")
print("=" * 80)

# ====================
# 1. LOAD OOF PREDICTIONS
# ====================
print("\n1. Loading model artifacts...")

# v34a XGBoost (best LB)
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)
xgb_oof = v34a['oof_preds']
xgb_test = v34a['test_preds']
xgb_thresh = v34a['best_threshold']
print(f"   v34a XGBoost: OOF F1={v34a['oof_f1']:.4f}")

# Best LightGBM (update this based on LB results)
# Try v110, v111, v112 - use whichever has best LB
lgb_version = 'v112'  # UPDATE THIS based on LB results
with open(base_path / f'data/processed/{lgb_version}_artifacts.pkl', 'rb') as f:
    lgb_artifacts = pickle.load(f)
lgb_oof = lgb_artifacts['oof_preds']
lgb_test = lgb_artifacts['test_preds']
lgb_thresh = lgb_artifacts['best_threshold']
print(f"   {lgb_version} LightGBM: OOF F1={lgb_artifacts['oof_f1']:.4f}")

# Load targets
from utils.data_loader import load_all_data
data = load_all_data()
y = data['train_meta']['target'].values
test_ids = data['test_meta']['object_id'].tolist()

# ====================
# 2. OPTIMIZE ENSEMBLE WEIGHTS
# ====================
print("\n2. Optimizing ensemble weights...")

best_f1 = 0
best_weight = 0.5
best_thresh = 0.1

for w in np.linspace(0.3, 0.7, 41):  # XGBoost weight
    ensemble_oof = w * xgb_oof + (1 - w) * lgb_oof

    for t in np.linspace(0.03, 0.5, 100):
        f1 = f1_score(y, (ensemble_oof > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_weight = w
            best_thresh = t

print(f"   Best XGB weight: {best_weight:.3f}")
print(f"   Best LGB weight: {1-best_weight:.3f}")
print(f"   Best threshold: {best_thresh:.3f}")
print(f"   Ensemble OOF F1: {best_f1:.4f}")

# ====================
# 3. CREATE SUBMISSION
# ====================
print("\n3. Creating ensemble submission...")

ensemble_test = best_weight * xgb_test + (1 - best_weight) * lgb_test
test_binary = (ensemble_test > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

submission_path = base_path / 'submissions/submission_v113_ensemble.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name}")
print(f"   Predicted TDEs: {test_binary.sum()}")

# ====================
# 4. ANALYSIS
# ====================
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Agreement analysis
xgb_binary = (xgb_oof > xgb_thresh).astype(int)
lgb_binary = (lgb_oof > lgb_thresh).astype(int)
agreement = np.mean(xgb_binary == lgb_binary)
print(f"\n   Model agreement: {agreement*100:.1f}%")

# Where do they disagree?
disagree_mask = xgb_binary != lgb_binary
n_disagree = np.sum(disagree_mask)
print(f"   Disagreements: {n_disagree} samples")

if n_disagree > 0:
    # Of disagreements, who was right?
    xgb_right = np.sum((xgb_binary[disagree_mask] == y[disagree_mask]))
    lgb_right = np.sum((lgb_binary[disagree_mask] == y[disagree_mask]))
    print(f"   XGB correct when disagree: {xgb_right}/{n_disagree} ({100*xgb_right/n_disagree:.1f}%)")
    print(f"   LGB correct when disagree: {lgb_right}/{n_disagree} ({100*lgb_right/n_disagree:.1f}%)")

# ====================
# 5. COMPARISON
# ====================
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(f"""
   Model                    OOF F1   LB F1    Notes
   -----                    ------   -----    -----
   v34a XGBoost             {v34a['oof_f1']:.4f}   0.6907   Best solo LB
   {lgb_version} LightGBM            {lgb_artifacts['oof_f1']:.4f}   ???      Best reg LGB
   v113 Ensemble            {best_f1:.4f}   ???      This run
""")

# Save artifacts
artifacts = {
    'ensemble_oof': best_weight * xgb_oof + (1 - best_weight) * lgb_oof,
    'ensemble_test': ensemble_test,
    'xgb_weight': best_weight,
    'lgb_weight': 1 - best_weight,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'xgb_version': 'v34a',
    'lgb_version': lgb_version
}

with open(base_path / 'data/processed/v113_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("=" * 80)
print("v113 Complete")
print("=" * 80)
```

**Step 2: Run ensemble script (after identifying best LGB)**

Run: `cd "C:\Users\alexy\Documents\Claude_projects\Kaggle competition\MALLORN astrophysics" && python scripts/train_v113_xgb_lgb_ensemble.py`

**Step 3: Submit to Kaggle**

```bash
kaggle competitions submit -c mallorn-astronomical-classification-challenge -f submissions/submission_v113_ensemble.csv -m "v113: XGBoost + LightGBM ensemble"
```

**Step 4: Commit**

```bash
git add scripts/train_v113_xgb_lgb_ensemble.py
git commit -m "feat(v113): XGBoost + regularized LightGBM ensemble"
```

---

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Best LGB LB F1 | ≥0.68 | Close gap to v34a's 0.6907 |
| Ensemble LB F1 | ≥0.70 | Benefit from diversity |
| OOF-LB gap | <0.02 | Better generalization |

## Key Insight

The goal is NOT to maximize OOF F1. We want to find the regularization sweet spot where LightGBM generalizes like XGBoost does. If v110-v112 achieve OOF ~0.65-0.68 with LB ~0.68+, that's success!
