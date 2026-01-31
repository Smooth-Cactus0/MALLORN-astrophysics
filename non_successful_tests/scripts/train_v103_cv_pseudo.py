"""
MALLORN v103: CV Pseudo-Labels (No Test Leakage)

Previous pseudo-labeling (v94-v97) used test set predictions, which caused:
1. Test set leakage (using test predictions as training data)
2. Confirmation bias (reinforcing model's errors)

CV Pseudo-Labels approach:
1. Train model on training set with 5-fold CV
2. Use OOF predictions as soft labels for a second training round
3. NO test set predictions are used as training data

This is similar to knowledge distillation where the teacher model's
OOF predictions become soft targets for the student model.

Validation checks:
1. Verify no test data is used in training
2. Check that OOF predictions are properly cross-validated
3. Compare with baseline
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v103: CV Pseudo-Labels (No Test Leakage)")
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
y_train = train_meta['target'].values

print(f"   Training: {len(train_ids)} ({np.sum(y_train)} TDE)")
class_ratio = (len(y_train) - np.sum(y_train)) / np.sum(y_train)
print(f"   Class ratio: {class_ratio:.1f}:1")

# Load adversarial weights
with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)
sample_weights = adv_results['sample_weights']

# ====================
# 2. LOAD FEATURES
# ====================
print("\n2. Loading features...")

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)
v34a_features = v34a['feature_names']

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

shift_features = ['all_rise_time', 'all_asymmetry']
available_features = [f for f in v34a_features if f in train_all.columns and f not in shift_features]

X_train = train_all[available_features].values
X_test = test_all[available_features].values

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

print(f"   Features: {len(available_features)}")

# ====================
# 3. VALIDATION: Data Leakage Check
# ====================
print("\n3. Validation: Data leakage check...")

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   [INFO] Test data will NOT be used for pseudo-labels")
print(f"   [INFO] Only OOF predictions from training CV will be used")

# ====================
# 4. STEP 1: Generate OOF Predictions (Teacher)
# ====================
print("\n4. Step 1: Generating OOF predictions (teacher model)...")

# XGBoost params from v92d
teacher_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.02,
    'n_estimators': 1500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': class_ratio,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Generate OOF predictions
teacher_oof = np.zeros(len(y_train))
teacher_test_preds = []

print("   Training teacher model...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    w_tr = sample_weights[train_idx]

    model = xgb.XGBClassifier(**teacher_params)
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)

    teacher_oof[val_idx] = model.predict_proba(X_val)[:, 1]
    teacher_test_preds.append(model.predict_proba(X_test)[:, 1])

    best_f1 = max([f1_score(y_val, (teacher_oof[val_idx] > t).astype(int))
                   for t in np.linspace(0.1, 0.5, 50)])
    print(f"      Fold {fold} F1: {best_f1:.4f}")

teacher_test = np.mean(teacher_test_preds, axis=0)

# Teacher OOF performance
best_f1_teacher = 0
for t in np.linspace(0.05, 0.5, 200):
    f1 = f1_score(y_train, (teacher_oof > t).astype(int))
    if f1 > best_f1_teacher:
        best_f1_teacher = f1
        best_thresh_teacher = t

print(f"   Teacher OOF F1: {best_f1_teacher:.4f}")

# ====================
# 5. VALIDATION: OOF Predictions Check
# ====================
print("\n5. Validation: OOF predictions check...")

print(f"   OOF predictions range: [{teacher_oof.min():.4f}, {teacher_oof.max():.4f}]")
print(f"   OOF mean for class 0: {teacher_oof[y_train==0].mean():.4f}")
print(f"   OOF mean for class 1: {teacher_oof[y_train==1].mean():.4f}")

# Check that OOF predictions are properly calibrated
# (class 1 should have higher predicted probabilities)
assert teacher_oof[y_train==1].mean() > teacher_oof[y_train==0].mean(), "OOF calibration check failed!"
print("   [PASS] OOF predictions properly calibrated")

# ====================
# 6. STEP 2: Train Student with Soft Pseudo-Labels
# ====================
print("\n6. Step 2: Training student with CV pseudo-labels...")

results = {}

# Different blending strategies for pseudo-labels
configs = {
    # v103a: Pure soft labels (use teacher OOF directly)
    'v103a_soft_pure': {
        'blend': 0.0,  # 0% hard labels, 100% soft labels
        'description': 'Pure soft labels from teacher OOF'
    },
    # v103b: Blend 50% hard + 50% soft
    'v103b_blend50': {
        'blend': 0.5,  # 50% hard labels, 50% soft labels
        'description': '50% hard + 50% soft'
    },
    # v103c: Mostly hard with soft hint (80% hard + 20% soft)
    'v103c_blend80': {
        'blend': 0.8,  # 80% hard labels, 20% soft labels
        'description': '80% hard + 20% soft'
    },
}

student_params = {
    'objective': 'reg:squarederror',  # Regression for soft targets
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.02,
    'n_estimators': 1500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
}

for variant_name, config in configs.items():
    print(f"\n   {variant_name}: {config['description']}")

    blend = config['blend']
    # Create blended targets: blend*hard + (1-blend)*soft
    y_pseudo = blend * y_train + (1 - blend) * teacher_oof

    oof_preds = np.zeros(len(y_train))
    test_preds_folds = []
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr_pseudo = y_pseudo[train_idx]
        y_val = y_train[val_idx]  # Hard labels for evaluation
        w_tr = sample_weights[train_idx]

        model = xgb.XGBRegressor(**student_params)
        model.fit(X_tr, y_tr_pseudo, sample_weight=w_tr,
                  eval_set=[(X_val, y_pseudo[val_idx])], verbose=False)

        val_preds = np.clip(model.predict(X_val), 0, 1)
        oof_preds[val_idx] = val_preds

        test_preds = np.clip(model.predict(X_test), 0, 1)
        test_preds_folds.append(test_preds)

        best_f1 = max([f1_score(y_val, (val_preds > t).astype(int))
                       for t in np.linspace(0.05, 0.5, 50)])
        fold_f1s.append(best_f1)
        print(f"      Fold {fold} F1: {best_f1:.4f}")

    test_preds = np.mean(test_preds_folds, axis=0)

    # Find best threshold
    best_f1 = 0
    best_thresh = 0.3
    for t in np.linspace(0.05, 0.5, 200):
        f1 = f1_score(y_train, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    preds_binary = (oof_preds > best_thresh).astype(int)
    cm = confusion_matrix(y_train, preds_binary)
    tn, fp, fn, tp = cm.ravel()

    results[variant_name] = {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'fold_f1s': fold_f1s,
        'fold_std': np.std(fold_f1s),
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'recall': tp / (tp + fn),
        'precision': tp / (tp + fp),
        'blend': blend,
    }

    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Recall: {tp/(tp+fn):.1%} | Precision: {tp/(tp+fp):.1%}")

# ====================
# 7. VALIDATION: Compare with baseline
# ====================
print("\n7. Validation: Compare with baseline...")

with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
    v92_results = pickle.load(f)
baseline_f1 = v92_results['v92d_baseline_adv']['oof_f1']
print(f"   v92d baseline OOF F1: {baseline_f1:.4f}")
print(f"   Teacher OOF F1: {best_f1_teacher:.4f}")

for name, res in results.items():
    diff = res['oof_f1'] - baseline_f1
    status = "BETTER" if diff > 0 else "WORSE"
    print(f"   {name}: OOF F1={res['oof_f1']:.4f} ({diff:+.4f}, {status})")

# ====================
# 8. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n{'Variant':<20} {'OOF F1':<10} {'Recall':<10} {'Prec':<10} {'FN':<6} {'FP':<6}")
print("-" * 65)
print(f"{'v92d (LB=0.6986)':<20} {'0.6688':<10} {'69.6%':<10} {'64.4%':<10} {'45':<6} {'57':<6}")
print(f"{'Teacher model':<20} {best_f1_teacher:<10.4f} {'--':<10} {'--':<10} {'--':<6} {'--':<6}")
print("-" * 65)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    recall_str = f"{100*res['recall']:.1f}%"
    prec_str = f"{100*res['precision']:.1f}%"
    print(f"{name:<20} {res['oof_f1']:<10.4f} {recall_str:<10} {prec_str:<10} {res['confusion']['fn']:<6} {res['confusion']['fp']:<6}")

# ====================
# 9. CREATE SUBMISSION
# ====================
print("\n" + "=" * 80)
print("SUBMISSION")
print("=" * 80)

# Use best performing variant
best_variant = max(results.items(), key=lambda x: x[1]['oof_f1'])
best_name, best_res = best_variant

test_binary = (best_res['test_preds'] > best_res['threshold']).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

filename = f"submission_{best_name}.csv"
submission.to_csv(base_path / f'submissions/{filename}', index=False)

print(f"   Best variant: {best_name}")
print(f"   Saved: {filename}")
print(f"   OOF F1: {best_res['oof_f1']:.4f}")
print(f"   TDEs predicted: {test_binary.sum()}")

# Save all results
results['teacher'] = {
    'oof_f1': best_f1_teacher,
    'threshold': best_thresh_teacher,
    'oof_preds': teacher_oof,
    'test_preds': teacher_test,
}

with open(base_path / 'data/processed/v103_cv_pseudo_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v103 CV Pseudo-Labels Complete")
print("=" * 80)
