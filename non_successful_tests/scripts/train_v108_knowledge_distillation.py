"""
MALLORN v108: Knowledge Distillation

Knowledge distillation (Hinton et al., 2015) trains a "student" model to mimic
a "teacher" model's soft predictions, which can improve generalization.

Key concepts:
1. Temperature scaling: p_soft = sigmoid(logit/T) - higher T = softer predictions
2. Student learns from blend of hard labels and teacher's soft predictions
3. Teacher's soft predictions contain "dark knowledge" about decision uncertainty

We use two teachers:
- v92d: Best LB model (LB=0.6986, OOF=0.6688)
- v104: Best OOF model (OOF=0.6866)

Implementation:
- Since we have pre-computed OOF predictions, we can use them directly as soft targets
- Temperature is applied by transforming probabilities back to logits, scaling, then sigmoid
- Student trains on blended targets: α*hard + (1-α)*soft_teacher

Validation checks:
1. Verify temperature scaling works correctly
2. Check that student predictions are valid
3. Compare with original teacher performance
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
print("MALLORN v108: Knowledge Distillation")
print("=" * 80)
print("\nGoal: Train student models that generalize better using teacher knowledge")

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
# 3. LOAD TEACHER PREDICTIONS
# ====================
print("\n3. Loading teacher predictions...")

# v92d: Best LB model
with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
    v92_results = pickle.load(f)
v92d_oof = v92_results['v92d_baseline_adv']['oof_preds']
v92d_test = v92_results['v92d_baseline_adv']['test_preds']
v92d_f1 = v92_results['v92d_baseline_adv']['oof_f1']
print(f"   v92d Teacher: OOF F1={v92d_f1:.4f} (LB=0.6986)")

# v104: Best OOF model
with open(base_path / 'data/processed/v104_seed_ensemble_artifacts.pkl', 'rb') as f:
    v104_results = pickle.load(f)
v104_oof = v104_results['v104_seed_ensemble']['oof_preds']
v104_test = v104_results['v104_seed_ensemble']['test_preds']
v104_f1 = v104_results['v104_seed_ensemble']['oof_f1']
print(f"   v104 Teacher: OOF F1={v104_f1:.4f}")

# ====================
# 4. KNOWLEDGE DISTILLATION IMPLEMENTATION
# ====================
print("\n4. Knowledge distillation implementation...")

def prob_to_logit(p, eps=1e-7):
    """Convert probability to logit (inverse sigmoid)."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def logit_to_prob(z):
    """Convert logit to probability (sigmoid)."""
    return 1 / (1 + np.exp(-z))

def temperature_scale(probs, temperature):
    """
    Apply temperature scaling to probabilities.
    Higher temperature = softer (more uniform) predictions.
    Temperature=1 = no change.
    """
    if temperature == 1.0:
        return probs

    # Convert to logits, scale, convert back
    logits = prob_to_logit(probs)
    scaled_logits = logits / temperature
    return logit_to_prob(scaled_logits)

def create_distillation_targets(hard_labels, soft_preds, alpha, temperature=1.0):
    """
    Create distillation targets: α*hard + (1-α)*soft_teacher.

    Args:
        hard_labels: Original binary labels (0/1)
        soft_preds: Teacher's soft predictions
        alpha: Weight for hard labels (0=pure soft, 1=pure hard)
        temperature: Temperature for softening teacher predictions

    Returns:
        Blended soft targets
    """
    soft_scaled = temperature_scale(soft_preds, temperature)
    return alpha * hard_labels + (1 - alpha) * soft_scaled

# ====================
# 5. VALIDATION: Implementation Check
# ====================
print("\n5. Validation: Implementation check...")

# Test temperature scaling
test_probs = np.array([0.1, 0.5, 0.9])
print(f"   Original probs: {test_probs}")
for T in [1.0, 2.0, 5.0]:
    scaled = temperature_scale(test_probs, T)
    print(f"   T={T}: {scaled.round(4)}")

# Verify temperature scaling pushes toward 0.5 (uniform)
assert np.all(temperature_scale(test_probs, 5.0) > temperature_scale(test_probs, 1.0)[:1])
print("   [PASS] Higher T pushes low probs toward 0.5")

# Test distillation targets
test_hard = np.array([0, 0, 1, 1])
test_soft = np.array([0.1, 0.3, 0.7, 0.9])
targets = create_distillation_targets(test_hard, test_soft, alpha=0.5, temperature=1.0)
expected = 0.5 * test_hard + 0.5 * test_soft
assert np.allclose(targets, expected)
print("   [PASS] Distillation target blending verified")

# ====================
# 6. TRAIN STUDENT MODELS
# ====================
print("\n6. Training student models...")

# Student model params (same architecture as teacher, but learning from soft targets)
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
    'n_jobs': -1,
}

# Configurations to test
configs = {
    # v92d as teacher
    'v108a_v92d_T1_a50': {'teacher_oof': v92d_oof, 'teacher_test': v92d_test,
                          'temperature': 1.0, 'alpha': 0.5, 'teacher_name': 'v92d'},
    'v108b_v92d_T2_a50': {'teacher_oof': v92d_oof, 'teacher_test': v92d_test,
                          'temperature': 2.0, 'alpha': 0.5, 'teacher_name': 'v92d'},
    'v108c_v92d_T1_a30': {'teacher_oof': v92d_oof, 'teacher_test': v92d_test,
                          'temperature': 1.0, 'alpha': 0.3, 'teacher_name': 'v92d'},

    # v104 as teacher
    'v108d_v104_T1_a50': {'teacher_oof': v104_oof, 'teacher_test': v104_test,
                          'temperature': 1.0, 'alpha': 0.5, 'teacher_name': 'v104'},
    'v108e_v104_T2_a50': {'teacher_oof': v104_oof, 'teacher_test': v104_test,
                          'temperature': 2.0, 'alpha': 0.5, 'teacher_name': 'v104'},
    'v108f_v104_T1_a30': {'teacher_oof': v104_oof, 'teacher_test': v104_test,
                          'temperature': 1.0, 'alpha': 0.3, 'teacher_name': 'v104'},

    # Ensemble teacher (average of v92d and v104)
    'v108g_ensemble_T1_a50': {'teacher_oof': (v92d_oof + v104_oof) / 2,
                              'teacher_test': (v92d_test + v104_test) / 2,
                              'temperature': 1.0, 'alpha': 0.5, 'teacher_name': 'ensemble'},
}

results = {}
n_folds = 5

for variant_name, config in configs.items():
    teacher_oof = config['teacher_oof']
    teacher_test = config['teacher_test']
    temperature = config['temperature']
    alpha = config['alpha']
    teacher_name = config['teacher_name']

    print(f"\n   {variant_name} (teacher={teacher_name}, T={temperature}, alpha={alpha}):")

    # Use multiple seeds for stability (like v104)
    seeds = [42, 123, 456]
    all_oof_preds = []
    all_test_preds = []
    seed_f1s = []

    for seed_idx, seed in enumerate(seeds):
        params = student_params.copy()
        params['random_state'] = seed

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        oof_preds = np.zeros(len(y_train))
        test_preds_folds = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr_hard = y_train[train_idx]
            y_val = y_train[val_idx]
            w_tr = sample_weights[train_idx]

            # Create distillation targets for training data
            y_tr_distill = create_distillation_targets(
                y_tr_hard, teacher_oof[train_idx], alpha, temperature
            )

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_tr, y_tr_distill,
                sample_weight=w_tr,
                eval_set=[(X_val, y_val)],  # Validate on hard labels
                verbose=False
            )

            # Clip predictions to [0, 1]
            val_preds = np.clip(model.predict(X_val), 0, 1)
            oof_preds[val_idx] = val_preds

            test_preds = np.clip(model.predict(X_test), 0, 1)
            test_preds_folds.append(test_preds)

        test_preds = np.mean(test_preds_folds, axis=0)

        # Calculate OOF F1 for this seed
        best_f1 = 0
        for t in np.linspace(0.05, 0.5, 100):
            f1 = f1_score(y_train, (oof_preds > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1

        seed_f1s.append(best_f1)
        all_oof_preds.append(oof_preds)
        all_test_preds.append(test_preds)

    # Ensemble across seeds
    ensemble_oof = np.mean(all_oof_preds, axis=0)
    ensemble_test = np.mean(all_test_preds, axis=0)

    # Find best threshold
    best_f1 = 0
    best_thresh = 0.3
    for t in np.linspace(0.05, 0.5, 200):
        f1 = f1_score(y_train, (ensemble_oof > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    preds_binary = (ensemble_oof > best_thresh).astype(int)
    cm = confusion_matrix(y_train, preds_binary)
    tn, fp, fn, tp = cm.ravel()

    results[variant_name] = {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'oof_preds': ensemble_oof,
        'test_preds': ensemble_test,
        'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'recall': tp / (tp + fn),
        'precision': tp / (tp + fp),
        'temperature': temperature,
        'alpha': alpha,
        'teacher': teacher_name,
        'seed_f1s': seed_f1s,
        'seed_f1_mean': np.mean(seed_f1s),
        'seed_f1_std': np.std(seed_f1s),
    }

    print(f"      Seed F1s: {[f'{f:.4f}' for f in seed_f1s]}")
    print(f"      Ensemble OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Recall: {tp/(tp+fn):.1%} | Precision: {tp/(tp+fp):.1%}")

# ====================
# 7. COMPARE WITH TEACHERS
# ====================
print("\n7. Comparison with teachers...")

print(f"\n   Teachers:")
print(f"      v92d: OOF F1={v92d_f1:.4f} (LB=0.6986)")
print(f"      v104: OOF F1={v104_f1:.4f}")

print(f"\n   Student improvements vs teacher:")
for name, res in results.items():
    teacher = res['teacher']
    teacher_f1 = v92d_f1 if teacher == 'v92d' else v104_f1 if teacher == 'v104' else (v92d_f1 + v104_f1) / 2
    diff = res['oof_f1'] - teacher_f1
    status = "BETTER" if diff > 0 else "WORSE" if diff < 0 else "SAME"
    print(f"      {name}: {res['oof_f1']:.4f} ({diff:+.4f} vs {teacher}, {status})")

# ====================
# 8. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n{'Variant':<25} {'OOF F1':<10} {'Recall':<10} {'Prec':<10} {'Teacher':<10}")
print("-" * 70)
print(f"{'v92d (Teacher, LB=0.6986)':<25} {v92d_f1:<10.4f} {'69.6%':<10} {'64.4%':<10} {'--':<10}")
print(f"{'v104 (Teacher)':<25} {v104_f1:<10.4f} {'77.7%':<10} {'61.5%':<10} {'--':<10}")
print("-" * 70)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    recall_str = f"{100*res['recall']:.1f}%"
    prec_str = f"{100*res['precision']:.1f}%"
    print(f"{name:<25} {res['oof_f1']:<10.4f} {recall_str:<10} {prec_str:<10} {res['teacher']:<10}")

# ====================
# 9. SUBMISSION
# ====================
print("\n" + "=" * 80)
print("SUBMISSION")
print("=" * 80)

# Create submissions for best variants
for name, res in sorted_results[:3]:  # Top 3 only
    test_binary = (res['test_preds'] > res['threshold']).astype(int)

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': test_binary
    })

    filename = f"submission_{name}.csv"
    submission.to_csv(base_path / f'submissions/{filename}', index=False)

    print(f"   {filename}: OOF={res['oof_f1']:.4f}, TDEs={test_binary.sum()}")

# Save all results
with open(base_path / 'data/processed/v108_knowledge_distillation_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v108 Knowledge Distillation Complete")
print("=" * 80)

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

best_student = max(results.items(), key=lambda x: x[1]['oof_f1'])
best_name, best_res = best_student

print(f"\n   Best student: {best_name}")
print(f"      OOF F1: {best_res['oof_f1']:.4f}")
print(f"      Teacher: {best_res['teacher']}")
print(f"      Temperature: {best_res['temperature']}")
print(f"      Alpha: {best_res['alpha']}")

# Check if any student beat both teachers
beat_v92d = best_res['oof_f1'] > v92d_f1
beat_v104 = best_res['oof_f1'] > v104_f1

if beat_v92d and beat_v104:
    print(f"\n   [SUCCESS] Student beats BOTH teachers!")
elif beat_v92d:
    print(f"\n   [PARTIAL] Student beats v92d but not v104")
elif beat_v104:
    print(f"\n   [PARTIAL] Student beats v104 but not v92d")
else:
    print(f"\n   [INFO] No student beat the teachers - distillation didn't help OOF")
