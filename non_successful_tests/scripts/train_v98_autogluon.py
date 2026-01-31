"""
MALLORN v98: AutoGluon AutoML

AutoGluon dominated 2024 Kaggle tabular competitions:
- Won medals in 15/18 tabular contests
- 7 gold medals

Key advantage: Automatically tries multiple models and ensembles them:
- LightGBM, CatBoost, XGBoost, Random Forest, Neural Networks
- Automatic hyperparameter tuning
- Stacking and ensembling

Important: 1st place competitor confirmed test distribution differs from training.
Our adversarial weights already address this - we'll use them with AutoGluon.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v98: AutoGluon AutoML")
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

print(f"   Features: {len(available_features)}")

# Create DataFrames for AutoGluon
train_df = train_all[['object_id'] + available_features].copy()
train_df['target'] = y_train
train_df['sample_weight'] = sample_weights

test_df = test_all[['object_id'] + available_features].copy()

# ====================
# 3. AUTOGLUON TRAINING
# ====================
print("\n3. Training AutoGluon...")

from autogluon.tabular import TabularPredictor

# AutoGluon presets
# 'best_quality' - maximum accuracy, slower
# 'high_quality' - good balance
# 'good_quality' - faster
# 'medium_quality' - even faster

# We'll try multiple presets
presets_to_try = {
    'v98a_best': 'best_quality',
    'v98b_high': 'high_quality',
}

results = {}

for variant_name, preset in presets_to_try.items():
    print(f"\n   {variant_name}: AutoGluon preset={preset}")

    # Use 5-fold CV for proper OOF predictions
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y_train))
    test_preds_folds = []
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df[available_features], y_train), 1):
        print(f"      Fold {fold}...")

        fold_train = train_df.iloc[train_idx].copy()
        fold_val = train_df.iloc[val_idx].copy()

        # Prepare training data (drop object_id for training)
        train_data = fold_train[available_features + ['target', 'sample_weight']]

        # AutoGluon predictor
        predictor = TabularPredictor(
            label='target',
            sample_weight='sample_weight',
            eval_metric='f1',
            path=str(base_path / f'models/autogluon_{variant_name}_fold{fold}'),
            verbosity=0
        )

        # Train
        predictor.fit(
            train_data=train_data,
            presets=preset,
            time_limit=300,  # 5 minutes per fold
            num_bag_folds=0,  # Disable bagging for speed
            num_stack_levels=0,  # Disable stacking for speed
        )

        # Predict on validation
        val_data = fold_val[available_features]
        val_probs = predictor.predict_proba(val_data)

        # Handle different output formats
        if isinstance(val_probs, pd.DataFrame):
            if 1 in val_probs.columns:
                oof_preds[val_idx] = val_probs[1].values
            else:
                oof_preds[val_idx] = val_probs.iloc[:, 1].values
        else:
            oof_preds[val_idx] = val_probs[:, 1]

        # Predict on test
        test_data = test_df[available_features]
        test_probs = predictor.predict_proba(test_data)

        if isinstance(test_probs, pd.DataFrame):
            if 1 in test_probs.columns:
                test_preds_folds.append(test_probs[1].values)
            else:
                test_preds_folds.append(test_probs.iloc[:, 1].values)
        else:
            test_preds_folds.append(test_probs[:, 1])

        # Fold F1
        best_f1 = 0
        for t in np.linspace(0.1, 0.5, 50):
            f1 = f1_score(y_train[val_idx], (oof_preds[val_idx] > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)
        print(f"         Fold {fold} F1: {best_f1:.4f}")

        # Clean up to save disk space
        import shutil
        shutil.rmtree(str(base_path / f'models/autogluon_{variant_name}_fold{fold}'), ignore_errors=True)

    # Average test predictions
    test_preds = np.mean(test_preds_folds, axis=0)

    # OOF F1
    best_f1 = 0
    best_thresh = 0.3
    for t in np.linspace(0.1, 0.5, 200):
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
    }

    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Recall: {tp/(tp+fn):.4f} | Precision: {tp/(tp+fp):.4f}")
    print(f"      FN: {fn} | FP: {fp}")

# ====================
# 4. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n{'Variant':<20} {'OOF F1':<10} {'Recall':<10} {'Prec':<10}")
print("-" * 55)
print(f"{'v92d (LB=0.6986)':<20} {'0.6688':<10} {'69.6%':<10} {'64.4%':<10}")
print("-" * 55)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    print(f"{name:<20} {res['oof_f1']:<10.4f} {100*res['recall']:<10.1f}% {100*res['precision']:<10.1f}%")

# ====================
# 5. SUBMISSIONS
# ====================
print("\n" + "=" * 80)
print("SUBMISSIONS")
print("=" * 80)

for name, res in sorted_results:
    test_binary = (res['test_preds'] > res['threshold']).astype(int)

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': test_binary
    })

    filename = f"submission_{name}.csv"
    submission.to_csv(base_path / f'submissions/{filename}', index=False)

    print(f"   {filename}: OOF={res['oof_f1']:.4f}, TDEs={test_binary.sum()}")

with open(base_path / 'data/processed/v98_autogluon_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v98 AutoGluon Complete")
print("=" * 80)
