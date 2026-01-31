"""
MALLORN v84: Combined Best Approaches

Combines the most promising strategies from v81-v83:
1. Mild regularization (v81a - stable)
2. Light Gaussian noise (v83a - most stable, std=0.0264)
3. Adversarial sample weights (from adversarial validation)
4. Remove high-shift features (all_rise_time, all_asymmetry)

Goal: Break 0.70 LB by achieving healthy OOF-LB correlation
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

print("=" * 80)
print("MALLORN v84: Combined Best Approaches")
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
# 2. LOAD FEATURES AND ADVERSARIAL WEIGHTS
# ====================
print("\n2. Loading v34a features and adversarial weights...")

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

feature_names = v34a['feature_names']

# Load adversarial validation results
with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)

sample_weights = adv_results['sample_weights']
top_shift_features = adv_results['top_discriminative_features']

print(f"   v34a features: {len(feature_names)}")
print(f"   Top distribution-shift features: {top_shift_features[:5]}")

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

# Select v34a features (excluding high-shift features)
high_shift_features = ['all_rise_time', 'all_asymmetry']  # Top 2 distribution shift
available_features = [f for f in feature_names if f in train_all.columns]
print(f"   Available features before filtering: {len(available_features)}")

# ====================
# 3. DEFINE VARIANTS
# ====================
print("\n3. Defining combined variants...")

# Noise injection function
def add_gaussian_noise(X, noise_level, feature_stds, rng):
    noise = rng.normal(0, noise_level, X.shape) * feature_stds
    X_noisy = X + noise
    X_noisy = np.where(np.isnan(X), np.nan, X_noisy)
    return X_noisy

variants = {
    'v84a_baseline_adv_weights': {
        'features': available_features,  # All features
        'use_adv_weights': True,
        'noise_level': 0.0,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
        'max_depth': 5,
        'description': 'v34a + adversarial weights only'
    },
    'v84b_no_shift_features': {
        'features': [f for f in available_features if f not in high_shift_features],
        'use_adv_weights': False,
        'noise_level': 0.0,
        'reg_alpha': 0.2,
        'reg_lambda': 1.5,
        'max_depth': 5,
        'description': 'v34a minus high-shift features'
    },
    'v84c_mild_reg_noise': {
        'features': available_features,
        'use_adv_weights': False,
        'noise_level': 0.05,
        'reg_alpha': 0.3,
        'reg_lambda': 2.25,
        'max_depth': 5,
        'description': 'Mild regularization + 5% noise'
    },
    'v84d_all_combined': {
        'features': [f for f in available_features if f not in high_shift_features],
        'use_adv_weights': True,
        'noise_level': 0.05,
        'reg_alpha': 0.3,
        'reg_lambda': 2.25,
        'max_depth': 5,
        'description': 'All strategies combined'
    },
    'v84e_strong_combined': {
        'features': [f for f in available_features if f not in high_shift_features],
        'use_adv_weights': True,
        'noise_level': 0.08,
        'reg_alpha': 0.4,
        'reg_lambda': 3.0,
        'max_depth': 4,
        'description': 'Strong combined (more reg + noise)'
    },
    'v84f_conservative': {
        'features': available_features,
        'use_adv_weights': True,
        'noise_level': 0.03,
        'reg_alpha': 0.25,
        'reg_lambda': 2.0,
        'max_depth': 5,
        'description': 'Conservative (light touch on all)'
    }
}

for name, cfg in variants.items():
    print(f"   {name}: {cfg['description']}")

# ====================
# 4. TRAIN ALL VARIANTS
# ====================
print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {}

for variant_name, cfg in variants.items():
    print(f"\n   {variant_name}: {cfg['description']}")

    # Select features
    features = cfg['features']
    X_train = train_all[features].values.copy()
    X_test = test_all[features].values.copy()

    # Handle infinities
    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

    # Compute feature stds for noise
    feature_stds = np.nanstd(X_train, axis=0)
    feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)

    # XGBoost params
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': cfg['max_depth'],
        'learning_rate': 0.025,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': cfg['reg_alpha'],
        'reg_lambda': cfg['reg_lambda'],
        'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros((len(X_test), n_folds))
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_val = X_train[train_idx].copy(), X_train[val_idx].copy()
        y_tr, y_val = y[train_idx], y[val_idx]

        rng = np.random.default_rng(42 + fold)

        # Apply noise to training data
        if cfg['noise_level'] > 0:
            # Create augmented training data
            X_tr_noisy = add_gaussian_noise(X_tr, cfg['noise_level'], feature_stds, rng)
            X_tr = np.vstack([X_tr, X_tr_noisy])
            y_tr = np.concatenate([y_tr, y_tr])

            # Also double weights if using adversarial weights
            if cfg['use_adv_weights']:
                fold_weights = np.concatenate([sample_weights[train_idx], sample_weights[train_idx]])
            else:
                fold_weights = None
        else:
            if cfg['use_adv_weights']:
                fold_weights = sample_weights[train_idx]
            else:
                fold_weights = None

        # Create DMatrix
        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=fold_weights, feature_names=features)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
        dtest = xgb.DMatrix(X_test, feature_names=features)

        model = xgb.train(
            params,
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
        for t in np.linspace(0.05, 0.5, 50):
            f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
            if f1 > best_f1:
                best_f1 = f1
        fold_f1s.append(best_f1)

    # Overall OOF F1
    best_f1 = 0
    best_thresh = 0.3
    for t in np.linspace(0.05, 0.5, 200):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    results[variant_name] = {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'fold_f1s': fold_f1s,
        'fold_std': np.std(fold_f1s),
        'oof_preds': oof_preds,
        'test_preds': test_preds.mean(axis=1),
        'config': cfg
    }

    print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"      Fold Std: {np.std(fold_f1s):.4f}")
    print(f"      Features: {len(features)}")

# ====================
# 5. RESULTS SUMMARY
# ====================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print(f"\n{'Variant':<30} {'OOF F1':<10} {'Std':<10} {'Features':<10}")
print("-" * 65)

# Reference
print(f"{'v34a (baseline)':<30} {'0.6667':<10} {'-':<10} {'223':<10}")
print(f"{'v83a_light_noise (best std)':<30} {'0.6472':<10} {'0.0264':<10} {'223':<10}")
print("-" * 65)

# Sort by std (most stable first)
sorted_results = sorted(results.items(), key=lambda x: x[1]['fold_std'])

for name, res in sorted_results:
    n_feats = len(res['config']['features'])
    print(f"{name:<30} {res['oof_f1']:<10.4f} {res['fold_std']:<10.4f} {n_feats:<10}")

# ====================
# 6. SUBMISSIONS
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
    print(f"   {filename}: OOF={res['oof_f1']:.4f}, std={res['fold_std']:.4f}, TDEs={test_binary.sum()}")

# Save artifacts
with open(base_path / 'data/processed/v84_combined_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

# ====================
# 7. BEST PICKS
# ====================
print("\n" + "=" * 80)
print("TOP PICKS FOR SUBMISSION")
print("=" * 80)

# Most stable
most_stable = sorted_results[0]
print(f"\n   Most Stable: {most_stable[0]}")
print(f"      OOF: {most_stable[1]['oof_f1']:.4f}, Std: {most_stable[1]['fold_std']:.4f}")

# Lowest OOF (most underfit)
lowest_oof = min(results.items(), key=lambda x: x[1]['oof_f1'])
print(f"\n   Lowest OOF (most underfit): {lowest_oof[0]}")
print(f"      OOF: {lowest_oof[1]['oof_f1']:.4f}, Std: {lowest_oof[1]['fold_std']:.4f}")

# Best balance (low OOF + low std)
# Score = OOF * std (lower is better)
balance_scores = {name: res['oof_f1'] * res['fold_std'] * 100 for name, res in results.items()}
best_balance = min(balance_scores.items(), key=lambda x: x[1])
print(f"\n   Best Balance (OOF * std): {best_balance[0]}")
print(f"      OOF: {results[best_balance[0]]['oof_f1']:.4f}, Std: {results[best_balance[0]]['fold_std']:.4f}")

print("\n" + "=" * 80)
print("v84 Complete")
print("=" * 80)
