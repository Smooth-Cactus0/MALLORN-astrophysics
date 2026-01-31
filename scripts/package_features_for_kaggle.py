"""
Package all feature caches into a single file for Kaggle upload.
Creates a compressed pickle file with all features needed for the ensemble.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

base_path = Path(__file__).parent.parent
sys.path.insert(0, str(base_path / 'src'))

print("=" * 60)
print("Packaging Features for Kaggle Upload")
print("=" * 60)

# Load all feature caches
print("\n1. Loading feature caches...")

print("   - Loading base features (v4)...")
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']
print(f"      Train: {train_base.shape}, Test: {test_base.shape}")

print("   - Loading TDE physics features...")
tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')

print("   - Loading multiband GP features...")
with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)

print("   - Loading Bazin features...")
with open(base_path / 'data/processed/bazin_features_cache.pkl', 'rb') as f:
    bazin_cache = pickle.load(f)

print("   - Loading research features...")
with open(base_path / 'data/processed/research_features_cache.pkl', 'rb') as f:
    research_cache = pickle.load(f)

print("   - Loading adversarial validation results...")
with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)

print("   - Loading v34a artifacts (feature list)...")
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

print("   - Loading v92d artifacts (predictions for reference)...")
with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
    v92 = pickle.load(f)

# Merge all features
print("\n2. Merging features...")

train_all = train_base.merge(tde_cached['train'], on='object_id', how='left')
train_all = train_all.merge(gp2d_data['train'], on='object_id', how='left')
train_all = train_all.merge(bazin_cache['train'], on='object_id', how='left')
train_all = train_all.merge(research_cache['train'], on='object_id', how='left')

test_all = test_base.merge(tde_cached['test'], on='object_id', how='left')
test_all = test_all.merge(gp2d_data['test'], on='object_id', how='left')
test_all = test_all.merge(bazin_cache['test'], on='object_id', how='left')
test_all = test_all.merge(research_cache['test'], on='object_id', how='left')

print(f"   Train shape: {train_all.shape}")
print(f"   Test shape: {test_all.shape}")

# Load targets
print("\n3. Loading metadata and targets...")
from utils.data_loader import load_all_data
data = load_all_data()
train_meta = data['train_meta']
test_meta = data['test_meta']
y = train_meta['target'].values

# Define feature sets
print("\n4. Defining feature sets...")

v34a_features = v34a['feature_names']
adv_discriminative = ['all_rise_time', 'all_asymmetry']
base_features = [f for f in v34a_features if f not in adv_discriminative]

minimal_research = [
    'nuclear_concentration', 'nuclear_smoothness',
    'g_r_color_at_peak', 'r_i_color_at_peak',
    'mhps_10_100_ratio', 'mhps_30_100_ratio'
]

# Feature sets for each model
feature_sets = {
    'v92d': v34a_features,  # Original v34a features (with adversarial weights)
    'v34a': v34a_features,  # Same features, no adversarial weights
    'v114d': base_features + minimal_research,  # Base + minimal research
}

print(f"   v92d features: {len(feature_sets['v92d'])}")
print(f"   v34a features: {len(feature_sets['v34a'])}")
print(f"   v114d features: {len(feature_sets['v114d'])}")

# Package everything
print("\n5. Creating package...")

package = {
    # Merged feature DataFrames
    'train_features': train_all,
    'test_features': test_all,

    # Targets and IDs
    'y': y,
    'train_ids': train_meta['object_id'].values,
    'test_ids': test_meta['object_id'].values,

    # Adversarial weights
    'sample_weights': adv_results['sample_weights'],

    # Feature sets for each model
    'feature_sets': feature_sets,

    # Model configs
    'model_configs': {
        'v92d': {
            'type': 'xgboost',
            'use_adv_weights': True,
            'features': feature_sets['v92d'],
            'threshold': v92['v92d_baseline_adv']['threshold'],
            'params': {
                'objective': 'binary:logistic',
                'max_depth': 5,
                'learning_rate': 0.025,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'reg_alpha': 0.2,
                'reg_lambda': 1.5,
                'tree_method': 'hist',
                'n_estimators': 1000,
            }
        },
        'v34a': {
            'type': 'xgboost',
            'use_adv_weights': False,
            'features': feature_sets['v34a'],
            'threshold': v34a['best_threshold'],
            'params': {
                'objective': 'binary:logistic',
                'max_depth': 5,
                'learning_rate': 0.025,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'reg_alpha': 0.2,
                'reg_lambda': 1.5,
                'tree_method': 'hist',
                'n_estimators': 1000,
            }
        },
        'v114d': {
            'type': 'lightgbm',
            'use_adv_weights': True,
            'features': feature_sets['v114d'],
            'threshold': 0.363,  # From Optuna tuning
            'params': {
                # Optuna-tuned parameters (OOF F1: 0.6852)
                'objective': 'binary',
                'boosting_type': 'gbdt',
                'num_leaves': 8,
                'max_depth': 5,
                'learning_rate': 0.0394,
                'n_estimators': 654,
                'feature_fraction': 0.591,
                'bagging_fraction': 0.659,
                'bagging_freq': 5,
                'reg_alpha': 1.524,
                'reg_lambda': 2.72,
                'min_child_samples': 42,
            }
        }
    },

    # Ensemble weights (based on LB performance)
    'ensemble_weights': {
        'v92d': 0.45,   # Best LB (0.6986)
        'v34a': 0.30,   # Second best LB (0.6907)
        'v114d': 0.25,  # Best LightGBM (0.6797) - adds diversity
    },

    # Metadata
    'metadata': {
        'created': pd.Timestamp.now().isoformat(),
        'n_train': len(train_all),
        'n_test': len(test_all),
        'n_features_total': len(train_all.columns),
        'known_lb_scores': {
            'v92d': 0.6986,
            'v34a': 0.6907,
            'v114d': 0.6797,  # Before Optuna tuning
            'v114d_tuned_oof': 0.6852,  # After Optuna tuning
        }
    }
}

# Save package
output_path = base_path / 'data/kaggle_ensemble_package.pkl'
print(f"\n6. Saving to {output_path}...")

with open(output_path, 'wb') as f:
    pickle.dump(package, f, protocol=4)

# Check file size
import os
file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"   File size: {file_size_mb:.1f} MB")

# Also create a compressed version
print("\n7. Creating compressed version...")
import gzip
output_path_gz = base_path / 'data/kaggle_ensemble_package.pkl.gz'
with gzip.open(output_path_gz, 'wb') as f:
    pickle.dump(package, f, protocol=4)

file_size_gz_mb = os.path.getsize(output_path_gz) / (1024 * 1024)
print(f"   Compressed size: {file_size_gz_mb:.1f} MB")

print("\n" + "=" * 60)
print("Package created successfully!")
print("=" * 60)
print(f"\nUpload this file to Kaggle as a dataset:")
print(f"   {output_path_gz}")
print(f"\nPackage contents:")
print(f"   - Train features: {package['train_features'].shape}")
print(f"   - Test features: {package['test_features'].shape}")
print(f"   - Feature sets: {list(package['feature_sets'].keys())}")
print(f"   - Model configs: {list(package['model_configs'].keys())}")
print(f"   - Ensemble weights: {package['ensemble_weights']}")
