"""
MALLORN v115: XGBoost with Research-Based Enhancements

Applying lessons learned from LightGBM experiments to our best XGBoost:

Base: v34a XGBoost (LB 0.6907)
Enhancements:
1. Adversarial sample weights (from v92d, LB 0.6986)
2. Minimal research features (from v114d success):
   - nuclear_concentration, nuclear_smoothness
   - g_r_color_at_peak, r_i_color_at_peak
   - mhps_10_100_ratio, mhps_30_100_ratio
3. Remove adversarial-discriminative features (all_rise_time, all_asymmetry)

Target: Beat v92d (0.6986) or at least match it with different predictions for ensemble.
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
print("MALLORN v115: XGBoost with Research-Based Enhancements")
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
print(f"   Test: {len(test_ids)} objects")

# ====================
# 2. LOAD ADVERSARIAL WEIGHTS
# ====================
print("\n2. Loading adversarial weights...")

with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)
sample_weights = adv_results['sample_weights']
print(f"   Weights range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")

# ====================
# 3. LOAD ALL FEATURES
# ====================
print("\n3. Loading feature data...")

# v34a features
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)
v34a_features = v34a['feature_names']

# Base features
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

# Research features
with open(base_path / 'data/processed/research_features_cache.pkl', 'rb') as f:
    research_cache = pickle.load(f)
train_research = research_cache['train']
test_research = research_cache['test']

# Merge all
train_all = train_base.merge(train_tde, on='object_id', how='left')
train_all = train_all.merge(train_gp2d, on='object_id', how='left')
train_all = train_all.merge(train_bazin, on='object_id', how='left')
train_all = train_all.merge(train_research, on='object_id', how='left')

test_all = test_base.merge(test_tde, on='object_id', how='left')
test_all = test_all.merge(test_gp2d, on='object_id', how='left')
test_all = test_all.merge(test_bazin, on='object_id', how='left')
test_all = test_all.merge(test_research, on='object_id', how='left')

print(f"   Total available features: {len(train_all.columns) - 1}")

# ====================
# 4. DEFINE FEATURE SETS
# ====================
print("\n4. Defining feature sets...")

# Adversarial-discriminative features to remove
adv_discriminative = ['all_rise_time', 'all_asymmetry']

# Minimal research features (from v114d success)
minimal_research = [
    'nuclear_concentration', 'nuclear_smoothness',
    'g_r_color_at_peak', 'r_i_color_at_peak',
    'mhps_10_100_ratio', 'mhps_30_100_ratio'
]

# Additional promising features from v114 analysis
extended_research = minimal_research + [
    'nuclear_position_score',
    'mhps_10d', 'mhps_30d',
    'g_r_color_peak_to_late', 'r_i_color_peak_to_late'
]

# Define experiments
experiments = {
    'v115a_baseline_adv': {
        'description': 'v34a + adversarial weights (like v92d)',
        'base_features': v34a_features,
        'extra_features': [],
        'remove_features': adv_discriminative,
        'use_adv_weights': True,
        'params': 'v34a'
    },
    'v115b_minimal_research': {
        'description': 'v34a + adversarial + minimal research (6 features)',
        'base_features': v34a_features,
        'extra_features': minimal_research,
        'remove_features': adv_discriminative,
        'use_adv_weights': True,
        'params': 'v34a'
    },
    'v115c_extended_research': {
        'description': 'v34a + adversarial + extended research (11 features)',
        'base_features': v34a_features,
        'extra_features': extended_research,
        'remove_features': adv_discriminative,
        'use_adv_weights': True,
        'params': 'v34a'
    },
    'v115d_minimal_no_adv_weight': {
        'description': 'v34a + minimal research, NO adversarial weights',
        'base_features': v34a_features,
        'extra_features': minimal_research,
        'remove_features': adv_discriminative,
        'use_adv_weights': False,
        'params': 'v34a'
    },
    'v115e_minimal_regularized': {
        'description': 'v34a + minimal research + more regularization',
        'base_features': v34a_features,
        'extra_features': minimal_research,
        'remove_features': adv_discriminative,
        'use_adv_weights': True,
        'params': 'regularized'
    },
}

print(f"   Experiments defined: {len(experiments)}")
for name, cfg in experiments.items():
    print(f"      {name}: {cfg['description']}")

# ====================
# 5. DEFINE XGB PARAMETERS
# ====================
print("\n5. Defining XGBoost parameters...")

# v34a original parameters (proven to work)
params_v34a = {
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

# More regularized version
params_regularized = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 4,  # Reduced
    'learning_rate': 0.02,  # Slower
    'subsample': 0.75,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,  # Higher
    'reg_alpha': 0.5,  # More L1
    'reg_lambda': 2.5,  # More L2
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

param_sets = {
    'v34a': params_v34a,
    'regularized': params_regularized
}

print(f"   v34a params: depth={params_v34a['max_depth']}, lr={params_v34a['learning_rate']}")
print(f"   regularized params: depth={params_regularized['max_depth']}, lr={params_regularized['learning_rate']}")

# ====================
# 6. TRAIN ALL EXPERIMENTS
# ====================
print("\n" + "=" * 80)
print("TRAINING EXPERIMENTS")
print("=" * 80)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {}

for exp_name, exp_config in experiments.items():
    print(f"\n{'-' * 60}")
    print(f"   {exp_name}: {exp_config['description']}")
    print(f"{'-' * 60}")

    # Build feature set
    feature_set = list(exp_config['base_features']) + exp_config['extra_features']
    feature_set = [f for f in feature_set if f not in exp_config['remove_features']]
    feature_set = list(dict.fromkeys(feature_set))  # Remove duplicates
    available_features = [f for f in feature_set if f in train_all.columns]

    print(f"   Features: {len(available_features)}")
    print(f"   Adversarial weights: {exp_config['use_adv_weights']}")

    X_train = train_all[available_features].values
    X_test = test_all[available_features].values

    # Handle infinities
    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

    # Get parameters
    xgb_params = param_sets[exp_config['params']].copy()

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros((len(X_test), n_folds))
    feature_importance = np.zeros(len(available_features))
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Apply adversarial weights if specified
        if exp_config['use_adv_weights']:
            fold_weights = sample_weights[train_idx]
        else:
            fold_weights = None

        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=fold_weights, feature_names=available_features)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=available_features)
        dtest = xgb.DMatrix(X_test, feature_names=available_features)

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=600,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        oof_preds[val_idx] = model.predict(dval)
        test_preds[:, fold-1] = model.predict(dtest)

        # Feature importance
        importance = model.get_score(importance_type='gain')
        for feat, gain in importance.items():
            if feat in available_features:
                idx = available_features.index(feat)
                feature_importance[idx] += gain

        # Fold F1
        best_fold_f1 = 0
        for t in np.linspace(0.03, 0.5, 50):
            f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
            if f1 > best_fold_f1:
                best_fold_f1 = f1
        fold_f1s.append(best_fold_f1)

    # Find optimal threshold
    best_f1 = 0
    best_thresh = 0.1
    for t in np.linspace(0.03, 0.5, 200):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # Confusion matrix
    preds_binary = (oof_preds > best_thresh).astype(int)
    cm = confusion_matrix(y, preds_binary)
    tn, fp, fn, tp = cm.ravel()

    print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
    print(f"   Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}")
    print(f"   Fold std: {np.std(fold_f1s):.4f}")
    print(f"   TP={tp}, FP={fp}, FN={fn}")

    # Estimate LB (XGBoost typically: OOF + 0.02 to 0.03)
    estimated_lb = best_f1 + 0.025
    print(f"   Estimated LB: ~{estimated_lb:.4f}")

    results[exp_name] = {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'fold_f1s': fold_f1s,
        'fold_std': np.std(fold_f1s),
        'oof_preds': oof_preds,
        'test_preds': test_preds.mean(axis=1),
        'confusion': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
        'feature_importance': feature_importance / n_folds,
        'feature_names': available_features,
        'config': exp_config,
        'estimated_lb': estimated_lb
    }

# ====================
# 7. RESULTS COMPARISON
# ====================
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

print(f"\n{'Experiment':<35} {'OOF F1':<10} {'Est. LB':<10} {'Fold Std':<10}")
print("-" * 70)

# Baselines
print(f"{'v34a XGB (actual LB=0.6907)':<35} {'0.6667':<10} {'0.6907':<10} {'-':<10}")
print(f"{'v92d XGB+Adv (actual LB=0.6986)':<35} {'~0.67':<10} {'0.6986':<10} {'-':<10}")
print("-" * 70)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    print(f"{name:<35} {res['oof_f1']:<10.4f} {res['estimated_lb']:<10.4f} {res['fold_std']:<10.4f}")

# ====================
# 8. CREATE SUBMISSIONS
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

    print(f"   {filename}: OOF={res['oof_f1']:.4f}, Est.LB={res['estimated_lb']:.4f}, TDEs={test_binary.sum()}")

# ====================
# 9. FEATURE IMPORTANCE ANALYSIS
# ====================
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE: Research Features")
print("=" * 80)

# Analyze best experiment
best_exp = sorted_results[0][0]
best_result = results[best_exp]

importance_df = pd.DataFrame({
    'feature': best_result['feature_names'],
    'importance': best_result['feature_importance']
}).sort_values('importance', ascending=False)

all_research = [
    'nuclear_concentration', 'nuclear_smoothness', 'nuclear_position_score', 'nuclear_variability_ratio',
    'g_r_color_at_peak', 'r_i_color_at_peak', 'g_r_color_peak_to_late', 'r_i_color_peak_to_late',
    'mhps_10_100_ratio', 'mhps_30_100_ratio', 'mhps_10d', 'mhps_30d', 'mhps_100d', 'mhps_dominant_scale'
]

print(f"\n   Best experiment: {best_exp}")
print(f"\n   Research features ranking:")

research_in_model = importance_df[importance_df['feature'].isin(all_research)]
for _, row in research_in_model.iterrows():
    rank = list(importance_df['feature']).index(row['feature']) + 1
    print(f"      #{rank:3d}: {row['feature']:<30} {row['importance']:8.1f}")

# ====================
# 10. SAVE ARTIFACTS
# ====================
print("\n10. Saving artifacts...")

artifacts = {
    'results': results,
    'experiments': experiments,
    'param_sets': param_sets,
    'minimal_research_features': minimal_research,
    'extended_research_features': extended_research
}

with open(base_path / 'data/processed/v115_xgb_research_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("   Saved: v115_xgb_research_artifacts.pkl")

# ====================
# 11. SUMMARY
# ====================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

best_oof = sorted_results[0][1]['oof_f1']
best_name = sorted_results[0][0]
best_est_lb = sorted_results[0][1]['estimated_lb']

print(f"""
   Best model: {best_name}
   Best OOF F1: {best_oof:.4f}
   Estimated LB: {best_est_lb:.4f}

   Comparison to baselines:
   - v34a XGBoost: OOF 0.6667 -> LB 0.6907
   - v92d XGBoost+Adv: OOF ~0.67 -> LB 0.6986

   Next steps:
   1. Submit best v115 variants to Kaggle
   2. If any beat v92d (0.6986), use for ensemble
   3. Run Optuna on best performing model
   4. Create final ensemble: best XGBoost + best LightGBM
""")

print("=" * 80)
print("v115 Complete - Submit to Kaggle!")
print("=" * 80)
