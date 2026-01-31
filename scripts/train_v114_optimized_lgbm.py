"""
MALLORN v114: Optimized LightGBM with Best Research Features

Based on LB feedback from v113 experiments:
- v113b (nuclear): LB 0.6717 (BEST) - nuclear features help!
- v113c (color_peak): LB 0.6713 - color at peak helps
- v113f (all_research): LB 0.672 - combined is good
- v113d (mhps): LB 0.67 - variability helps
- v113a (powerlaw): LB 0.65 (WORST) - power law features HURT

Strategy:
1. Select top features from EACH category based on importance
2. Limit power law features (they hurt LB)
3. Emphasize nuclear + color_peak + mhps features
4. Try different regularization levels
5. Use feature importance to create optimal subset
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import lightgbm as lgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v114: Optimized LightGBM with Best Research Features")
print("=" * 80)

# ====================
# 1. LOAD V113 ARTIFACTS
# ====================
print("\n1. Loading v113 artifacts and analyzing feature importance...")

with open(base_path / 'data/processed/v113_research_artifacts.pkl', 'rb') as f:
    v113 = pickle.load(f)

# LB scores from Kaggle submissions
lb_scores = {
    'v113a_powerlaw': 0.65,
    'v113b_nuclear': 0.6717,
    'v113c_color_peak': 0.6713,
    'v113d_mhps': 0.67,
    'v113e_luminosity': None,  # Not submitted yet
    'v113f_all_research': 0.672
}

print("\n   LB Performance (from submissions):")
for name, score in lb_scores.items():
    if score:
        oof = v113['results'][name]['oof_f1']
        delta = score - oof
        print(f"      {name}: LB={score:.4f} (OOF={oof:.4f}, delta={delta:+.4f})")

# ====================
# 2. ANALYZE FEATURE IMPORTANCE BY CATEGORY
# ====================
print("\n2. Analyzing feature importance from best models...")

research_features = v113['research_features']

# Define feature categories
powerlaw_features = [c for c in research_features if 'powerlaw' in c]
nuclear_features = [c for c in research_features if 'nuclear' in c]
color_peak_features = [c for c in research_features if 'color_at_peak' in c or 'color_peak' in c]
mhps_features = [c for c in research_features if 'mhps' in c]
luminosity_features = [c for c in research_features if 'luminosity' in c]

# Get feature importance from v113f (all features)
all_result = v113['results']['v113f_all_research']
all_importance = pd.DataFrame({
    'feature': all_result['feature_names'],
    'importance': all_result['feature_importance']
}).sort_values('importance', ascending=False)

# Identify best research features (top N from each category)
def get_top_features(feature_list, importance_df, n=3):
    """Get top N features from a category based on importance."""
    cat_features = importance_df[importance_df['feature'].isin(feature_list)]
    return cat_features.head(n)['feature'].tolist()

print("\n   Top features by category (from v113f importance):")

# Nuclear (BEST LB) - take more
top_nuclear = get_top_features(nuclear_features, all_importance, n=4)
print(f"   Nuclear (LB=0.6717): {top_nuclear}")

# Color at peak (good LB)
top_color = get_top_features(color_peak_features, all_importance, n=4)
print(f"   Color Peak (LB=0.6713): {top_color}")

# MHPS (good LB)
top_mhps = get_top_features(mhps_features, all_importance, n=4)
print(f"   MHPS (LB=0.67): {top_mhps}")

# Luminosity (unknown LB, include top 2)
top_lum = get_top_features(luminosity_features, all_importance, n=2)
print(f"   Luminosity: {top_lum}")

# Power law (WORST LB) - only take 1-2 most important
top_powerlaw = get_top_features(powerlaw_features, all_importance, n=2)
print(f"   Powerlaw (LB=0.65 - LIMIT): {top_powerlaw}")

# ====================
# 3. LOAD DATA
# ====================
print("\n3. Loading data...")

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDE)")

# Load adversarial weights
with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)
sample_weights = adv_results['sample_weights']

# Load all features
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']

tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
with open(base_path / 'data/processed/bazin_features_cache.pkl', 'rb') as f:
    bazin_cache = pickle.load(f)
with open(base_path / 'data/processed/research_features_cache.pkl', 'rb') as f:
    research_cache = pickle.load(f)

# Merge all
train_all = train_base.merge(tde_cached['train'], on='object_id', how='left')
train_all = train_all.merge(gp2d_data['train'], on='object_id', how='left')
train_all = train_all.merge(bazin_cache['train'], on='object_id', how='left')
train_all = train_all.merge(research_cache['train'], on='object_id', how='left')

test_all = test_base.merge(tde_cached['test'], on='object_id', how='left')
test_all = test_all.merge(gp2d_data['test'], on='object_id', how='left')
test_all = test_all.merge(bazin_cache['test'], on='object_id', how='left')
test_all = test_all.merge(research_cache['test'], on='object_id', how='left')

# ====================
# 4. DEFINE OPTIMIZED FEATURE SETS
# ====================
print("\n4. Defining optimized feature sets...")

# Base v34a features (remove adversarial discriminative)
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)
v34a_features = v34a['feature_names']
adv_discriminative = ['all_rise_time', 'all_asymmetry']
v34a_clean = [f for f in v34a_features if f in train_all.columns and f not in adv_discriminative]

# Different feature combinations to test
experiments = {
    'v114a_best_research': {
        'description': 'v34a + Best features from each category (curated)',
        'extra_features': top_nuclear + top_color + top_mhps + top_lum,  # NO powerlaw
        'params_variant': 'balanced'
    },
    'v114b_best_with_limited_powerlaw': {
        'description': 'v34a + Best features + limited powerlaw (top 2)',
        'extra_features': top_nuclear + top_color + top_mhps + top_lum + top_powerlaw[:2],
        'params_variant': 'balanced'
    },
    'v114c_nuclear_heavy': {
        'description': 'v34a + ALL nuclear + top from others',
        'extra_features': nuclear_features + top_color + top_mhps[:2],
        'params_variant': 'regularized'
    },
    'v114d_minimal_research': {
        'description': 'v34a + Only top 2 from each (minimal)',
        'extra_features': top_nuclear[:2] + top_color[:2] + top_mhps[:2],
        'params_variant': 'light'
    },
    'v114e_importance_top30': {
        'description': 'Top 30 research features by importance',
        'extra_features': all_importance[all_importance['feature'].isin(research_features)].head(30)['feature'].tolist(),
        'params_variant': 'balanced'
    },
}

print(f"\n   Experiments:")
for name, cfg in experiments.items():
    print(f"      {name}: {len(cfg['extra_features'])} new features ({cfg['params_variant']})")

# ====================
# 5. DEFINE LGB PARAMETER VARIANTS
# ====================
print("\n5. Defining LightGBM parameter variants...")

# Based on what we learned: lower complexity helps LB
param_variants = {
    'balanced': {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
        'max_depth': 4,
        'num_leaves': 12,
        'learning_rate': 0.025,
        'n_estimators': 600,
        'feature_fraction': 0.45,
        'bagging_fraction': 0.65,
        'bagging_freq': 5,
        'reg_alpha': 2.5,
        'reg_lambda': 4.0,
        'min_child_samples': 35,
    },
    'regularized': {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
        'max_depth': 3,  # Lower depth
        'num_leaves': 8,  # Fewer leaves
        'learning_rate': 0.02,
        'n_estimators': 700,
        'feature_fraction': 0.35,  # More aggressive
        'bagging_fraction': 0.55,
        'bagging_freq': 7,
        'reg_alpha': 4.0,  # Higher regularization
        'reg_lambda': 6.0,
        'min_child_samples': 50,
    },
    'light': {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
        'max_depth': 5,
        'num_leaves': 15,
        'learning_rate': 0.03,
        'n_estimators': 500,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'reg_alpha': 2.0,
        'reg_lambda': 3.0,
        'min_child_samples': 30,
    },
}

# Add class weight to all
scale_pos_weight = len(y[y==0]) / len(y[y==1])
for variant in param_variants.values():
    variant['scale_pos_weight'] = scale_pos_weight

# ====================
# 6. TRAIN ALL EXPERIMENTS
# ====================
print("\n" + "=" * 80)
print("TRAINING OPTIMIZED EXPERIMENTS")
print("=" * 80)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

results = {}

for exp_name, exp_config in experiments.items():
    print(f"\n{'-' * 60}")
    print(f"   {exp_name}: {exp_config['description']}")
    print(f"{'-' * 60}")

    # Get parameters
    lgb_params = param_variants[exp_config['params_variant']]

    # Combine features
    feature_set = v34a_clean + exp_config['extra_features']
    # Remove duplicates while preserving order
    feature_set = list(dict.fromkeys(feature_set))
    available_features = [f for f in feature_set if f in train_all.columns]
    print(f"   Features: {len(available_features)} ({exp_config['params_variant']} params)")

    X_train = train_all[available_features].values
    X_test = test_all[available_features].values

    # Handle infinities
    X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

    oof_preds = np.zeros(len(y))
    test_preds = np.zeros((len(X_test), n_folds))
    feature_importance = np.zeros(len(available_features))
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        fold_weights = sample_weights[train_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr, weight=fold_weights, feature_name=available_features)
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

        importance = model.feature_importance(importance_type='gain')
        feature_importance += importance

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

    # Estimate LB based on v113 patterns
    # v113b: OOF 0.6328 -> LB 0.6717 (delta +0.039)
    estimated_lb = best_f1 + 0.035  # Conservative estimate
    print(f"   Estimated LB: ~{estimated_lb:.4f} (based on v113 pattern)")

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

print(f"\n{'Experiment':<30} {'OOF F1':<10} {'Est. LB':<10} {'Fold Std':<10} {'Features':<10}")
print("-" * 75)

# Baselines
print(f"{'v34a XGB (actual LB=0.6907)':<30} {'0.6667':<10} {'0.6907':<10} {'-':<10} {'224':<10}")
print(f"{'v92d XGB+Adv (LB=0.6986)':<30} {'~0.67':<10} {'0.6986':<10} {'-':<10} {'222':<10}")
print(f"{'v113b nuclear (LB=0.6717)':<30} {'0.6328':<10} {'0.6717':<10} {'0.0829':<10} {'225':<10}")
print("-" * 75)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    n_features = len(res['feature_names'])
    print(f"{name:<30} {res['oof_f1']:<10.4f} {res['estimated_lb']:<10.4f} {res['fold_std']:<10.4f} {n_features:<10}")

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
# 9. SAVE ARTIFACTS
# ====================
print("\n9. Saving artifacts...")

artifacts = {
    'results': results,
    'experiments': experiments,
    'param_variants': param_variants,
    'lb_scores_v113': lb_scores,
    'top_features_by_category': {
        'nuclear': top_nuclear,
        'color_peak': top_color,
        'mhps': top_mhps,
        'luminosity': top_lum,
        'powerlaw': top_powerlaw
    }
}

with open(base_path / 'data/processed/v114_optimized_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("   Saved: v114_optimized_artifacts.pkl")

# ====================
# 10. ANALYSIS: WHICH NEW FEATURES HELP MOST?
# ====================
print("\n" + "=" * 80)
print("FEATURE ANALYSIS: Which research features rank highest?")
print("=" * 80)

best_exp = sorted_results[0][0]
best_result = results[best_exp]

importance_df = pd.DataFrame({
    'feature': best_result['feature_names'],
    'importance': best_result['feature_importance']
}).sort_values('importance', ascending=False)

# All research features
all_research = nuclear_features + color_peak_features + mhps_features + luminosity_features + powerlaw_features

print(f"\n   Best experiment: {best_exp}")
print(f"\n   Top 10 research features in this model:")
research_in_model = importance_df[importance_df['feature'].isin(all_research)]
for i, (_, row) in enumerate(research_in_model.head(10).iterrows()):
    rank = list(importance_df['feature']).index(row['feature']) + 1
    # Identify category
    if row['feature'] in nuclear_features:
        cat = 'nuclear'
    elif row['feature'] in color_peak_features:
        cat = 'color'
    elif row['feature'] in mhps_features:
        cat = 'mhps'
    elif row['feature'] in luminosity_features:
        cat = 'lum'
    else:
        cat = 'power'
    print(f"      #{rank:3d}: {row['feature']:<35} {row['importance']:8.1f} [{cat}]")

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

   Key insight from v113 LB scores:
   - Nuclear features (LB 0.6717) > Color peak (0.6713) > All (0.672) > MHPS (0.67)
   - Power law features HURT (LB 0.65)

   Recommendations:
   1. Submit all v114 experiments to compare
   2. If v114 beats v113b (0.6717), we're on the right track
   3. Consider ensemble with v92d XGBoost for final submission
""")

print("=" * 80)
print("v114 Complete - Submit to Kaggle!")
print("=" * 80)
