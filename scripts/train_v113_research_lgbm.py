"""
MALLORN v113: LightGBM with Research-Based Features + Adversarial Weights

Based on extensive research from:
- PLAsTiCC 1st place (Avocado/Kyle Boone)
- ALeRCE TDE classifier
- FLEET algorithm
- PLAsTiCC 3rd place solution

Five new feature categories:
1. Power law fit quality (t^-5/3 TDE decay)
2. Nuclear position proxy
3. Color at peak
4. Multi-timescale variability (MHPS)
5. Absolute luminosity

Training approach:
- LightGBM with nyanp-inspired parameters (max_depth=3, reg_lambda=3)
- Adversarial sample weights (from v92d best)
- Feature ablation study: one submission per feature category
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
print("MALLORN v113: Research-Based LightGBM with Adversarial Weights")
print("=" * 80)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...")

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_lc = data['train_lc']
test_lc = data['test_lc']

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
# 3. LOAD EXISTING FEATURES (v34a baseline)
# ====================
print("\n3. Loading v34a baseline features...")

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

# Merge base features
train_all = train_base.merge(train_tde, on='object_id', how='left')
train_all = train_all.merge(train_gp2d, on='object_id', how='left')
train_all = train_all.merge(train_bazin, on='object_id', how='left')

test_all = test_base.merge(test_tde, on='object_id', how='left')
test_all = test_all.merge(test_gp2d, on='object_id', how='left')
test_all = test_all.merge(test_bazin, on='object_id', how='left')

# Remove adversarial-discriminative features (reduce overfitting)
adv_discriminative = ['all_rise_time', 'all_asymmetry']
v34a_clean = [f for f in v34a_features if f in train_all.columns and f not in adv_discriminative]

print(f"   v34a features: {len(v34a_features)} -> {len(v34a_clean)} (removed adversarial)")

# ====================
# 4. EXTRACT RESEARCH FEATURES
# ====================
print("\n4. Extracting research features...")

cache_path = base_path / 'data/processed/research_features_cache.pkl'

if cache_path.exists():
    print("   Loading from cache...")
    with open(cache_path, 'rb') as f:
        research_cache = pickle.load(f)
    train_research = research_cache['train']
    test_research = research_cache['test']
else:
    print("   Computing new features (this may take a few minutes)...")
    from features.research_features import extract_research_features

    print("   Processing training set...")
    train_research = extract_research_features(train_lc, train_ids, train_meta, verbose=True)

    print("   Processing test set...")
    test_research = extract_research_features(test_lc, test_ids, test_meta, verbose=True)

    # Cache for future use
    with open(cache_path, 'wb') as f:
        pickle.dump({'train': train_research, 'test': test_research}, f)
    print(f"   Cached to: {cache_path.name}")

research_feature_cols = [c for c in train_research.columns if c != 'object_id']
print(f"   Research features: {len(research_feature_cols)}")

# Merge research features
train_all = train_all.merge(train_research, on='object_id', how='left')
test_all = test_all.merge(test_research, on='object_id', how='left')

# ====================
# 5. DEFINE FEATURE CATEGORIES
# ====================
print("\n5. Defining feature categories for ablation study...")

# Category 1: Power law features
powerlaw_features = [c for c in research_feature_cols if 'powerlaw' in c]
print(f"   1. Power law features: {len(powerlaw_features)}")

# Category 2: Nuclear position features
nuclear_features = [c for c in research_feature_cols if 'nuclear' in c]
print(f"   2. Nuclear position features: {len(nuclear_features)}")

# Category 3: Color at peak features
color_peak_features = [c for c in research_feature_cols if 'color_at_peak' in c or 'color_peak' in c]
print(f"   3. Color at peak features: {len(color_peak_features)}")

# Category 4: MHPS features
mhps_features = [c for c in research_feature_cols if 'mhps' in c]
print(f"   4. MHPS features: {len(mhps_features)}")

# Category 5: Luminosity features
luminosity_features = [c for c in research_feature_cols if 'luminosity' in c or 'mean_luminosity' in c or 'peak_luminosity' in c]
print(f"   5. Luminosity features: {len(luminosity_features)}")

# ====================
# 6. TRAINING CONFIGURATIONS
# ====================
print("\n6. Setting up training configurations...")

# LightGBM parameters based on PLAsTiCC 3rd place + constrained regularization
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,

    # Regularization (nyanp-inspired + our constrained search)
    'max_depth': 4,
    'num_leaves': 12,
    'learning_rate': 0.03,
    'n_estimators': 600,

    # Aggressive subsampling
    'feature_fraction': 0.4,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,

    # Strong regularization
    'reg_alpha': 3.0,
    'reg_lambda': 5.0,
    'min_child_samples': 40,

    # Class imbalance
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
}

print(f"   Key params: max_depth={lgb_params['max_depth']}, num_leaves={lgb_params['num_leaves']}")
print(f"   Regularization: alpha={lgb_params['reg_alpha']}, lambda={lgb_params['reg_lambda']}")
print(f"   Subsampling: feature={lgb_params['feature_fraction']}, bagging={lgb_params['bagging_fraction']}")

# Define experiment configurations
experiments = {
    'v113a_powerlaw': {
        'description': 'v34a + Power Law (t^-5/3) features',
        'extra_features': powerlaw_features
    },
    'v113b_nuclear': {
        'description': 'v34a + Nuclear Position proxy features',
        'extra_features': nuclear_features
    },
    'v113c_color_peak': {
        'description': 'v34a + Color at Peak features',
        'extra_features': color_peak_features
    },
    'v113d_mhps': {
        'description': 'v34a + MHPS variability features',
        'extra_features': mhps_features
    },
    'v113e_luminosity': {
        'description': 'v34a + Absolute Luminosity features',
        'extra_features': luminosity_features
    },
    'v113f_all_research': {
        'description': 'v34a + ALL research features',
        'extra_features': research_feature_cols
    },
}

# ====================
# 7. TRAIN ALL EXPERIMENTS
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

    # Combine features
    feature_set = v34a_clean + exp_config['extra_features']
    available_features = [f for f in feature_set if f in train_all.columns]
    print(f"   Features: {len(available_features)}")

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

        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        feature_importance += importance

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

    # Save results
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
        'config': exp_config
    }

# ====================
# 8. RESULTS COMPARISON
# ====================
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

print(f"\n{'Experiment':<25} {'OOF F1':<10} {'Fold Std':<10} {'TP':<6} {'FP':<6} {'FN':<6}")
print("-" * 70)

# Add baseline for comparison
print(f"{'v34a XGB (LB=0.6907)':<25} {'0.6667':<10} {'~0.06':<10} {'-':<6} {'-':<6} {'-':<6}")
print(f"{'v92d XGB+Adv (LB=0.6986)':<25} {'~0.67':<10} {'~0.05':<10} {'-':<6} {'-':<6} {'-':<6}")
print("-" * 70)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    print(f"{name:<25} {res['oof_f1']:<10.4f} {res['fold_std']:<10.4f} {res['confusion']['tp']:<6} {res['confusion']['fp']:<6} {res['confusion']['fn']:<6}")

# ====================
# 9. CREATE SUBMISSIONS
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

    print(f"   {filename}: OOF F1={res['oof_f1']:.4f}, TDEs={test_binary.sum()}")

# ====================
# 10. SAVE ARTIFACTS
# ====================
print("\n10. Saving artifacts...")

artifacts = {
    'results': results,
    'lgb_params': lgb_params,
    'v34a_clean_features': v34a_clean,
    'research_features': research_feature_cols,
    'experiments': experiments
}

with open(base_path / 'data/processed/v113_research_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("   Saved: v113_research_artifacts.pkl")

# ====================
# 11. FEATURE IMPORTANCE ANALYSIS
# ====================
print("\n" + "=" * 80)
print("TOP FEATURES FROM BEST MODEL")
print("=" * 80)

best_exp = sorted_results[0][0]
best_result = results[best_exp]

importance_df = pd.DataFrame({
    'feature': best_result['feature_names'],
    'importance': best_result['feature_importance']
}).sort_values('importance', ascending=False)

print(f"\n   Best experiment: {best_exp}")
print(f"\n   Top 20 features:")
for i, row in importance_df.head(20).iterrows():
    is_new = '*' if row['feature'] in research_feature_cols else ''
    print(f"      {row['importance']:10.1f}  {row['feature']}{is_new}")

print("\n   (* = new research feature)")

# Count new features in top 30
top30_new = sum(1 for f in importance_df.head(30)['feature'] if f in research_feature_cols)
print(f"\n   New research features in top 30: {top30_new}")

# ====================
# 12. SUMMARY
# ====================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

best_oof = sorted_results[0][1]['oof_f1']
best_name = sorted_results[0][0]

print(f"""
   Best model: {best_name}
   Best OOF F1: {best_oof:.4f}

   Key insight: Adversarial weights + research features should improve generalization.

   LB target comparison:
   - v34a XGBoost: OOF 0.6667 -> LB 0.6907 (+0.024)
   - v92d XGBoost+Adv: OOF ~0.67 -> LB 0.6986 (+0.03)

   If our OOF {best_oof:.4f} follows same pattern:
   - Expected LB: ~{best_oof + 0.024:.4f} to ~{best_oof + 0.03:.4f}

   Submit all 6 experiments to Kaggle to identify best generalizing model!
""")

print("=" * 80)
print("v113 Complete - Submit to Kaggle!")
print("=" * 80)
