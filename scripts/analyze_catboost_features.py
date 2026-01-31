"""
MALLORN: CatBoost Feature Analysis
==================================

Analyze CatBoost features to improve its unique TDE discovery capability:
1. Feature importance ranking
2. Low-importance features (candidates for removal)
3. Highly correlated/redundant features
4. Feature clusters
"""

import sys
import pickle
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier, Pool
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent

print("=" * 70)
print("MALLORN: CatBoost Feature Analysis")
print("=" * 70)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")

with gzip.open(base_path / 'data/kaggle_ensemble_package.pkl.gz', 'rb') as f:
    package = pickle.load(f)

train_features = package['train_features']
test_features = package['test_features']
y = package['y']
sample_weights = package['sample_weights']

with open(base_path / 'data/processed/v118_catboost_artifacts.pkl', 'rb') as f:
    cb_arts = pickle.load(f)

feature_names = cb_arts['feature_names']
best_params = cb_arts['best_params']

print(f"   Total features: {len(feature_names)}")
print(f"   Training samples: {len(y)}")

# Prepare feature matrix
X = train_features[feature_names].values
X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)

# ============================================================================
# 2. TRAIN CATBOOST FOR FEATURE IMPORTANCE
# ============================================================================
print("\n[2/5] Training CatBoost to extract feature importance...")

scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

params = {
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'random_seed': 42,
    'verbose': False,
    'allow_writing_files': False,
    'scale_pos_weight': scale_pos_weight,
    **best_params
}

train_pool = Pool(X, y, weight=sample_weights, feature_names=feature_names)

model = CatBoostClassifier(**params)
model.fit(train_pool, verbose=False)

# Get feature importance
importance = model.get_feature_importance()
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False).reset_index(drop=True)

print("\n   Top 30 Most Important Features:")
print(f"   {'Rank':<6} {'Feature':<35} {'Importance':<12}")
print("   " + "-" * 53)
for i, row in importance_df.head(30).iterrows():
    print(f"   {i+1:<6} {row['feature']:<35} {row['importance']:<12.2f}")

# ============================================================================
# 3. ANALYZE LOW-IMPORTANCE FEATURES
# ============================================================================
print("\n[3/5] Analyzing low-importance features...")

# Features with near-zero importance
zero_importance = importance_df[importance_df['importance'] < 0.1]
low_importance = importance_df[importance_df['importance'] < 1.0]
very_low = importance_df[importance_df['importance'] < 0.5]

print(f"\n   Importance Distribution:")
print(f"   - Zero importance (<0.1): {len(zero_importance)} features")
print(f"   - Very low (<0.5): {len(very_low)} features")
print(f"   - Low (<1.0): {len(low_importance)} features")
print(f"   - Medium+ (>=1.0): {len(importance_df) - len(low_importance)} features")

print(f"\n   Zero/Near-Zero Importance Features ({len(zero_importance)}):")
for i, row in zero_importance.iterrows():
    print(f"      {row['feature']}: {row['importance']:.3f}")

# ============================================================================
# 4. CORRELATION ANALYSIS
# ============================================================================
print("\n[4/5] Analyzing feature correlations...")

# Use top 100 features for correlation analysis (memory efficient)
top_features = importance_df.head(100)['feature'].tolist()
X_top = train_features[top_features].values
X_top = np.nan_to_num(X_top, nan=0, posinf=1e10, neginf=-1e10)

# Compute correlation matrix
print("   Computing correlation matrix for top 100 features...")
corr_matrix = np.corrcoef(X_top.T)

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(top_features)):
    for j in range(i+1, len(top_features)):
        corr = abs(corr_matrix[i, j])
        if corr > 0.9:
            high_corr_pairs.append({
                'feature1': top_features[i],
                'feature2': top_features[j],
                'correlation': corr,
                'imp1': importance_df[importance_df['feature'] == top_features[i]]['importance'].values[0],
                'imp2': importance_df[importance_df['feature'] == top_features[j]]['importance'].values[0],
            })

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)

print(f"\n   Highly Correlated Feature Pairs (|r| > 0.9): {len(high_corr_df)}")
if len(high_corr_df) > 0:
    print(f"   {'Feature 1':<30} {'Feature 2':<30} {'Corr':<8} {'Imp1':<8} {'Imp2':<8}")
    print("   " + "-" * 84)
    for _, row in high_corr_df.head(20).iterrows():
        print(f"   {row['feature1']:<30} {row['feature2']:<30} {row['correlation']:<8.3f} {row['imp1']:<8.1f} {row['imp2']:<8.1f}")

# Identify redundant features (lower importance in correlated pair)
redundant_features = set()
for _, row in high_corr_df.iterrows():
    if row['imp1'] > row['imp2']:
        redundant_features.add(row['feature2'])
    else:
        redundant_features.add(row['feature1'])

print(f"\n   Potentially Redundant Features (lower imp in corr pair): {len(redundant_features)}")

# ============================================================================
# 5. FEATURE REDUCTION EXPERIMENT
# ============================================================================
print("\n[5/5] Testing feature reduction impact...")

from sklearn.model_selection import StratifiedKFold

def evaluate_features(feature_list, name):
    """Evaluate CatBoost with given feature set."""
    X_subset = train_features[feature_list].values
    X_subset = np.nan_to_num(X_subset, nan=0, posinf=1e10, neginf=-1e10)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_subset, y)):
        X_tr, X_val = X_subset[train_idx], X_subset[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        fold_weights = sample_weights[train_idx]

        train_pool = Pool(X_tr, y_tr, weight=fold_weights, feature_names=feature_list)
        val_pool = Pool(X_val, y_val, feature_names=feature_list)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    # Find best F1
    best_f1 = 0
    for t in np.linspace(0.03, 0.7, 100):
        f1 = f1_score(y, (oof_preds > t).astype(int))
        best_f1 = max(best_f1, f1)

    return best_f1, oof_preds

# Test different feature sets
print("\n   Testing feature reduction strategies...")

results = {}

# Baseline: All features
print("   [A] All features (230)...", end=" ", flush=True)
f1_all, _ = evaluate_features(feature_names, "all")
print(f"F1={f1_all:.4f}")
results['all_230'] = f1_all

# Top 150 features
top150 = importance_df.head(150)['feature'].tolist()
print("   [B] Top 150 features...", end=" ", flush=True)
f1_150, _ = evaluate_features(top150, "top150")
print(f"F1={f1_150:.4f}")
results['top_150'] = f1_150

# Top 100 features
top100 = importance_df.head(100)['feature'].tolist()
print("   [C] Top 100 features...", end=" ", flush=True)
f1_100, _ = evaluate_features(top100, "top100")
print(f"F1={f1_100:.4f}")
results['top_100'] = f1_100

# Top 75 features
top75 = importance_df.head(75)['feature'].tolist()
print("   [D] Top 75 features...", end=" ", flush=True)
f1_75, _ = evaluate_features(top75, "top75")
print(f"F1={f1_75:.4f}")
results['top_75'] = f1_75

# Top 50 features
top50 = importance_df.head(50)['feature'].tolist()
print("   [E] Top 50 features...", end=" ", flush=True)
f1_50, _ = evaluate_features(top50, "top50")
print(f"F1={f1_50:.4f}")
results['top_50'] = f1_50

# Remove redundant features
non_redundant = [f for f in feature_names if f not in redundant_features]
print(f"   [F] Non-redundant features ({len(non_redundant)})...", end=" ", flush=True)
f1_nr, _ = evaluate_features(non_redundant, "non_redundant")
print(f"F1={f1_nr:.4f}")
results['non_redundant'] = f1_nr

# Top 100 without redundant
top100_nr = [f for f in top100 if f not in redundant_features]
print(f"   [G] Top 100 non-redundant ({len(top100_nr)})...", end=" ", flush=True)
f1_100_nr, _ = evaluate_features(top100_nr, "top100_nr")
print(f"F1={f1_100_nr:.4f}")
results['top_100_nr'] = f1_100_nr

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FEATURE ANALYSIS SUMMARY")
print("=" * 70)

print("\n   Feature Reduction Results:")
print(f"   {'Feature Set':<30} {'Count':<10} {'OOF F1':<10} {'vs All':<10}")
print("   " + "-" * 60)
for name, f1 in sorted(results.items(), key=lambda x: x[1], reverse=True):
    count = {'all_230': 230, 'top_150': 150, 'top_100': 100, 'top_75': 75,
             'top_50': 50, 'non_redundant': len(non_redundant),
             'top_100_nr': len(top100_nr)}.get(name, '?')
    diff = f1 - f1_all
    print(f"   {name:<30} {count:<10} {f1:<10.4f} {diff:+.4f}")

# Find best feature set
best_set = max(results.items(), key=lambda x: x[1])
print(f"\n   Best feature set: {best_set[0]} (F1={best_set[1]:.4f})")

# Save results
analysis = {
    'importance_df': importance_df,
    'high_corr_pairs': high_corr_df,
    'redundant_features': redundant_features,
    'feature_sets': {
        'top150': top150,
        'top100': top100,
        'top75': top75,
        'top50': top50,
        'non_redundant': non_redundant,
        'top100_nr': top100_nr,
    },
    'results': results,
}

with open(base_path / 'data/processed/catboost_feature_analysis.pkl', 'wb') as f:
    pickle.dump(analysis, f)

print(f"""
Key Findings:
   1. {len(zero_importance)} features have near-zero importance
   2. {len(high_corr_df)} highly correlated feature pairs found
   3. {len(redundant_features)} potentially redundant features identified
   4. Best performing feature set: {best_set[0]}

Recommendations:
   - If top_100 or top_75 performs better, use reduced feature set
   - This can reduce overfitting and improve generalization
   - May improve CatBoost's unique TDE discovery rate
""")
