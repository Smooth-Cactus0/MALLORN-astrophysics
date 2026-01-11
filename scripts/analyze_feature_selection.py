"""
Feature Selection Analysis for MALLORN

Goals:
1. Compare feature importances from v34a (XGBoost) and v77 (LightGBM)
2. Identify high signal-to-noise features (important in BOTH models)
3. Categorize physics-based vs statistical features
4. Recommend a curated feature set

Physics-based features for TDE detection:
- Color features: TDEs maintain hot temperatures (stay blue)
- Bazin parameters: Characteristic rise/fall shape
- GP color evolution: How colors change post-peak
- Skewness: Light curve asymmetry
- Duration features: TDEs have specific timescales
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

base_path = Path(__file__).parent.parent

print("=" * 80)
print("FEATURE SELECTION ANALYSIS")
print("=" * 80)

# ====================
# 1. LOAD FEATURE IMPORTANCES
# ====================
print("\n1. Loading feature importances...")

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

with open(base_path / 'data/processed/v77_artifacts.pkl', 'rb') as f:
    v77 = pickle.load(f)

xgb_imp = v34a['feature_importance'].copy()
lgb_imp = v77['feature_importance'].copy()

print(f"   v34a XGBoost features: {len(xgb_imp)}")
print(f"   v77 LightGBM features: {len(lgb_imp)}")

# ====================
# 2. NORMALIZE AND MERGE
# ====================
print("\n2. Normalizing and merging importances...")

# Normalize to 0-100 scale
xgb_imp['xgb_norm'] = 100 * xgb_imp['importance'] / xgb_imp['importance'].max()
lgb_imp['lgb_norm'] = 100 * lgb_imp['importance'] / lgb_imp['importance'].max()

# Merge
merged = xgb_imp[['feature', 'xgb_norm']].merge(
    lgb_imp[['feature', 'lgb_norm']], on='feature', how='inner'
)

# Calculate stability score (geometric mean - rewards features important in BOTH)
merged['stability'] = np.sqrt(merged['xgb_norm'] * merged['lgb_norm'])

# Calculate SNR proxy (average importance / difference)
merged['avg_imp'] = (merged['xgb_norm'] + merged['lgb_norm']) / 2
merged['imp_diff'] = np.abs(merged['xgb_norm'] - merged['lgb_norm'])
merged['snr'] = merged['avg_imp'] / (merged['imp_diff'] + 1)  # +1 to avoid division by zero

merged = merged.sort_values('stability', ascending=False)

print(f"   Merged features: {len(merged)}")

# ====================
# 3. CATEGORIZE FEATURES
# ====================
print("\n3. Categorizing features by physics relevance...")

# Define physics-based feature patterns
physics_patterns = {
    'color': ['color', 'gr_', 'ri_', '_gr', '_ri'],  # Color features
    'bazin': ['bazin'],  # Bazin model parameters
    'gp_physics': ['gp_gr', 'gp_ri', 'gp_flux', 'gp2d'],  # GP-derived physics
    'shape': ['skew', 'kurt', 'rise', 'fall', 'decay', 'duration', 'peak'],  # Light curve shape
    'variability': ['beyond', 'std', 'amplitude', 'rebrightening'],  # Variability metrics
}

statistical_patterns = {
    'basic_stats': ['mean', 'median', 'min', 'max', '_p10', '_p25', '_p50', '_p75', '_p90'],
    'counts': ['n_obs', 'count', 'n_det'],
    'time': ['time_span', 'cadence'],
}

def categorize_feature(name):
    name_lower = name.lower()

    # Check physics patterns first
    for category, patterns in physics_patterns.items():
        if any(p in name_lower for p in patterns):
            return f'physics_{category}'

    # Check statistical patterns
    for category, patterns in statistical_patterns.items():
        if any(p in name_lower for p in patterns):
            return f'stat_{category}'

    return 'other'

merged['category'] = merged['feature'].apply(categorize_feature)

# ====================
# 4. ANALYZE BY CATEGORY
# ====================
print("\n4. Feature importance by category...")

category_stats = merged.groupby('category').agg({
    'stability': ['mean', 'count'],
    'snr': 'mean'
}).round(2)
category_stats.columns = ['avg_stability', 'count', 'avg_snr']
category_stats = category_stats.sort_values('avg_stability', ascending=False)

print(category_stats.to_string())

# ====================
# 5. TOP FEATURES BY STABILITY
# ====================
print("\n" + "=" * 80)
print("TOP 50 HIGH-SNR FEATURES (Stable across both models)")
print("=" * 80)

print(f"\n{'Rank':<5} {'Feature':<40} {'XGB':<8} {'LGB':<8} {'Stability':<10} {'Category':<20}")
print("-" * 95)

for i, row in merged.head(50).iterrows():
    rank = list(merged['feature']).index(row['feature']) + 1
    print(f"{rank:<5} {row['feature']:<40} {row['xgb_norm']:<8.1f} {row['lgb_norm']:<8.1f} {row['stability']:<10.1f} {row['category']:<20}")

# ====================
# 6. PHYSICS-FOCUSED SELECTION
# ====================
print("\n" + "=" * 80)
print("PHYSICS-FOCUSED FEATURE SELECTION")
print("=" * 80)

# Select features that are:
# 1. Physics-based (not pure statistics)
# 2. High stability (important in both models)

physics_features = merged[merged['category'].str.startswith('physics')]
physics_features = physics_features.sort_values('stability', ascending=False)

print(f"\nPhysics-based features: {len(physics_features)}")
print(f"\nTop 40 physics features by stability:")
print(f"\n{'Rank':<5} {'Feature':<45} {'Stability':<10} {'Category':<20}")
print("-" * 85)

for i, (_, row) in enumerate(physics_features.head(40).iterrows(), 1):
    print(f"{i:<5} {row['feature']:<45} {row['stability']:<10.1f} {row['category']:<20}")

# ====================
# 7. RECOMMENDED FEATURE SET
# ====================
print("\n" + "=" * 80)
print("RECOMMENDED FEATURE SET")
print("=" * 80)

# Strategy:
# - Top 30 physics features (high stability)
# - Top 10 statistical features (for robustness)
# - Exclude low-stability features

top_physics = physics_features.head(30)['feature'].tolist()
top_stats = merged[~merged['category'].str.startswith('physics')].head(10)['feature'].tolist()

recommended = top_physics + top_stats

print(f"\nRecommended features: {len(recommended)}")
print(f"   Physics-based: {len(top_physics)}")
print(f"   Statistical: {len(top_stats)}")

print(f"\n{'Physics Features:'}")
for i, f in enumerate(top_physics, 1):
    cat = merged[merged['feature'] == f]['category'].values[0]
    stab = merged[merged['feature'] == f]['stability'].values[0]
    print(f"   {i:2d}. {f:<40} ({cat}, stability={stab:.1f})")

print(f"\n{'Statistical Features (for robustness):'}")
for i, f in enumerate(top_stats, 1):
    cat = merged[merged['feature'] == f]['category'].values[0]
    stab = merged[merged['feature'] == f]['stability'].values[0]
    print(f"   {i:2d}. {f:<40} ({cat}, stability={stab:.1f})")

# ====================
# 8. SAVE ANALYSIS
# ====================
analysis = {
    'merged_importance': merged,
    'category_stats': category_stats,
    'recommended_features': recommended,
    'physics_features': top_physics,
    'stat_features': top_stats
}

with open(base_path / 'data/processed/feature_selection_analysis.pkl', 'wb') as f:
    pickle.dump(analysis, f)

print(f"\nAnalysis saved to data/processed/feature_selection_analysis.pkl")

# ====================
# 9. ALTERNATIVE SETS
# ====================
print("\n" + "=" * 80)
print("ALTERNATIVE FEATURE SETS TO TEST")
print("=" * 80)

# Set A: Ultra-lean (top 20 physics only)
set_a = physics_features.head(20)['feature'].tolist()
print(f"\nSet A (Ultra-lean): {len(set_a)} features - top 20 physics only")

# Set B: Balanced (top 30 physics + 10 stats)
set_b = recommended
print(f"Set B (Balanced): {len(set_b)} features - 30 physics + 10 stats")

# Set C: Extended (top 50 physics + 20 stats)
set_c = physics_features.head(50)['feature'].tolist() + \
        merged[~merged['category'].str.startswith('physics')].head(20)['feature'].tolist()
print(f"Set C (Extended): {len(set_c)} features - 50 physics + 20 stats")

# Save alternative sets
alt_sets = {
    'set_a_ultra_lean': set_a,
    'set_b_balanced': set_b,
    'set_c_extended': set_c
}

with open(base_path / 'data/processed/feature_sets.pkl', 'wb') as f:
    pickle.dump(alt_sets, f)

print(f"\nFeature sets saved to data/processed/feature_sets.pkl")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
