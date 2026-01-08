"""
Quick comparison of all experiments.
"""

import pickle
import pandas as pd
from pathlib import Path

base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN COMPREHENSIVE EXPERIMENT RESULTS")
print("=" * 80)

# Results from all experiments
results = [
    {'Version': 'v21', 'Name': 'Baseline XGBoost', 'OOF F1': 0.6708, 'Status': 'BEST'},
    {'Version': 'v31', 'Name': 'CatBoost Ordered', 'OOF F1': 0.5754, 'Status': 'Failed'},
    {'Version': 'v32', 'Name': 'Feature Interactions', 'OOF F1': 0.6405, 'Status': 'Failed'},
    {'Version': 'v33', 'Name': 'Diverse Ensemble', 'OOF F1': 0.6467, 'Status': 'Failed'},
]

results_df = pd.DataFrame(results)

# Add improvement column
baseline_f1 = 0.6708
results_df['vs v21'] = results_df['OOF F1'] - baseline_f1
results_df['vs v21 %'] = 100 * results_df['vs v21'] / baseline_f1

print("\nRESULTS SUMMARY:")
print(results_df.to_string(index=False))

print("\n" + "=" * 80)
print("DETAILED FINDINGS")
print("=" * 80)

print("\nExperiment 1: CatBoost with Ordered Boosting")
print("-" * 80)
print(f"OOF F1: 0.5754 (-9.54% vs v21)")
print("Key issue: CatBoost's ordered boosting did not help on this dataset")
print("Possible reasons:")
print("  - Small dataset (n=3043) may not benefit from ordered boosting")
print("  - Feature types already well-suited for XGBoost's algorithm")
print("  - Auto class weights may have hurt precision")

print("\nExperiment 2: Feature Interactions")
print("-" * 80)
print(f"OOF F1: 0.6405 (-3.03% vs v21)")
print("Created 14 physics-motivated interactions, selected top 5")
print("Top interactions: r_skew_squared (rank 7), g_r_post_20d_div_Z (rank 28)")
print("Key issue: Interactions added noise rather than signal")
print("Possible reasons:")
print("  - XGBoost already captures interactions via tree splits")
print("  - Only 5 interactions passed correlation filter (low signal)")
print("  - Physics-motivated interactions may not align with TDE vs non-TDE boundary")

print("\nExperiment 3: Diverse Ensemble")
print("-" * 80)
print(f"OOF F1: 0.6467 (-2.41% vs v21)")
print("Individual models:")
print("  - XGBoost (shallow):  F1 = 0.6488 (-2.2%)")
print("  - LightGBM (deep):    F1 = 0.6254 (-4.5%)")
print("  - CatBoost (ordered): F1 = 0.5588 (-11.2%)")
print("  - Simple Average:     F1 = 0.6467 (-2.4%)")
print("Key issue: Weak models in ensemble dragged down performance")
print("Possible reasons:")
print("  - CatBoost significantly underperformed, hurting average")
print("  - Diversity came at cost of individual model quality")
print("  - No single diverse model beat v21's configuration")

print("\n" + "=" * 80)
print("OVERALL ANALYSIS")
print("=" * 80)

print("\nWHY OPTION B (MODEL IMPROVEMENTS) FAILED:")
print("\n1. v21's XGBoost configuration is near-optimal for this problem")
print("   - Depth=5, lr=0.025, class_weight=20:1 appears ideal")
print("   - Feature set (147 features) is well-tuned")
print("\n2. Small dataset limits complexity")
print("   - ~3000 training samples insufficient for deep models")
print("   - CatBoost's ordered boosting needs more data")
print("   - LightGBM's depth=8 overfits")
print("\n3. Feature interactions redundant")
print("   - Tree models already learn interactions via splits")
print("   - Explicit interactions mostly failed correlation filter")
print("\n4. Ensemble diversity too costly")
print("   - Weak individual models hurt ensemble average")
print("   - Proper stacking might help but likely marginal")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

print("\nv21 (XGBoost, F1=0.6708, LB=0.6649) remains the BEST model.")
print("\nNo advanced modeling technique improved performance:")
print("  X CatBoost with ordered boosting")
print("  X Physics-motivated feature interactions")
print("  X Diverse ensemble (XGB + LGB + CAT)")
print("\nThis suggests:")
print("  1. v21 has reached the performance ceiling with current features")
print("  2. Further gains require NEW INFORMATION, not different models")
print("  3. Focus should shift to data/feature quality over model complexity")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

print("\nSince modeling improvements failed, consider:")
print("\n1. **External Data Augmentation**")
print("   - Add host galaxy features (if available)")
print("   - Use pre-trained astronomical embeddings (different from ASTROMER)")
print("   - Incorporate spectroscopic information")
print("\n2. **Feature Quality Improvements**")
print("   - Fix NaN-heavy features (late-time observations)")
print("   - Better GP fits (different kernels, longer time scales)")
print("   - Color corrections for extinction (EBV)")
print("\n3. **Threshold Optimization**")
print("   - v21 uses threshold ~0.30 (not 0.5)")
print("   - Fine-tune threshold on validation data")
print("   - Consider calibration (Platt scaling)")
print("\n4. **Accept Current Performance**")
print("   - v21 LB=0.6649 is solid (Rank 23/496)")
print("   - One month to deadline - focus on stability")
print("   - Avoid overfitting leaderboard")

print("\n" + "=" * 80)
