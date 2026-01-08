"""
Compare all MALLORN experiments and generate comprehensive report.

Compares:
- v21: Baseline XGBoost
- v31: CatBoost with ordered boosting
- v32: XGBoost with feature interactions
- v33: Diverse ensemble (XGB + LGB + CAT)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN EXPERIMENT COMPARISON")
print("=" * 80)

# Load all artifacts
experiments = {}

for version, name in [
    ('v21', 'Baseline (XGBoost)'),
    ('v31', 'CatBoost Ordered'),
    ('v32', 'Feature Interactions'),
    ('v33', 'Diverse Ensemble')
]:
    artifact_path = base_path / f'data/processed/{version}_artifacts.pkl'
    if artifact_path.exists():
        with open(artifact_path, 'rb') as f:
            experiments[version] = {
                'name': name,
                'data': pickle.load(f)
            }
        print(f"Loaded {version}: {name}")
    else:
        print(f"Missing {version}: {name}")

print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

# Create comparison table
results = []

for version in ['v21', 'v31', 'v32', 'v33']:
    if version not in experiments:
        continue

    data = experiments[version]['data']
    name = experiments[version]['name']

    oof_preds = data['oof_preds']
    threshold = data['best_threshold']
    oof_f1 = data['oof_f1']

    # Load y
    from utils.data_loader import load_all_data
    train_meta = load_all_data()['train_meta']
    y = train_meta['target'].values

    final_preds = (oof_preds > threshold).astype(int)
    precision = precision_score(y, final_preds)
    recall = recall_score(y, final_preds)

    # Count TDEs predicted on test
    test_preds = data['test_preds']
    test_final = (test_preds > threshold).astype(int)
    n_tdes = test_final.sum() if hasattr(test_final, 'sum') else 0

    results.append({
        'Version': version,
        'Name': name,
        'OOF F1': oof_f1,
        'Precision': precision,
        'Recall': recall,
        'Threshold': threshold,
        'Test TDEs': n_tdes
    })

results_df = pd.DataFrame(results)

# Sort by OOF F1
results_df = results_df.sort_values('OOF F1', ascending=False)

print("\n" + results_df.to_string(index=False))

# Compute improvements vs v21
if 'v21' in experiments:
    baseline_f1 = experiments['v21']['data']['oof_f1']
    print(f"\n\nIMPROVEMENT vs v21 BASELINE (F1={baseline_f1:.4f}):")
    print("-" * 80)

    for _, row in results_df.iterrows():
        if row['Version'] != 'v21':
            improvement = row['OOF F1'] - baseline_f1
            improvement_pct = 100 * improvement / baseline_f1
            status = "SUCCESS" if improvement > 0 else "NO IMPROVEMENT"
            print(f"{row['Version']} ({row['Name']:25s}): {improvement:+.4f} ({improvement_pct:+.2f}%) - {status}")

# Best model
print("\n" + "=" * 80)
best_version = results_df.iloc[0]['Version']
best_name = results_df.iloc[0]['Name']
best_f1 = results_df.iloc[0]['OOF F1']
print(f"BEST MODEL: {best_version} ({best_name}) with OOF F1 = {best_f1:.4f}")
print("=" * 80)

# Feature importance comparison (if available)
print("\n\nTOP FEATURES COMPARISON:")
print("-" * 80)

for version in ['v21', 'v31', 'v32', 'v33']:
    if version not in experiments:
        continue

    data = experiments[version]['data']
    if 'feature_importance' in data:
        importance_df = data['feature_importance']
        print(f"\n{version} - Top 10 Features:")
        print(importance_df.head(10)[['feature', 'importance']].to_string(index=False))

print("\n\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if best_version == 'v21':
    print("v21 baseline remains the best model.")
    print("Advanced techniques (CatBoost, interactions, ensemble) did not improve performance.")
    print("\nPossible reasons:")
    print("- v21's feature set already captures the essential physics")
    print("- Small dataset size (~3000 samples) limits model complexity benefits")
    print("- High class imbalance makes improvements difficult")
else:
    print(f"SUCCESS: {best_version} ({best_name}) improved upon v21 baseline!")
    print(f"Gain: {results_df.iloc[0]['OOF F1'] - baseline_f1:+.4f} F1 ({100*(results_df.iloc[0]['OOF F1'] - baseline_f1)/baseline_f1:+.2f}%)")
    print(f"\nRecommendation: Submit {best_version} to leaderboard")

print("=" * 80)
