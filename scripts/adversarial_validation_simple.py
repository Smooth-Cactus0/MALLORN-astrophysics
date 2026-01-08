"""
MALLORN: Adversarial Validation (Simplified)
Uses v4 baseline features to check train/test distribution shift
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("=" * 80)
print("MALLORN: Adversarial Validation Analysis")
print("=" * 80)

# Load data
print("\n1. Loading data...")
from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']

print(f"   Train samples: {len(train_meta)}")
print(f"   Test samples: {len(test_meta)}")

# Load v4 baseline features
print("\n2. Loading v4 baseline features...")
cached = pd.read_pickle(Path(__file__).parent.parent / 'data/processed/features_v4_cache.pkl')
train_features = cached['train_features']
test_features = cached['test_features']

# Get feature columns (exclude object_id)
feature_cols = [c for c in train_features.columns if c != 'object_id']
print(f"   Features: {len(feature_cols)}")

X_train = train_features[feature_cols].values
X_test = test_features[feature_cols].values

print(f"   Train shape: {X_train.shape}")
print(f"   Test shape: {X_test.shape}")

# Create adversarial dataset
print("\n3. Creating adversarial validation dataset...")
y_train_adv = np.zeros(len(X_train))
y_test_adv = np.ones(len(X_test))

X_combined = np.vstack([X_train, X_test])
y_combined = np.concatenate([y_train_adv, y_test_adv])

print(f"   Combined: {X_combined.shape}")
print(f"   Train (label=0): {len(y_train_adv)}")
print(f"   Test (label=1): {len(y_test_adv)}")

# Train adversarial classifier
print("\n4. Training adversarial classifier...")

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_combined))
feature_importance = np.zeros(len(feature_cols))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined), 1):
    X_tr, X_val = X_combined[train_idx], X_combined[val_idx]
    y_tr, y_val = y_combined[train_idx], y_combined[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=100,
        evals=[(dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    oof_preds[val_idx] = model.predict(dval)

    importance = model.get_score(importance_type='gain')
    for feat, gain in importance.items():
        if feat in feature_cols:
            idx = feature_cols.index(feat)
            feature_importance[idx] += gain

    print(f"   Fold {fold} complete")

auc = roc_auc_score(y_combined, oof_preds)

print(f"\n{'='*80}")
print(f"ADVERSARIAL AUC: {auc:.4f}")
print(f"{'='*80}")

# Interpret results
print("\n5. Interpretation:")

if auc < 0.52:
    print("   EXCELLENT: AUC < 0.52 - Train and test are nearly identical!")
    print("   No distribution shift detected.")
    print("   OOF-LB gaps likely due to random variance or overfitting.")
    distribution_shift = False
elif auc < 0.55:
    print("   GOOD: AUC < 0.55 - Minimal distribution shift")
    print("   Train and test are very similar.")
    print("   Adversarial reweighting unlikely to help much.")
    distribution_shift = False
elif auc < 0.60:
    print("   MODERATE: AUC = 0.55-0.60 - Some distribution shift detected")
    print("   Consider adversarial reweighting in training.")
    distribution_shift = True
else:
    print("   SIGNIFICANT: AUC > 0.60 - Strong distribution shift!")
    print("   Train and test have different distributions.")
    print("   Adversarial reweighting STRONGLY recommended.")
    distribution_shift = True

# Top discriminative features
print("\n6. Top 20 Features Distinguishing Train vs Test:")

feature_importance = feature_importance / 5  # Average over folds
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(20).to_string(index=False))

# Analyze top 5 features
print("\n7. Feature Distribution Differences:")

top_features = importance_df.head(5)['feature'].tolist()

for feat in top_features:
    train_vals = train_features[feat].values
    test_vals = test_features[feat].values

    # Remove NaN
    train_vals = train_vals[~np.isnan(train_vals)]
    test_vals = test_vals[~np.isnan(test_vals)]

    if len(train_vals) > 0 and len(test_vals) > 0:
        train_mean = np.mean(train_vals)
        test_mean = np.mean(test_vals)
        train_std = np.std(train_vals)
        test_std = np.std(test_vals)

        diff_pct = 100 * (test_mean - train_mean) / (train_mean + 1e-9)

        print(f"\n   {feat}:")
        print(f"      Train: mean={train_mean:.3f}, std={train_std:.3f}")
        print(f"      Test:  mean={test_mean:.3f}, std={test_std:.3f}")
        print(f"      Difference: {diff_pct:+.1f}%")

# Create sample weights
print("\n8. Creating sample weights for training...")

train_adv_preds = oof_preds[:len(X_train)]

if distribution_shift:
    # Map [0, 1] to [0.5, 2.0] weights
    # test-like (p=1) → weight=2.0
    # train-like (p=0) → weight=0.5
    sample_weights = 0.5 + 1.5 * train_adv_preds

    print(f"   Weight range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")
    print(f"   Mean weight: {sample_weights.mean():.2f}")
    print(f"   Median weight: {np.median(sample_weights):.2f}")

    q25 = np.percentile(sample_weights, 25)
    q75 = np.percentile(sample_weights, 75)
    print(f"   25th percentile: {q25:.2f}")
    print(f"   75th percentile: {q75:.2f}")
else:
    # No distribution shift - use uniform weights
    sample_weights = np.ones(len(X_train))
    print(f"   No distribution shift - using uniform weights")

# Save results
print("\n9. Saving results...")

results = {
    'auc': auc,
    'distribution_shift': distribution_shift,
    'feature_importance': importance_df,
    'sample_weights': sample_weights,
    'train_adv_preds': train_adv_preds,
    'top_discriminative_features': top_features
}

base_path = Path(__file__).parent.parent
with open(base_path / 'data/processed/adversarial_validation.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"   Results saved: adversarial_validation.pkl")

# Final recommendations
print(f"\n{'='*80}")
print("ADVERSARIAL VALIDATION COMPLETE")
print(f"{'='*80}")
print(f"AUC: {auc:.4f}")
print(f"Distribution Shift: {'YES' if distribution_shift else 'NO'}")

if distribution_shift:
    print("\nRECOMMENDATIONS:")
    print("1. Use sample_weights in next training (v39a)")
    print("2. Up-weight test-like samples (weight > 1.0)")
    print("3. Down-weight train-like samples (weight < 1.0)")
    print("4. Expected gain: +1-2% from better generalization")
    print("\nNEXT STEP: Train v39a with adversarial weights")
else:
    print("\nRECOMMENDATIONS:")
    print("1. No adversarial reweighting needed")
    print("2. Train/test distributions are similar")
    print("3. OOF-LB gaps are due to other factors:")
    print("   - Random variance in test set")
    print("   - Overfitting to specific training patterns")
    print("   - Threshold optimization on OOF data")
    print("\nNEXT STEP: Move to other techniques (Focal Loss, Fourier, etc.)")

print(f"{'='*80}")
