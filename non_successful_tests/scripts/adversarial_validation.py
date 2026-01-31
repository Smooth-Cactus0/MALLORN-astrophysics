"""
MALLORN: Adversarial Validation (Technique #5)

Check if train/test sets have different distributions that could explain
OOF-LB performance gaps.

Approach:
1. Train binary classifier: train (label=0) vs test (label=1)
2. If AUC > 0.55: distribution shift detected
3. Analyze which features differ most
4. Create sample weights for v39a training

Reference: PLAsTiCC 1st place used adversarial validation
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN: Adversarial Validation Analysis", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD v34a FEATURES
# ====================
print("\n1. Loading v34a (Bazin) feature set...", flush=True)

# Load v34a features
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

feature_names = v34a['feature_names']
print(f"   Features: {len(feature_names)}", flush=True)

# Load feature data
from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()

# Reconstruct feature sets from v34a
# Load baseline features
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']

# Load Bazin features
bazin_cached = pd.read_pickle(base_path / 'data/processed/bazin_features_cache.pkl')
train_bazin = bazin_cached['train']
test_bazin = bazin_cached['test']

# Load selection
selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
importance_df = selection['importance_df']
high_corr_df = selection['high_corr_df']

# Remove high correlation features
corr_to_drop = set()
for _, row in high_corr_df.iterrows():
    if row['feature_1'] not in corr_to_drop:
        corr_to_drop.add(row['feature_2'])

clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
selected_120 = clean_features.head(120)['feature'].tolist()

# Merge
train_features = train_base[['object_id'] + selected_120].copy()
train_features = train_features.merge(train_bazin, on='object_id', how='left')

test_features = test_base[['object_id'] + selected_120].copy()
test_features = test_features.merge(test_bazin, on='object_id', how='left')

print(f"   Train: {len(train_features)} samples", flush=True)
print(f"   Test: {len(test_features)} samples", flush=True)

# ====================
# 2. CREATE ADVERSARIAL DATASET
# ====================
print("\n2. Creating adversarial validation dataset...", flush=True)

X_train = train_features.drop(columns=['object_id']).values
X_test = test_features.drop(columns=['object_id']).values

# Update feature_names to match actual columns
feature_names = [c for c in train_features.columns if c != 'object_id']
print(f"   Updated feature count: {len(feature_names)}", flush=True)

# Labels: train=0, test=1
y_train_adv = np.zeros(len(X_train))
y_test_adv = np.ones(len(X_test))

X_combined = np.vstack([X_train, X_test])
y_combined = np.concatenate([y_train_adv, y_test_adv])

print(f"   Combined dataset: {X_combined.shape}", flush=True)
print(f"   Train samples (label=0): {len(y_train_adv)}", flush=True)
print(f"   Test samples (label=1): {len(y_test_adv)}", flush=True)

# ====================
# 3. TRAIN ADVERSARIAL CLASSIFIER
# ====================
print("\n3. Training adversarial classifier (train vs test)...", flush=True)

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
feature_importance = np.zeros(len(feature_names))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined), 1):
    X_tr, X_val = X_combined[train_idx], X_combined[val_idx]
    y_tr, y_val = y_combined[train_idx], y_combined[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

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
        if feat in feature_names:
            idx = feature_names.index(feat)
            feature_importance[idx] += gain

auc = roc_auc_score(y_combined, oof_preds)

print(f"\n   Adversarial AUC: {auc:.4f}", flush=True)

# ====================
# 4. INTERPRET RESULTS
# ====================
print("\n4. Interpretation:", flush=True)

if auc < 0.52:
    print("   EXCELLENT: AUC < 0.52 - Train and test are nearly identical!", flush=True)
    print("   No distribution shift detected.", flush=True)
    print("   OOF-LB gaps likely due to random variance or overfitting.", flush=True)
    distribution_shift = False
elif auc < 0.55:
    print("   GOOD: AUC < 0.55 - Minimal distribution shift", flush=True)
    print("   Train and test are very similar.", flush=True)
    print("   Adversarial reweighting unlikely to help much.", flush=True)
    distribution_shift = False
elif auc < 0.60:
    print("   MODERATE: AUC = 0.55-0.60 - Some distribution shift detected", flush=True)
    print("   Consider adversarial reweighting in training.", flush=True)
    distribution_shift = True
else:
    print("   SIGNIFICANT: AUC > 0.60 - Strong distribution shift!", flush=True)
    print("   Train and test have different distributions.", flush=True)
    print("   Adversarial reweighting STRONGLY recommended.", flush=True)
    distribution_shift = True

# ====================
# 5. FEATURE ANALYSIS
# ====================
print("\n5. Top 20 Features Distinguishing Train vs Test:", flush=True)

feature_importance = feature_importance / 5  # Average over folds
importance_df_result = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df_result.head(20).to_string(index=False), flush=True)

# ====================
# 6. ANALYZE FEATURE DISTRIBUTIONS
# ====================
print("\n6. Feature distribution differences:", flush=True)

# Get top 10 discriminative features
top_features = importance_df_result.head(10)['feature'].tolist()

train_df = train_features[['object_id'] + top_features].copy()
test_df = test_features[['object_id'] + top_features].copy()

for feat in top_features[:5]:  # Show top 5
    train_vals = train_df[feat].values
    test_vals = test_df[feat].values

    # Remove NaN
    train_vals = train_vals[~np.isnan(train_vals)]
    test_vals = test_vals[~np.isnan(test_vals)]

    if len(train_vals) > 0 and len(test_vals) > 0:
        train_mean = np.mean(train_vals)
        test_mean = np.mean(test_vals)
        train_std = np.std(train_vals)
        test_std = np.std(test_vals)

        diff_pct = 100 * (test_mean - train_mean) / (train_mean + 1e-9)

        print(f"\n   {feat}:", flush=True)
        print(f"      Train: mean={train_mean:.3f}, std={train_std:.3f}", flush=True)
        print(f"      Test:  mean={test_mean:.3f}, std={test_std:.3f}", flush=True)
        print(f"      Difference: {diff_pct:+.1f}%", flush=True)

# ====================
# 7. CREATE SAMPLE WEIGHTS
# ====================
print("\n7. Creating sample weights for training...", flush=True)

# Extract train predictions (first len(X_train) samples)
train_adv_preds = oof_preds[:len(X_train)]

# Higher probability = more test-like = higher weight
# Lower probability = more train-like = lower weight
if distribution_shift:
    # Map [0, 1] to [0.5, 2.0] weights
    # test-like (p=1) → weight=2.0
    # train-like (p=0) → weight=0.5
    sample_weights = 0.5 + 1.5 * train_adv_preds

    print(f"   Weight range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]", flush=True)
    print(f"   Mean weight: {sample_weights.mean():.2f}", flush=True)
    print(f"   Median weight: {np.median(sample_weights):.2f}", flush=True)

    # Show weight distribution
    q25 = np.percentile(sample_weights, 25)
    q75 = np.percentile(sample_weights, 75)
    print(f"   25th percentile: {q25:.2f}", flush=True)
    print(f"   75th percentile: {q75:.2f}", flush=True)
else:
    # No distribution shift - use uniform weights
    sample_weights = np.ones(len(X_train))
    print(f"   No distribution shift - using uniform weights", flush=True)

# ====================
# 8. SAVE RESULTS
# ====================
print("\n8. Saving adversarial validation results...", flush=True)

results = {
    'auc': auc,
    'distribution_shift': distribution_shift,
    'feature_importance': importance_df_result,
    'sample_weights': sample_weights,
    'train_adv_preds': train_adv_preds,
    'top_discriminative_features': top_features
}

with open(base_path / 'data/processed/adversarial_validation.pkl', 'wb') as f:
    pickle.dump(results, f)

print(f"   Results saved: adversarial_validation.pkl", flush=True)

# ====================
# 9. RECOMMENDATIONS
# ====================
print("\n" + "=" * 80, flush=True)
print("ADVERSARIAL VALIDATION COMPLETE", flush=True)
print("=" * 80, flush=True)
print(f"AUC: {auc:.4f}", flush=True)
print(f"Distribution Shift: {'YES' if distribution_shift else 'NO'}", flush=True)

if distribution_shift:
    print("\nRECOMMENDATIONS:", flush=True)
    print("1. Use sample_weights in next training (v39a)", flush=True)
    print("2. Up-weight test-like samples (weight > 1.0)", flush=True)
    print("3. Down-weight train-like samples (weight < 1.0)", flush=True)
    print("4. Expected gain: +1-2% from better generalization", flush=True)
    print("\nNEXT STEP: Train v39a with adversarial weights", flush=True)
else:
    print("\nRECOMMENDATIONS:", flush=True)
    print("1. No adversarial reweighting needed", flush=True)
    print("2. Train/test distributions are similar", flush=True)
    print("3. OOF-LB gaps are due to other factors:", flush=True)
    print("   - Random variance in test set", flush=True)
    print("   - Overfitting to specific training patterns", flush=True)
    print("   - Threshold optimization on OOF data", flush=True)
    print("\nNEXT STEP: Move to Technique #6 (Focal Loss) or #8 (Fourier)", flush=True)

print("=" * 80, flush=True)
