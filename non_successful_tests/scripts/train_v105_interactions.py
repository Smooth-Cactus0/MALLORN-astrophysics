"""
MALLORN v105: Cross-Feature Interactions

Goal: Capture physics relationships that XGBoost might miss with single features.

Strategy:
1. Identify top features from feature importance
2. Create meaningful interactions (products, ratios)
3. Focus on physics-motivated combinations:
   - Color × temperature interactions
   - Timescale × amplitude relationships
   - Multi-band flux ratios

This aims for a ROBUST model that generalizes well, not LB fitting.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v105: Cross-Feature Interactions")
print("=" * 80)
print("\nGoal: Capture physics relationships for robust generalization")

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
y_train = train_meta['target'].values

print(f"   Training: {len(train_ids)} ({np.sum(y_train)} TDE)")
class_ratio = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

# Load adversarial weights
with open(base_path / 'data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)
sample_weights = adv_results['sample_weights']

# ====================
# 2. LOAD FEATURES
# ====================
print("\n2. Loading features...")

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

train_all = train_base.merge(train_tde, on='object_id', how='left')
train_all = train_all.merge(train_gp2d, on='object_id', how='left')
train_all = train_all.merge(train_bazin, on='object_id', how='left')

test_all = test_base.merge(test_tde, on='object_id', how='left')
test_all = test_all.merge(test_gp2d, on='object_id', how='left')
test_all = test_all.merge(test_bazin, on='object_id', how='left')

shift_features = ['all_rise_time', 'all_asymmetry']
available_features = [f for f in v34a_features if f in train_all.columns and f not in shift_features]

print(f"   Base features: {len(available_features)}")

# ====================
# 3. CREATE INTERACTION FEATURES
# ====================
print("\n3. Creating physics-motivated interaction features...")

def create_interactions(df, feature_list):
    """Create interaction features based on physics understanding."""
    new_features = {}

    # Get column names that exist in the dataframe
    cols = [c for c in feature_list if c in df.columns]

    # 1. Color interactions (g-r, r-i at different phases)
    color_features = [c for c in cols if 'color' in c.lower() or 'g_r' in c or 'r_i' in c]
    for i, c1 in enumerate(color_features[:5]):  # Limit to top 5
        for c2 in color_features[i+1:6]:
            if c1 in df.columns and c2 in df.columns:
                name = f"int_{c1}_x_{c2}"[:50]  # Truncate long names
                new_features[name] = df[c1].values * df[c2].values

    # 2. Flux ratios between bands
    flux_features = [c for c in cols if 'mean_flux' in c or 'peak_flux' in c]
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    for b1, b2 in [('g', 'r'), ('r', 'i'), ('i', 'z'), ('g', 'i')]:
        f1 = [c for c in flux_features if f'_{b1}_' in c or c.endswith(f'_{b1}')]
        f2 = [c for c in flux_features if f'_{b2}_' in c or c.endswith(f'_{b2}')]
        if f1 and f2 and f1[0] in df.columns and f2[0] in df.columns:
            name = f"ratio_{b1}_{b2}_flux"
            denom = df[f2[0]].values
            denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
            new_features[name] = df[f1[0]].values / denom

    # 3. Amplitude × timescale interactions (physics: energy considerations)
    amp_features = [c for c in cols if 'amplitude' in c.lower() or 'amp' in c.lower()]
    time_features = [c for c in cols if 'duration' in c or 'time' in c.lower() or 'decay' in c]
    for af in amp_features[:3]:
        for tf in time_features[:3]:
            if af in df.columns and tf in df.columns:
                name = f"int_{af[:20]}_x_{tf[:20]}"
                new_features[name] = df[af].values * df[tf].values

    # 4. Redshift corrections (physics: k-correction proxy)
    if 'Z' in df.columns:
        z = df['Z'].values
        for f in ['g_mean_flux', 'r_mean_flux', 'g_peak_flux', 'r_peak_flux']:
            if f in df.columns:
                name = f"zcorr_{f}"
                # Approximate k-correction: divide by (1+z)
                new_features[name] = df[f].values / (1 + z + 1e-10)

    # 5. Color evolution rate × baseline color
    evol_features = [c for c in cols if 'slope' in c.lower() or 'evol' in c.lower()]
    for ef in evol_features[:3]:
        for cf in color_features[:3]:
            if ef in df.columns and cf in df.columns:
                name = f"int_{ef[:20]}_x_{cf[:20]}"
                new_features[name] = df[ef].values * df[cf].values

    # 6. GP length scale interactions (variability timescales)
    gp_features = [c for c in cols if 'gp_' in c.lower() or 'length' in c.lower()]
    for i, g1 in enumerate(gp_features[:4]):
        for g2 in gp_features[i+1:5]:
            if g1 in df.columns and g2 in df.columns:
                name = f"int_{g1[:20]}_x_{g2[:20]}"
                new_features[name] = df[g1].values * df[g2].values

    # 7. Temperature × color (physics: blackbody relationship)
    temp_features = [c for c in cols if 'temp' in c.lower() or 'bb' in c.lower()]
    for tf in temp_features[:3]:
        for cf in color_features[:3]:
            if tf in df.columns and cf in df.columns:
                name = f"int_{tf[:20]}_x_{cf[:20]}"
                new_features[name] = df[tf].values * df[cf].values

    return new_features

# Create interactions for train and test
print("   Creating interaction features for training set...")
train_interactions = create_interactions(train_all, available_features)
print(f"   Created {len(train_interactions)} interaction features")

print("   Creating interaction features for test set...")
test_interactions = create_interactions(test_all, available_features)

# Combine base features with interactions
X_train_base = train_all[available_features].values
X_test_base = test_all[available_features].values

X_train_int = np.column_stack([train_interactions[k] for k in train_interactions.keys()])
X_test_int = np.column_stack([test_interactions[k] for k in test_interactions.keys()])

# Handle inf/nan
X_train_base = np.nan_to_num(X_train_base, nan=0.0, posinf=0.0, neginf=0.0)
X_test_base = np.nan_to_num(X_test_base, nan=0.0, posinf=0.0, neginf=0.0)
X_train_int = np.nan_to_num(X_train_int, nan=0.0, posinf=0.0, neginf=0.0)
X_test_int = np.nan_to_num(X_test_int, nan=0.0, posinf=0.0, neginf=0.0)

# Combine
X_train_full = np.hstack([X_train_base, X_train_int])
X_test_full = np.hstack([X_test_base, X_test_int])

all_feature_names = available_features + list(train_interactions.keys())
print(f"   Total features: {len(all_feature_names)} ({len(available_features)} base + {len(train_interactions)} interactions)")

# ====================
# 4. TRAIN MODELS
# ====================
print("\n4. Training models...")

# v92d's best params
base_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.02,
    'n_estimators': 1500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': class_ratio,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
}

results = {}
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# v105a: Base features only (control)
print("\n   v105a: Base features only (control)...")
oof_preds = np.zeros(len(y_train))
test_preds_folds = []
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_base, y_train), 1):
    X_tr, X_val = X_train_base[train_idx], X_train_base[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    w_tr = sample_weights[train_idx]

    model = xgb.XGBClassifier(**base_params)
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds_folds.append(model.predict_proba(X_test_base)[:, 1])

    best_f1 = max([f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
                   for t in np.linspace(0.05, 0.5, 50)])
    fold_f1s.append(best_f1)
    print(f"      Fold {fold} F1: {best_f1:.4f}")

test_preds = np.mean(test_preds_folds, axis=0)
best_f1 = max([f1_score(y_train, (oof_preds > t).astype(int))
               for t in np.linspace(0.05, 0.5, 200)])

results['v105a_base'] = {
    'oof_f1': best_f1,
    'oof_preds': oof_preds,
    'test_preds': test_preds,
}
print(f"      OOF F1: {best_f1:.4f}")

# v105b: With interaction features
print("\n   v105b: With interaction features...")
oof_preds = np.zeros(len(y_train))
test_preds_folds = []
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train), 1):
    X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    w_tr = sample_weights[train_idx]

    model = xgb.XGBClassifier(**base_params)
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds_folds.append(model.predict_proba(X_test_full)[:, 1])

    best_f1 = max([f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
                   for t in np.linspace(0.05, 0.5, 50)])
    fold_f1s.append(best_f1)
    print(f"      Fold {fold} F1: {best_f1:.4f}")

test_preds = np.mean(test_preds_folds, axis=0)

best_f1 = 0
best_thresh = 0.3
for t in np.linspace(0.05, 0.5, 200):
    f1 = f1_score(y_train, (oof_preds > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

preds_binary = (oof_preds > best_thresh).astype(int)
cm = confusion_matrix(y_train, preds_binary)
tn, fp, fn, tp = cm.ravel()

results['v105b_interactions'] = {
    'oof_f1': best_f1,
    'threshold': best_thresh,
    'oof_preds': oof_preds,
    'test_preds': test_preds,
    'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
    'recall': tp / (tp + fn),
    'precision': tp / (tp + fp),
    'feature_names': all_feature_names,
}

print(f"      OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}")
print(f"      Recall: {tp/(tp+fn):.1%} | Precision: {tp/(tp+fp):.1%}")

# ====================
# 5. FEATURE IMPORTANCE ANALYSIS
# ====================
print("\n5. Top interaction features (by importance)...")

# Train one more model to get feature importances
model = xgb.XGBClassifier(**base_params)
model.fit(X_train_full, y_train, sample_weight=sample_weights, verbose=False)

importances = model.feature_importances_
feature_imp = pd.DataFrame({
    'feature': all_feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Show top interaction features
int_features = feature_imp[feature_imp['feature'].str.startswith('int_') |
                           feature_imp['feature'].str.startswith('ratio_') |
                           feature_imp['feature'].str.startswith('zcorr_')]
print("\n   Top 10 interaction features:")
for i, row in int_features.head(10).iterrows():
    print(f"      {row['feature'][:40]:<40}: {row['importance']:.4f}")

# ====================
# 6. COMPARISON
# ====================
print("\n6. Comparison with baseline...")

with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
    v92_results = pickle.load(f)
baseline_f1 = v92_results['v92d_baseline_adv']['oof_f1']

print(f"   v92d baseline:        {baseline_f1:.4f}")
print(f"   v105a (base only):    {results['v105a_base']['oof_f1']:.4f}")
print(f"   v105b (interactions): {results['v105b_interactions']['oof_f1']:.4f}")

diff = results['v105b_interactions']['oof_f1'] - baseline_f1
print(f"\n   Improvement vs baseline: {diff:+.4f}")

# ====================
# 7. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n{'Variant':<25} {'OOF F1':<10} {'Features':<10}")
print("-" * 50)
print(f"{'v92d (LB=0.6986)':<25} {'0.6688':<10} {'221':<10}")
print("-" * 50)
print(f"{'v105a_base':<25} {results['v105a_base']['oof_f1']:<10.4f} {len(available_features):<10}")
print(f"{'v105b_interactions':<25} {results['v105b_interactions']['oof_f1']:<10.4f} {len(all_feature_names):<10}")

# ====================
# 8. SUBMISSION
# ====================
print("\n" + "=" * 80)
print("SUBMISSION")
print("=" * 80)

best_res = results['v105b_interactions']
test_binary = (best_res['test_preds'] > best_res['threshold']).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

filename = "submission_v105b_interactions.csv"
submission.to_csv(base_path / f'submissions/{filename}', index=False)

print(f"   Saved: {filename}")
print(f"   OOF F1: {best_res['oof_f1']:.4f}")
print(f"   TDEs predicted: {test_binary.sum()}")

with open(base_path / 'data/processed/v105_interactions_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v105 Interactions Complete")
print("=" * 80)
