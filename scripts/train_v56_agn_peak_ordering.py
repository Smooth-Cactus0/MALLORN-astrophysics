"""
MALLORN v56: XGBoost + AGN Probability + Peak Ordering Features

New physics-informed features:
1. AGN probability (from AGN vs Rest classifier)
2. Peak ordering features:
   - g_to_r_peak_delay (statistically significant! p=0.0011)
   - is_blue_first (binary)
   - g_peaks_last (binary, TDE indicator)
   - first_peak_band (one-hot encoded)

v55 baseline: OOF F1=0.6751, LB F1=0.6873
Target: Close gap to 0.73 leaderboard top
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.signal import find_peaks
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v56: AGN Probability + Peak Ordering Features", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_lc = data['train_lc']
test_lc = data['test_lc']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y_train = train_meta['target'].values

# Create AGN labels for training
train_meta['is_agn'] = (train_meta['SpecType'] == 'AGN').astype(int)
y_agn = train_meta['is_agn'].values

print(f"   Training objects: {len(train_ids)}", flush=True)
print(f"   Test objects: {len(test_ids)}", flush=True)
print(f"   TDEs: {y_train.sum()} ({100*y_train.mean():.1f}%)", flush=True)
print(f"   AGN: {y_agn.sum()} ({100*y_agn.mean():.1f}%)", flush=True)

# ====================
# 2. EXTRACT AGN PROBABILITY
# ====================
print("\n2. Training AGN classifier and extracting probabilities...", flush=True)

# Load base features for AGN classifier
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']

feature_cols_base = [c for c in train_base.columns if c != 'object_id']
X_base_train = train_base[feature_cols_base].values
X_base_test = test_base[feature_cols_base].values

# Train AGN classifier with CV to get OOF predictions
xgb_agn_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

agn_proba_train = np.zeros(len(X_base_train))
agn_proba_test = np.zeros((len(X_base_test), n_folds))

print("   Training AGN classifier...", flush=True)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_base_train, y_agn), 1):
    X_tr, X_val = X_base_train[train_idx], X_base_train[val_idx]
    y_tr, y_val = y_agn[train_idx], y_agn[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_base_test)

    model = xgb.train(
        xgb_agn_params,
        dtrain,
        num_boost_round=300,
        evals=[(dval, 'val')],
        early_stopping_rounds=30,
        verbose_eval=False
    )

    agn_proba_train[val_idx] = model.predict(dval)
    agn_proba_test[:, fold-1] = model.predict(dtest)

agn_proba_test_avg = agn_proba_test.mean(axis=1)

# Evaluate AGN classifier
from sklearn.metrics import accuracy_score
agn_acc = accuracy_score(y_agn, (agn_proba_train > 0.5).astype(int))
print(f"   AGN classifier OOF accuracy: {agn_acc:.4f} ({100*agn_acc:.1f}%)", flush=True)

# ====================
# 3. EXTRACT PEAK ORDERING FEATURES
# ====================
print("\n3. Extracting peak ordering features...", flush=True)

BANDS = ['u', 'g', 'r', 'i', 'z', 'y']

def extract_peak_ordering_features(obj_id, lc_data):
    """Extract peak ordering features for a single object."""
    features = {'object_id': obj_id}

    # Get peak time for each band
    peak_times = {}
    peak_fluxes = {}

    for band in BANDS:
        band_lc = lc_data[(lc_data['object_id'] == obj_id) & (lc_data['Filter'] == band)]

        if len(band_lc) < 3:
            peak_times[band] = np.nan
            peak_fluxes[band] = np.nan
            continue

        band_lc = band_lc.sort_values('Time (MJD)')
        times = band_lc['Time (MJD)'].values
        flux = band_lc['Flux'].values

        # Find peak
        peak_idx = np.argmax(flux)
        peak_times[band] = times[peak_idx]
        peak_fluxes[band] = flux[peak_idx]

    # Feature 1: g_to_r_peak_delay (most significant!)
    if not np.isnan(peak_times.get('g', np.nan)) and not np.isnan(peak_times.get('r', np.nan)):
        features['g_to_r_peak_delay'] = peak_times['r'] - peak_times['g']
    else:
        features['g_to_r_peak_delay'] = np.nan

    # Feature 2: u_to_i_peak_delay
    if not np.isnan(peak_times.get('u', np.nan)) and not np.isnan(peak_times.get('i', np.nan)):
        features['u_to_i_peak_delay'] = peak_times['i'] - peak_times['u']
    else:
        features['u_to_i_peak_delay'] = np.nan

    # Feature 3: blue_to_red_delay (earliest blue - earliest red)
    blue_times = [peak_times[b] for b in ['u', 'g'] if not np.isnan(peak_times.get(b, np.nan))]
    red_times = [peak_times[b] for b in ['z', 'y'] if not np.isnan(peak_times.get(b, np.nan))]

    if blue_times and red_times:
        features['blue_to_red_delay'] = min(red_times) - min(blue_times)
        features['is_blue_first'] = 1 if min(blue_times) < min(red_times) else 0
    else:
        features['blue_to_red_delay'] = np.nan
        features['is_blue_first'] = np.nan

    # Feature 4: Which band peaks first/last
    valid_bands = [(b, t) for b, t in peak_times.items() if not np.isnan(t)]

    if len(valid_bands) >= 2:
        sorted_bands = sorted(valid_bands, key=lambda x: x[1])
        first_band = sorted_bands[0][0]
        last_band = sorted_bands[-1][0]

        # One-hot for first peak band
        for band in BANDS:
            features[f'first_peak_{band}'] = 1 if first_band == band else 0

        # g_peaks_last indicator (TDE signal)
        features['g_peaks_last'] = 1 if last_band == 'g' else 0

        # Peak time spread (last - first)
        features['peak_time_spread'] = sorted_bands[-1][1] - sorted_bands[0][1]
    else:
        for band in BANDS:
            features[f'first_peak_{band}'] = np.nan
        features['g_peaks_last'] = np.nan
        features['peak_time_spread'] = np.nan

    # Feature 5: Peak flux ratios
    if not np.isnan(peak_fluxes.get('g', np.nan)) and not np.isnan(peak_fluxes.get('r', np.nan)):
        if peak_fluxes['r'] != 0:
            features['peak_flux_g_over_r'] = peak_fluxes['g'] / peak_fluxes['r']
        else:
            features['peak_flux_g_over_r'] = np.nan
    else:
        features['peak_flux_g_over_r'] = np.nan

    return features

# Extract for training set
print("   Extracting for training set...", flush=True)
train_peak_features = []
for i, obj_id in enumerate(train_ids):
    if (i + 1) % 500 == 0:
        print(f"    Progress: {i+1}/{len(train_ids)}", flush=True)
    feat = extract_peak_ordering_features(obj_id, train_lc)
    train_peak_features.append(feat)

train_peak_df = pd.DataFrame(train_peak_features)

# Extract for test set
print("   Extracting for test set...", flush=True)
test_peak_features = []
for i, obj_id in enumerate(test_ids):
    if (i + 1) % 1000 == 0:
        print(f"    Progress: {i+1}/{len(test_ids)}", flush=True)
    feat = extract_peak_ordering_features(obj_id, test_lc)
    test_peak_features.append(feat)

test_peak_df = pd.DataFrame(test_peak_features)

peak_feature_cols = [c for c in train_peak_df.columns if c != 'object_id']
print(f"   Peak ordering features: {len(peak_feature_cols)}", flush=True)

# ====================
# 4. COMBINE ALL FEATURES
# ====================
print("\n4. Combining all features...", flush=True)

# Load v55 features (which include v34a + power law)
selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
importance_df = selection['importance_df']
high_corr_df = selection['high_corr_df']

corr_to_drop = set()
for _, row in high_corr_df.iterrows():
    if row['feature_1'] not in corr_to_drop:
        corr_to_drop.add(row['feature_2'])

clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
selected_120 = clean_features.head(120)['feature'].tolist()

# Load additional feature sets
tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']

with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

bazin_cache = pd.read_pickle(base_path / 'data/processed/bazin_features_cache.pkl')
train_bazin = bazin_cache['train']
test_bazin = bazin_cache['test']

# Build combined training features
train_combined = train_base[['object_id'] + selected_120].copy()
train_combined = train_combined.merge(train_tde, on='object_id', how='left')
train_combined = train_combined.merge(train_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')
train_combined = train_combined.merge(train_bazin, on='object_id', how='left')

# Add AGN probability
train_combined['agn_probability'] = agn_proba_train

# Add peak ordering features
train_combined = train_combined.merge(train_peak_df, on='object_id', how='left')

# Build combined test features
test_combined = test_base[['object_id'] + selected_120].copy()
test_combined = test_combined.merge(test_tde, on='object_id', how='left')
test_combined = test_combined.merge(test_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')
test_combined = test_combined.merge(test_bazin, on='object_id', how='left')

# Add AGN probability
test_combined['agn_probability'] = agn_proba_test_avg

# Add peak ordering features
test_combined = test_combined.merge(test_peak_df, on='object_id', how='left')

# Prepare final feature matrices
feature_cols = [c for c in train_combined.columns if c != 'object_id']
X_train = train_combined[feature_cols].values
X_test = test_combined[feature_cols].values

print(f"   Base features: {len(selected_120)}", flush=True)
print(f"   + TDE physics: {len(train_tde.columns) - 1}", flush=True)
print(f"   + GP features: {len(gp2d_cols)}", flush=True)
print(f"   + Bazin features: {len(train_bazin.columns) - 1}", flush=True)
print(f"   + AGN probability: 1", flush=True)
print(f"   + Peak ordering: {len(peak_feature_cols)}", flush=True)
print(f"   = Total features: {len(feature_cols)}", flush=True)
print(f"   Training shape: {X_train.shape}", flush=True)

# ====================
# 5. TRAIN MODEL
# ====================
print("\n5. Training XGBoost with 5-fold CV...", flush=True)

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.025,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros((len(X_test), n_folds))
feature_importance = np.zeros(len(feature_cols))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    oof_preds[val_idx] = model.predict(dval)
    test_preds[:, fold-1] = model.predict(dtest)

    importance = model.get_score(importance_type='gain')
    for feat, gain in importance.items():
        if feat in feature_cols:
            idx = feature_cols.index(feat)
            feature_importance[idx] += gain

    # Find best threshold
    best_f1 = 0
    best_thresh = 0.5
    for t in np.linspace(0.05, 0.5, 50):
        preds_binary = (oof_preds[val_idx] > t).astype(int)
        f1 = f1_score(y_val, preds_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"      Best threshold: {best_thresh:.3f}, F1: {best_f1:.4f}", flush=True)

# ====================
# 6. RESULTS
# ====================
print("\n" + "=" * 80, flush=True)
print("CROSS-VALIDATION RESULTS", flush=True)
print("=" * 80, flush=True)

best_f1 = 0
best_thresh = 0.5
for t in np.linspace(0.05, 0.5, 100):
    preds_binary = (oof_preds > t).astype(int)
    f1 = f1_score(y_train, preds_binary)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}", flush=True)
print(f"   v55 baseline: OOF F1=0.6751, LB F1=0.6873", flush=True)
print(f"   v34a baseline: OOF F1=0.6667, LB F1=0.6907", flush=True)
print(f"   Change vs v55: {100*(best_f1 - 0.6751)/0.6751:+.2f}%", flush=True)

# Confusion matrix
final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y_train == 1))
fp = np.sum((final_preds == 1) & (y_train == 0))
fn = np.sum((final_preds == 0) & (y_train == 1))
tn = np.sum((final_preds == 0) & (y_train == 0))

print(f"\n   Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}", flush=True)
print(f"   Recall: {tp/(tp+fn):.4f}", flush=True)

# ====================
# 7. FEATURE IMPORTANCE ANALYSIS
# ====================
print("\n" + "=" * 80, flush=True)
print("FEATURE IMPORTANCE - New Features", flush=True)
print("=" * 80, flush=True)

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Mark new features
new_feature_names = ['agn_probability', 'g_to_r_peak_delay', 'u_to_i_peak_delay',
                     'blue_to_red_delay', 'is_blue_first', 'g_peaks_last',
                     'peak_time_spread', 'peak_flux_g_over_r'] + [f'first_peak_{b}' for b in BANDS]

importance_df['is_new'] = importance_df['feature'].isin(new_feature_names)

print("\n   New features ranking:", flush=True)
new_features_ranked = importance_df[importance_df['is_new']].head(20)
for i, row in new_features_ranked.iterrows():
    rank = importance_df.index.get_loc(i) + 1
    print(f"   #{rank:3d}: {row['feature']:<25} importance={row['importance']:.1f}", flush=True)

print("\n   Top 20 overall features:", flush=True)
for i, row in importance_df.head(20).iterrows():
    rank = importance_df.index.get_loc(i) + 1
    marker = " [NEW]" if row['is_new'] else ""
    print(f"   #{rank:2d}: {row['feature']:<30} {row['importance']:>10.1f}{marker}", flush=True)

# ====================
# 8. CREATE SUBMISSION
# ====================
print("\n8. Creating submission...", flush=True)

test_avg = test_preds.mean(axis=1)
test_final = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v56_agn_peak.csv'
submission.to_csv(submission_path, index=False)

print(f"   Submission saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_final.sum()} / {len(test_final)}", flush=True)

# Save artifacts
artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'feature_importance': importance_df,
    'agn_proba_train': agn_proba_train,
    'agn_proba_test': agn_proba_test_avg,
    'peak_features_train': train_peak_df,
    'peak_features_test': test_peak_df
}

with open(base_path / 'data/processed/v56_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v56 Complete: OOF F1 = {best_f1:.4f}", flush=True)
print("=" * 80, flush=True)
print("\nNew features added:", flush=True)
print("   - agn_probability (from AGN vs Rest classifier)", flush=True)
print("   - g_to_r_peak_delay (statistically significant!)", flush=True)
print("   - Peak ordering features (is_blue_first, g_peaks_last, etc.)", flush=True)
print("=" * 80, flush=True)
