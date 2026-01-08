"""
MALLORN v55: XGBoost + Power Law Decay Features

Adding 27 new power law R^2 features to v34a baseline:
- 9 decay models x 3 bands (g, r, i)
- Most discriminative: linear_r2 (TDE=0.52 vs SN=0.30)

v34a baseline: OOF F1=0.6667, LB F1=0.6907
Target: Improve with physics-informed decay features
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v55: XGBoost + Power Law Decay Features", flush=True)
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

print(f"   Training objects: {len(train_ids)}", flush=True)
print(f"   Test objects: {len(test_ids)}", flush=True)
print(f"   TDEs in training: {y_train.sum()} ({100*y_train.mean():.1f}%)", flush=True)

# Load v34a baseline
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)
print(f"   v34a baseline: OOF F1={v34a['oof_f1']:.4f}", flush=True)

# ====================
# 2. LOAD FEATURES
# ====================
print("\n2. Loading features...", flush=True)

# Load v34a feature set (baseline)
cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
train_base = cached['train_features']
test_base = cached['test_features']

selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
importance_df = selection['importance_df']
high_corr_df = selection['high_corr_df']

corr_to_drop = set()
for _, row in high_corr_df.iterrows():
    if row['feature_1'] not in corr_to_drop:
        corr_to_drop.add(row['feature_2'])

clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
selected_120 = clean_features.head(120)['feature'].tolist()

# Load TDE physics features
tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']

# Load GP features
with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

# Load Bazin features
bazin_cache = pd.read_pickle(base_path / 'data/processed/bazin_features_cache.pkl')
train_bazin = bazin_cache['train']
test_bazin = bazin_cache['test']

# Load NEW power law features
print("   Loading power law features...", flush=True)
powerlaw_train = pd.read_pickle(base_path / 'data/processed/powerlaw_features.pkl')
print(f"   Power law features: {len(powerlaw_train.columns) - 1}", flush=True)

# Need to extract power law features for test set
print("   Extracting power law features for test set...", flush=True)

# Import the extraction functions
from scipy.optimize import curve_fit

def powerlaw_5_3(t, A, t0):
    return A * np.power(np.maximum(t - t0, 0.1), -5/3)

def powerlaw_1(t, A, t0):
    return A * np.power(np.maximum(t - t0, 0.1), -1)

def powerlaw_1_5(t, A, t0):
    return A * np.power(np.maximum(t - t0, 0.1), -1.5)

def powerlaw_2(t, A, t0):
    return A * np.power(np.maximum(t - t0, 0.1), -2)

def powerlaw_2_5(t, A, t0):
    return A * np.power(np.maximum(t - t0, 0.1), -2.5)

def powerlaw_3(t, A, t0):
    return A * np.power(np.maximum(t - t0, 0.1), -3)

def powerlaw_0_5(t, A, t0):
    return A * np.power(np.maximum(t - t0, 0.1), -0.5)

def exponential(t, A, tau, t0):
    return A * np.exp(-np.maximum(t - t0, 0) / tau)

def linear(t, A, b, t0):
    return A - b * np.maximum(t - t0, 0)

MODELS = {
    'powerlaw_5_3': (powerlaw_5_3, ['A', 't0']),
    'powerlaw_1': (powerlaw_1, ['A', 't0']),
    'powerlaw_1_5': (powerlaw_1_5, ['A', 't0']),
    'powerlaw_2': (powerlaw_2, ['A', 't0']),
    'powerlaw_2_5': (powerlaw_2_5, ['A', 't0']),
    'powerlaw_3': (powerlaw_3, ['A', 't0']),
    'powerlaw_0_5': (powerlaw_0_5, ['A', 't0']),
    'exponential': (exponential, ['A', 'tau', 't0']),
    'linear': (linear, ['A', 'b', 't0']),
}

def fit_decline_models(obj_id, lc_data, band='r'):
    obj_lc = lc_data[(lc_data['object_id'] == obj_id) & (lc_data['Filter'] == band)]

    if len(obj_lc) < 5:
        return {model: np.nan for model in MODELS}

    obj_lc = obj_lc.sort_values('Time (MJD)')
    t = obj_lc['Time (MJD)'].values
    flux = obj_lc['Flux'].values

    peak_idx = np.argmax(flux)
    peak_time = t[peak_idx]
    peak_flux = flux[peak_idx]

    post_peak_mask = t > peak_time
    if np.sum(post_peak_mask) < 3:
        return {model: np.nan for model in MODELS}

    t_post = t[post_peak_mask] - peak_time
    flux_post = flux[post_peak_mask]

    results = {}

    for model_name, (model_func, params) in MODELS.items():
        try:
            if len(params) == 2:
                popt, _ = curve_fit(model_func, t_post, flux_post,
                                   p0=[peak_flux, 0], maxfev=1000,
                                   bounds=([0, -10], [1e6, 10]))
            elif len(params) == 3:
                if 'tau' in params:
                    popt, _ = curve_fit(model_func, t_post, flux_post,
                                       p0=[peak_flux, 30, 0], maxfev=1000,
                                       bounds=([0, 1, -10], [1e6, 500, 10]))
                else:
                    popt, _ = curve_fit(model_func, t_post, flux_post,
                                       p0=[peak_flux, 1, 0], maxfev=1000,
                                       bounds=([0, 0, -10], [1e6, 100, 10]))

            pred = model_func(t_post, *popt)
            ss_res = np.sum((flux_post - pred) ** 2)
            ss_tot = np.sum((flux_post - np.mean(flux_post)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            results[model_name] = r2
        except Exception:
            results[model_name] = np.nan

    return results

def extract_powerlaw_features(obj_id, lc_data):
    features = {'object_id': obj_id}
    for band in ['g', 'r', 'i']:
        results = fit_decline_models(obj_id, lc_data, band=band)
        for model_name, r2 in results.items():
            features[f'{band}_{model_name}_r2'] = r2
    return features

# Extract for test set
test_powerlaw_features = []
for i, obj_id in enumerate(test_ids):
    if (i + 1) % 1000 == 0:
        print(f"    Progress: {i+1}/{len(test_ids)}", flush=True)
    feat = extract_powerlaw_features(obj_id, test_lc)
    test_powerlaw_features.append(feat)

powerlaw_test = pd.DataFrame(test_powerlaw_features)
print(f"   Test power law features extracted: {len(powerlaw_test)}", flush=True)

# ====================
# 3. COMBINE FEATURES
# ====================
print("\n3. Combining all features...", flush=True)

# Build training features
train_v34a = train_base[['object_id'] + selected_120].copy()
train_v34a = train_v34a.merge(train_tde, on='object_id', how='left')
train_v34a = train_v34a.merge(train_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')
train_v34a = train_v34a.merge(train_bazin, on='object_id', how='left')

# Add power law features
powerlaw_cols = [c for c in powerlaw_train.columns if c != 'object_id']
train_combined = train_v34a.merge(powerlaw_train, on='object_id', how='left')

print(f"   v34a features: {len(train_v34a.columns) - 1}", flush=True)
print(f"   + Power law features: {len(powerlaw_cols)}", flush=True)
print(f"   = Total features: {len(train_combined.columns) - 1}", flush=True)

# Build test features
test_v34a = test_base[['object_id'] + selected_120].copy()
test_v34a = test_v34a.merge(test_tde, on='object_id', how='left')
test_v34a = test_v34a.merge(test_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')
test_v34a = test_v34a.merge(test_bazin, on='object_id', how='left')
test_combined = test_v34a.merge(powerlaw_test, on='object_id', how='left')

# Ensure same columns
feature_cols = [c for c in train_combined.columns if c != 'object_id']
X_train = train_combined[feature_cols].values
X_test = test_combined[feature_cols].values

print(f"   Training shape: {X_train.shape}", flush=True)
print(f"   Test shape: {X_test.shape}", flush=True)

# ====================
# 4. TRAIN MODEL
# ====================
print("\n4. Training XGBoost with 5-fold CV...", flush=True)

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

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

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

    # Find best threshold for this fold
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
# 5. RESULTS
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
print(f"   v34a baseline: OOF F1={v34a['oof_f1']:.4f}", flush=True)
print(f"   Change: {100*(best_f1 - v34a['oof_f1'])/v34a['oof_f1']:+.2f}%", flush=True)

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
# 6. FEATURE IMPORTANCE
# ====================
print("\n" + "=" * 80, flush=True)
print("TOP 20 FEATURES (by importance)", flush=True)
print("=" * 80, flush=True)

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Mark power law features
importance_df['is_powerlaw'] = importance_df['feature'].str.contains('_r2')

print("\n   Top 20 features:", flush=True)
for i, row in importance_df.head(20).iterrows():
    marker = " [POWERLAW]" if row['is_powerlaw'] else ""
    print(f"   {importance_df.index.get_loc(i)+1:2d}. {row['feature'][:40]:<40} {row['importance']:>10.1f}{marker}", flush=True)

# Count power law features in top 50
top50_powerlaw = importance_df.head(50)['is_powerlaw'].sum()
print(f"\n   Power law features in top 50: {top50_powerlaw}/50", flush=True)

# ====================
# 7. CREATE SUBMISSION
# ====================
print("\n7. Creating submission...", flush=True)

test_avg = test_preds.mean(axis=1)
test_final = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v55_powerlaw.csv'
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
    'n_powerlaw_features': len(powerlaw_cols)
}

with open(base_path / 'data/processed/v55_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v55 Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"v34a baseline: OOF F1 = {v34a['oof_f1']:.4f}, LB F1 = 0.6907", flush=True)
print(f"New features added: {len(powerlaw_cols)} power law R^2 values", flush=True)
print("=" * 80, flush=True)
