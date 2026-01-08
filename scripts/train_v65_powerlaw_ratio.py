"""
MALLORN v65: Power-Law Decay and MaxVar Features

Since R_bb features had coverage issues (7-34%), let's try features
that don't require synchronized multi-band observations:

1. Power-law decay exponent (TDEs: α ≈ -5/3 = -1.67)
   - Already have r_decay_alpha, but let's add refined versions
   - Multi-band decay comparison

2. MaxVar = (max - median) / MAD
   - Simple variability metric
   - AGN vs transient discrimination

3. Late-time flux ratios
   - TDEs fade more slowly than SNe
   - Compare flux at different epochs

4. Peak-to-baseline ratio
   - TDEs typically have larger amplitude events
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v65: Power-Law Decay + MaxVar Features", flush=True)
print("=" * 80, flush=True)


def extract_powerlaw_features(lightcurves: pd.DataFrame, object_ids: list) -> pd.DataFrame:
    """Extract power-law decay and variability features."""

    grouped = {oid: group for oid, group in lightcurves.groupby('object_id')}
    all_features = []

    for i, oid in enumerate(object_ids):
        if (i + 1) % 500 == 0:
            print(f"   Processing {i+1}/{len(object_ids)}...", flush=True)

        obj_lc = grouped.get(oid, pd.DataFrame())
        features = {'object_id': oid}

        if obj_lc.empty:
            all_features.append(features)
            continue

        # Per-band features
        for band in ['g', 'r', 'i']:
            band_lc = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

            if len(band_lc) < 5:
                features[f'{band}_maxvar'] = np.nan
                features[f'{band}_power_exponent'] = np.nan
                features[f'{band}_late_frac'] = np.nan
                continue

            flux = band_lc['Flux'].values
            times = band_lc['Time (MJD)'].values

            # MaxVar = (max - median) / MAD
            max_flux = np.max(flux)
            median_flux = np.median(flux)
            mad = median_abs_deviation(flux)
            if mad > 0:
                features[f'{band}_maxvar'] = (max_flux - median_flux) / mad
            else:
                features[f'{band}_maxvar'] = np.nan

            # Peak-to-baseline ratio
            baseline = np.percentile(flux, 10)  # Bottom 10% as baseline
            if baseline > 0:
                features[f'{band}_peak_baseline_ratio'] = max_flux / baseline
            else:
                features[f'{band}_peak_baseline_ratio'] = np.nan

            # Power-law decay fit (post-peak)
            peak_idx = np.argmax(flux)
            peak_time = times[peak_idx]
            peak_flux = flux[peak_idx]

            # Get post-peak data
            post_mask = times > peak_time + 5  # At least 5 days after peak
            if np.sum(post_mask) >= 3:
                post_times = times[post_mask] - peak_time
                post_flux = flux[post_mask]

                # Fit: F(t) = A * t^alpha
                # Log: log(F) = log(A) + alpha * log(t)
                valid = (post_flux > 0) & (post_times > 0)
                if np.sum(valid) >= 3:
                    try:
                        log_t = np.log10(post_times[valid])
                        log_f = np.log10(post_flux[valid])
                        coeffs = np.polyfit(log_t, log_f, 1)
                        features[f'{band}_power_exponent'] = coeffs[0]  # TDE ≈ -1.67
                    except:
                        features[f'{band}_power_exponent'] = np.nan
                else:
                    features[f'{band}_power_exponent'] = np.nan
            else:
                features[f'{band}_power_exponent'] = np.nan

            # Late-time flux fraction (flux at late times / peak)
            late_mask = times > peak_time + 50
            if np.sum(late_mask) > 0 and peak_flux > 0:
                late_flux = np.mean(flux[late_mask])
                features[f'{band}_late_frac'] = late_flux / peak_flux
            else:
                features[f'{band}_late_frac'] = np.nan

            # Very late fraction (100+ days)
            very_late_mask = times > peak_time + 100
            if np.sum(very_late_mask) > 0 and peak_flux > 0:
                very_late_flux = np.mean(flux[very_late_mask])
                features[f'{band}_very_late_frac'] = very_late_flux / peak_flux
            else:
                features[f'{band}_very_late_frac'] = np.nan

        # Cross-band features
        g_exp = features.get('g_power_exponent', np.nan)
        r_exp = features.get('r_power_exponent', np.nan)
        i_exp = features.get('i_power_exponent', np.nan)

        exponents = [e for e in [g_exp, r_exp, i_exp] if not np.isnan(e)]
        if len(exponents) >= 2:
            features['power_exp_std'] = np.std(exponents)  # Consistent across bands?
            features['power_exp_mean'] = np.mean(exponents)
            features['power_exp_min'] = np.min(exponents)
        else:
            features['power_exp_std'] = np.nan
            features['power_exp_mean'] = np.nan
            features['power_exp_min'] = np.nan

        # TDE indicator: exponent close to -5/3 = -1.67
        if not np.isnan(r_exp):
            features['tde_decay_score'] = -np.abs(r_exp + 1.67)  # Higher = closer to TDE
        else:
            features['tde_decay_score'] = np.nan

        # MaxVar comparison
        g_mv = features.get('g_maxvar', np.nan)
        r_mv = features.get('r_maxvar', np.nan)

        maxvars = [m for m in [g_mv, r_mv] if not np.isnan(m)]
        if len(maxvars) >= 1:
            features['maxvar_mean'] = np.mean(maxvars)
            features['maxvar_max'] = np.max(maxvars)
        else:
            features['maxvar_mean'] = np.nan
            features['maxvar_max'] = np.nan

        all_features.append(features)

    return pd.DataFrame(all_features)


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
y = train_meta['target'].values

print(f"   Training: {len(train_ids)} objects ({np.sum(y)} TDEs)", flush=True)

# Load v34a features
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

tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']

train_base = train_base.merge(train_tde, on='object_id', how='left')
test_base = test_base.merge(test_tde, on='object_id', how='left')

with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

train_v21 = train_base[['object_id'] + selected_120].copy()
train_v21 = train_v21.merge(train_tde, on='object_id', how='left')
train_v21 = train_v21.merge(train_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

test_v21 = test_base[['object_id'] + selected_120].copy()
test_v21 = test_v21.merge(test_tde, on='object_id', how='left')
test_v21 = test_v21.merge(test_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

# Bazin
bazin_cache_path = base_path / 'data/processed/bazin_features_cache.pkl'
with open(bazin_cache_path, 'rb') as f:
    bazin_cache = pickle.load(f)
train_bazin = bazin_cache['train']
test_bazin = bazin_cache['test']

train_v34a = train_v21.merge(train_bazin, on='object_id', how='left')
test_v34a = test_v21.merge(test_bazin, on='object_id', how='left')

print(f"   v34a features: {len(train_v34a.columns)-1}", flush=True)

# ====================
# 2. EXTRACT NEW FEATURES
# ====================
print("\n2. Extracting power-law + MaxVar features...", flush=True)

# Check cache
pl_cache_path = base_path / 'data/processed/powerlaw_features_cache.pkl'
if pl_cache_path.exists():
    print("   Loading cached features...", flush=True)
    with open(pl_cache_path, 'rb') as f:
        pl_cache = pickle.load(f)
    train_pl = pl_cache['train']
    test_pl = pl_cache['test']
else:
    print("   Training set...", flush=True)
    train_pl = extract_powerlaw_features(train_lc, train_ids)
    print("   Test set...", flush=True)
    test_pl = extract_powerlaw_features(test_lc, test_ids)

    with open(pl_cache_path, 'wb') as f:
        pickle.dump({'train': train_pl, 'test': test_pl}, f)

pl_cols = [c for c in train_pl.columns if c != 'object_id']
print(f"   Extracted {len(pl_cols)} features", flush=True)

# Coverage
for col in ['r_power_exponent', 'r_maxvar', 'r_late_frac', 'tde_decay_score']:
    if col in train_pl.columns:
        cov = train_pl[col].notna().sum() / len(train_pl)
        print(f"      {col}: {100*cov:.1f}%", flush=True)

# ====================
# 3. COMBINE
# ====================
print("\n3. Combining features...", flush=True)

train_combined = train_v34a.merge(train_pl, on='object_id', how='left')
test_combined = test_v34a.merge(test_pl, on='object_id', how='left')

feature_names = [c for c in train_combined.columns if c != 'object_id']
print(f"   Total: {len(feature_names)} features", flush=True)

X_train = train_combined.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values

X_train = np.nan_to_num(X_train, nan=np.nan, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=np.nan, posinf=1e10, neginf=-1e10)

# ====================
# 4. TRAIN
# ====================
print("\n4. Training XGBoost...", flush=True)

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
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros((len(X_test), n_folds))
feature_importance = np.zeros(len(feature_names))
fold_f1s = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)

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
        if feat in feature_names:
            idx = feature_names.index(feat)
            feature_importance[idx] += gain

    best_f1 = 0
    for t in np.linspace(0.03, 0.3, 50):
        preds_binary = (oof_preds[val_idx] > t).astype(int)
        f1 = f1_score(y_val, preds_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    fold_f1s.append(best_f1)
    print(f"      Fold F1: {best_f1:.4f}", flush=True)

# ====================
# 5. RESULTS
# ====================
print("\n" + "=" * 80, flush=True)
print("RESULTS", flush=True)
print("=" * 80, flush=True)

best_f1 = 0
best_thresh = 0.1
for t in np.linspace(0.03, 0.3, 200):
    preds_binary = (oof_preds > t).astype(int)
    f1 = f1_score(y, preds_binary)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}", flush=True)
print(f"   Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}", flush=True)

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
print(f"   TP={tp}, FP={fp}, FN={fn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}", flush=True)

# Feature importance
feature_importance = feature_importance / n_folds
importance_df_result = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n   Top 25 Features:", flush=True)
print(importance_df_result.head(25).to_string(index=False), flush=True)

# New features analysis
print("\n   New Feature Rankings:", flush=True)
for col in pl_cols:
    if col in importance_df_result['feature'].values:
        rank = list(importance_df_result['feature']).index(col) + 1
        imp = importance_df_result[importance_df_result['feature'] == col]['importance'].values[0]
        if imp > 0:
            print(f"      {rank:3d}. {col:30s} {imp:8.1f}", flush=True)

# ====================
# 6. SUBMISSION
# ====================
print("\n" + "=" * 80, flush=True)
print("SUBMISSION", flush=True)
print("=" * 80, flush=True)

test_avg = test_preds.mean(axis=1)
test_final = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v65_powerlaw.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_final.sum()}", flush=True)

# Artifacts
artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'feature_importance': importance_df_result,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'fold_f1s': fold_f1s
}

with open(base_path / 'data/processed/v65_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

# Comparison
print("\n" + "=" * 80, flush=True)
print("COMPARISON", flush=True)
print("=" * 80, flush=True)

v34a_f1 = 0.6667
improvement = (best_f1 - v34a_f1) / v34a_f1 * 100

print(f"\n   v34a baseline: {v34a_f1:.4f}", flush=True)
print(f"   v65 (+ power-law/MaxVar): {best_f1:.4f} ({improvement:+.2f}%)", flush=True)

if best_f1 > v34a_f1:
    expected_lb = 0.6907 * (best_f1 / v34a_f1)
    print(f"\n   Expected LB: ~{expected_lb:.4f}", flush=True)
    print("   SUCCESS!", flush=True)

print("\n" + "=" * 80, flush=True)
print("v65 Complete", flush=True)
print("=" * 80, flush=True)
