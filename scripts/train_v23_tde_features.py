"""
MALLORN v23: TDE-Specific Features

Based on 2025 Rubin TDE paper (arxiv.org/abs/2509.25902):
- Post-peak (r-i) color slopes (MISSING from v19-v21)
- Power-law decay index (t^-5/3 for TDE vs exponential for SNe)
- Late-time colors (150d after peak)
- All existing GP2D + color features

Target: Beat v21's 0.6708 OOF / 0.6649 LB
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
from scipy.optimize import curve_fit

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 60, flush=True)
print("MALLORN v23: TDE-Specific Feature Engineering", flush=True)
print("=" * 60, flush=True)

# ====================
# 1. LOAD EXISTING DATA & FEATURES
# ====================
print("\n1. Loading data and features...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_lc = data['train_lc']
test_lc = data['test_lc']
train_meta = data['train_meta']
test_meta = data['test_meta']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

# Load all cached features from v21
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

# TDE physics features
tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']
tde_cols = [c for c in train_tde.columns if c != 'object_id']

train_base = train_base.merge(train_tde, on='object_id', how='left')
test_base = test_base.merge(test_tde, on='object_id', how='left')

# GP2D features
with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

# Advanced features (selected)
with open(base_path / 'data/processed/advanced_features_cache.pkl', 'rb') as f:
    adv_data = pickle.load(f)
train_adv = adv_data['train']
test_adv = adv_data['test']

top_adv_features = [
    'g_mhps_100', 'pre_peak_r_i_slope', 'r_mhps_30', 'r_mhps_ratio_30_365',
    'r_abs_mag_peak', 'g_abs_mag_peak', 'r_fleet_width', 'r_fleet_asymmetry',
    'peak_lag_g_r', 'flux_skewness', 'r_acf_10d'
]
adv_cols = [c for c in train_adv.columns if c != 'object_id']
available_adv = [c for c in top_adv_features if c in adv_cols]

print(f"   Base features: {len(selected_120)}", flush=True)
print(f"   TDE physics: {len(tde_cols)}", flush=True)
print(f"   GP2D features: {len(gp2d_cols)}", flush=True)
print(f"   Advanced features: {len(available_adv)}", flush=True)

# ====================
# 2. EXTRACT NEW TDE-SPECIFIC FEATURES
# ====================
print("\n2. Computing new TDE-specific features...", flush=True)

# Band wavelengths for reference
BAND_WAVELENGTHS = {'u': 3670, 'g': 4825, 'r': 6222, 'i': 7545, 'z': 8691, 'y': 9710}
LSST_BANDS = ['u', 'g', 'r', 'i', 'z', 'y']


def compute_tde_specific_features(obj_lc: pd.DataFrame) -> dict:
    """
    Compute TDE-specific features for a single object.

    New features from 2025 TDE paper:
    1. Power-law decay index (t^alpha where alpha ~ -5/3 for TDEs)
    2. r-i color slopes
    3. Late-time colors (150d)
    4. Decay curvature (deviation from power-law)
    """
    features = {}

    # Organize by band
    band_data = {}
    for band in LSST_BANDS:
        band_lc = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')
        if len(band_lc) >= 3:
            band_data[band] = {
                'times': band_lc['Time (MJD)'].values,
                'fluxes': band_lc['Flux'].values,
                'errors': band_lc['Flux_err'].values
            }

    # Find peak time from r-band
    peak_time = np.nan
    peak_flux = np.nan
    if 'r' in band_data:
        peak_idx = np.argmax(band_data['r']['fluxes'])
        peak_time = band_data['r']['times'][peak_idx]
        peak_flux = band_data['r']['fluxes'][peak_idx]

    # === POWER-LAW DECAY INDEX ===
    # TDEs follow t^-5/3 ~ t^-1.67, SNe follow exponential decay
    # Fit: log(F) = alpha * log(t - t_peak) + const
    features['r_decay_alpha'] = np.nan
    features['r_decay_residual'] = np.nan
    features['g_decay_alpha'] = np.nan

    for band in ['r', 'g']:
        if band not in band_data or np.isnan(peak_time):
            continue

        times = band_data[band]['times']
        fluxes = band_data[band]['fluxes']

        # Use post-peak data only (t > peak + 5 days)
        post_peak_mask = (times > peak_time + 5) & (fluxes > 0)
        if np.sum(post_peak_mask) >= 5:
            t_post = times[post_peak_mask] - peak_time
            f_post = fluxes[post_peak_mask]

            # Log-log linear fit: log(F) = alpha * log(t) + const
            log_t = np.log10(t_post)
            log_f = np.log10(f_post)

            if np.std(log_t) > 0:
                try:
                    coeffs = np.polyfit(log_t, log_f, 1)
                    alpha = coeffs[0]  # Power-law index

                    # Compute residual (deviation from power-law)
                    predicted = coeffs[0] * log_t + coeffs[1]
                    residual = np.std(log_f - predicted)

                    features[f'{band}_decay_alpha'] = alpha
                    if band == 'r':
                        features['r_decay_residual'] = residual
                except:
                    pass

    # TDE signature: alpha close to -1.67 (t^-5/3)
    if not np.isnan(features.get('r_decay_alpha', np.nan)):
        features['r_decay_tde_like'] = abs(features['r_decay_alpha'] + 1.67)  # 0 = perfect TDE
    else:
        features['r_decay_tde_like'] = np.nan

    # === R-I COLOR SLOPES (missing from current implementation) ===
    features['ri_slope_50d'] = np.nan
    features['ri_slope_100d'] = np.nan

    if 'r' in band_data and 'i' in band_data and not np.isnan(peak_time):
        r_times = band_data['r']['times']
        r_fluxes = band_data['r']['fluxes']
        i_times = band_data['i']['times']
        i_fluxes = band_data['i']['fluxes']

        # Compute r-i colors at different epochs
        def get_flux_at_epoch(times, fluxes, target_time, window=10):
            """Get average flux near target time."""
            mask = np.abs(times - target_time) < window
            if np.sum(mask) >= 1:
                return np.mean(fluxes[mask])
            return np.nan

        # Colors at peak, +50d, +100d
        epochs = [0, 50, 100]
        ri_colors = {}

        for epoch in epochs:
            target_t = peak_time + epoch
            r_flux = get_flux_at_epoch(r_times, r_fluxes, target_t)
            i_flux = get_flux_at_epoch(i_times, i_fluxes, target_t)

            if r_flux > 0 and i_flux > 0:
                ri_colors[epoch] = -2.5 * np.log10(r_flux / i_flux)
            else:
                ri_colors[epoch] = np.nan

        # Compute slopes
        if not np.isnan(ri_colors.get(0, np.nan)) and not np.isnan(ri_colors.get(50, np.nan)):
            features['ri_slope_50d'] = (ri_colors[50] - ri_colors[0]) / 50.0

        if not np.isnan(ri_colors.get(0, np.nan)) and not np.isnan(ri_colors.get(100, np.nan)):
            features['ri_slope_100d'] = (ri_colors[100] - ri_colors[0]) / 100.0

    # === LATE-TIME COLORS (150d) ===
    features['gr_color_150d'] = np.nan
    features['ri_color_150d'] = np.nan

    if not np.isnan(peak_time):
        for b1, b2, name in [('g', 'r', 'gr'), ('r', 'i', 'ri')]:
            if b1 in band_data and b2 in band_data:
                target_t = peak_time + 150

                b1_times = band_data[b1]['times']
                b1_fluxes = band_data[b1]['fluxes']
                b2_times = band_data[b2]['times']
                b2_fluxes = band_data[b2]['fluxes']

                # Get flux near 150d
                mask1 = np.abs(b1_times - target_t) < 20
                mask2 = np.abs(b2_times - target_t) < 20

                if np.sum(mask1) >= 1 and np.sum(mask2) >= 1:
                    f1 = np.mean(b1_fluxes[mask1])
                    f2 = np.mean(b2_fluxes[mask2])

                    if f1 > 0 and f2 > 0:
                        features[f'{name}_color_150d'] = -2.5 * np.log10(f1 / f2)

    # === COLOR CURVATURE (second derivative) ===
    # TDEs: steady color, SNe: accelerating reddening
    features['gr_curvature'] = np.nan

    if 'g' in band_data and 'r' in band_data and not np.isnan(peak_time):
        g_times = band_data['g']['times']
        g_fluxes = band_data['g']['fluxes']
        r_times = band_data['r']['times']
        r_fluxes = band_data['r']['fluxes']

        # Compute colors at 0d, 50d, 100d
        gr_colors = []
        for epoch in [0, 50, 100]:
            target_t = peak_time + epoch

            g_mask = np.abs(g_times - target_t) < 15
            r_mask = np.abs(r_times - target_t) < 15

            if np.sum(g_mask) >= 1 and np.sum(r_mask) >= 1:
                g_flux = np.mean(g_fluxes[g_mask])
                r_flux = np.mean(r_fluxes[r_mask])

                if g_flux > 0 and r_flux > 0:
                    gr_colors.append(-2.5 * np.log10(g_flux / r_flux))
                else:
                    gr_colors.append(np.nan)
            else:
                gr_colors.append(np.nan)

        # Curvature = (c[100] - c[50]) - (c[50] - c[0]) = c[100] - 2*c[50] + c[0]
        if not any(np.isnan(c) for c in gr_colors):
            features['gr_curvature'] = gr_colors[2] - 2*gr_colors[1] + gr_colors[0]

    # === FLUX RATIO EVOLUTION ===
    # How does peak flux ratio compare to late-time?
    features['gr_flux_ratio_evolution'] = np.nan

    if 'g' in band_data and 'r' in band_data and not np.isnan(peak_time):
        g_times = band_data['g']['times']
        g_fluxes = band_data['g']['fluxes']
        r_times = band_data['r']['times']
        r_fluxes = band_data['r']['fluxes']

        # At peak
        g_peak = np.max(g_fluxes)
        r_peak = np.max(r_fluxes)

        # At late time (100d+)
        g_late_mask = (g_times > peak_time + 80)
        r_late_mask = (r_times > peak_time + 80)

        if np.sum(g_late_mask) >= 1 and np.sum(r_late_mask) >= 1:
            g_late = np.mean(g_fluxes[g_late_mask])
            r_late = np.mean(r_fluxes[r_late_mask])

            if r_peak > 0 and r_late > 0:
                peak_ratio = g_peak / r_peak
                late_ratio = g_late / r_late
                features['gr_flux_ratio_evolution'] = late_ratio / peak_ratio

    return features


def extract_tde_features_batch(lc_df, object_ids, verbose=True):
    """Extract TDE features for multiple objects."""
    grouped = {obj_id: group for obj_id, group in lc_df.groupby('object_id')}

    all_features = []
    for i, obj_id in enumerate(object_ids):
        if verbose and (i + 1) % 500 == 0:
            print(f"      TDE features: {i+1}/{len(object_ids)}...", flush=True)

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            features = {
                'object_id': obj_id,
                'r_decay_alpha': np.nan,
                'r_decay_residual': np.nan,
                'r_decay_tde_like': np.nan,
                'g_decay_alpha': np.nan,
                'ri_slope_50d': np.nan,
                'ri_slope_100d': np.nan,
                'gr_color_150d': np.nan,
                'ri_color_150d': np.nan,
                'gr_curvature': np.nan,
                'gr_flux_ratio_evolution': np.nan
            }
        else:
            features = compute_tde_specific_features(obj_lc)
            features['object_id'] = obj_id

        all_features.append(features)

    return pd.DataFrame(all_features)


# Check if cache exists
cache_path = base_path / 'data/processed/tde_specific_cache.pkl'

if cache_path.exists():
    print("   Loading cached TDE-specific features...", flush=True)
    with open(cache_path, 'rb') as f:
        tde_specific_cache = pickle.load(f)
    train_tde_spec = tde_specific_cache['train']
    test_tde_spec = tde_specific_cache['test']
else:
    print("   Computing TDE-specific features for train...", flush=True)
    train_tde_spec = extract_tde_features_batch(train_lc, train_ids)

    print("   Computing TDE-specific features for test...", flush=True)
    test_tde_spec = extract_tde_features_batch(test_lc, test_ids)

    # Cache
    with open(cache_path, 'wb') as f:
        pickle.dump({'train': train_tde_spec, 'test': test_tde_spec}, f)
    print("   Cached TDE-specific features.", flush=True)

tde_spec_cols = [c for c in train_tde_spec.columns if c != 'object_id']
print(f"   New TDE-specific features: {len(tde_spec_cols)}", flush=True)
print(f"   Features: {tde_spec_cols}", flush=True)

# ====================
# 3. COMBINE ALL FEATURES
# ====================
print("\n3. Combining all features...", flush=True)

# Merge all feature sets
train_combined = train_base.copy()
test_combined = test_base.copy()

# Add GP2D
train_combined = train_combined.merge(train_gp2d, on='object_id', how='left')
test_combined = test_combined.merge(test_gp2d, on='object_id', how='left')

# Add advanced (selected)
train_adv_selected = train_adv[['object_id'] + available_adv]
test_adv_selected = test_adv[['object_id'] + available_adv]
train_combined = train_combined.merge(train_adv_selected, on='object_id', how='left')
test_combined = test_combined.merge(test_adv_selected, on='object_id', how='left')

# Add NEW TDE-specific features
train_combined = train_combined.merge(train_tde_spec, on='object_id', how='left')
test_combined = test_combined.merge(test_tde_spec, on='object_id', how='left')

# Select feature columns
base_cols = [c for c in selected_120 if c in train_combined.columns]
all_feature_cols = base_cols + tde_cols + gp2d_cols + available_adv + tde_spec_cols
all_feature_cols = list(dict.fromkeys(all_feature_cols))  # Remove duplicates
all_feature_cols = [c for c in all_feature_cols if c in train_combined.columns]

print(f"   Total features: {len(all_feature_cols)}", flush=True)

# Prepare data
train_combined = train_combined.set_index('object_id').loc[train_ids].reset_index()

X = train_combined[all_feature_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
X_test = test_combined[all_feature_cols].values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

n_neg, n_pos = np.sum(y == 0), np.sum(y == 1)
scale_pos_weight = n_neg / n_pos

print(f"   Samples: {len(y)} ({n_pos} TDE, {n_neg} non-TDE)", flush=True)

# ====================
# 4. LOAD OPTUNA XGB PARAMS
# ====================
print("\n4. Loading Optuna XGBoost params...", flush=True)

with open(base_path / 'data/processed/optuna_v20c_results.pkl', 'rb') as f:
    optuna_data = pickle.load(f)

xgb_params = optuna_data['xgb_best_params']
print(f"   Params: {xgb_params}", flush=True)

# ====================
# 5. TRAIN XGBoost
# ====================
print("\n5. Training XGBoost (5-fold CV)...", flush=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros((len(test_ids), 5))
models = []
feature_importance = np.zeros(len(all_feature_cols))

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = xgb.XGBClassifier(
        **xgb_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds[:, fold] = model.predict_proba(X_test)[:, 1]
    models.append(model)
    feature_importance += model.feature_importances_

    # Fold F1
    best_f1 = 0
    for t in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_val, (oof_preds[val_idx] > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1

    print(f"   Fold {fold+1}: F1={best_f1:.4f}", flush=True)

# ====================
# 6. FIND OPTIMAL THRESHOLD
# ====================
print("\n6. Finding optimal threshold...", flush=True)

best_f1 = 0
best_thresh = 0.5

for t in np.arange(0.05, 0.95, 0.01):
    f1 = f1_score(y, (oof_preds > t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}", flush=True)

# Confusion matrix
final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
tn = np.sum((final_preds == 0) & (y == 0))

print(f"   Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}", flush=True)
print(f"   Recall: {tp/(tp+fn):.4f}", flush=True)

# ====================
# 7. ANALYZE NEW FEATURE IMPORTANCE
# ====================
print("\n7. New TDE feature importance...", flush=True)

feature_importance /= 5  # Average across folds
importance_df = pd.DataFrame({
    'feature': all_feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Focus on new TDE-specific features
new_features = importance_df[importance_df['feature'].isin(tde_spec_cols)]
print(f"   New TDE features ranked by importance:")
for _, row in new_features.iterrows():
    print(f"      {row['feature']}: {row['importance']:.4f}", flush=True)

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

submission_path = base_path / 'submissions/submission_v23_tde_features.csv'
submission.to_csv(submission_path, index=False)

print(f"   Saved to {submission_path}", flush=True)
print(f"   Predictions: {test_final.sum()} TDEs / {len(test_final)} total ({100*test_final.mean():.1f}%)", flush=True)

# Save model and results
with open(base_path / 'data/processed/models_v23_tde.pkl', 'wb') as f:
    pickle.dump({
        'models': models,
        'best_thresh': best_thresh,
        'feature_cols': all_feature_cols,
        'xgb_params': xgb_params,
        'oof_f1': best_f1,
        'importance_df': importance_df
    }, f)

# ====================
# SUMMARY
# ====================
print("\n" + "=" * 60, flush=True)
print("v23 TDE Features Training Complete!", flush=True)
print("=" * 60, flush=True)

print(f"\nVersion Comparison:", flush=True)
print(f"  v19 (GBM+GP2D):       OOF F1 = 0.6626, LB = 0.6649", flush=True)
print(f"  v21 (XGB only):       OOF F1 = 0.6708, LB = 0.6649", flush=True)
print(f"  v23 (TDE features):   OOF F1 = {best_f1:.4f}", flush=True)

delta = best_f1 - 0.6708
if delta > 0:
    print(f"\n  +{delta*100:.2f}% improvement over v21!", flush=True)
else:
    print(f"\n  {delta*100:.2f}% vs v21 (new features may not help)", flush=True)

print("\nTop 10 features:")
for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
    marker = "[NEW]" if row['feature'] in tde_spec_cols else "     "
    print(f"  {i+1}. {marker} {row['feature']}: {row['importance']:.4f}", flush=True)

print("=" * 60, flush=True)
