"""
MALLORN v59b: Color Temperature at FWHM Points

Physics:
- TDEs maintain hot temperatures (~20,000-40,000 K) throughout evolution
- SNe cool rapidly after peak (adiabatic expansion)
- Temperature at half-max points reveals cooling rate

Features:
- Temperature at rise half-max (before peak)
- Temperature at fall half-max (after peak)
- Temperature change from rise to fall half-max
- Temperature at peak vs half-max ratio
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

BANDS = ['u', 'g', 'r', 'i', 'z', 'y']


def estimate_temperature_from_gr(g_flux, r_flux):
    """
    Estimate blackbody temperature from g-r color.
    Uses Wien's law approximation.
    """
    if g_flux <= 0 or r_flux <= 0:
        return np.nan

    # g-r color in magnitudes
    g_r = -2.5 * np.log10(g_flux / r_flux)

    # Empirical calibration: T ~ 7000K / (g-r + 0.6)
    if g_r < -0.5:
        return 50000  # Very hot
    elif g_r > 2.0:
        return 3000   # Cool
    else:
        return 7000 / (g_r + 0.6)


def find_flux_at_time(times, fluxes, target_time, tolerance=10):
    """Interpolate flux at a target time."""
    if len(times) < 2:
        return np.nan

    # Find nearest points
    diffs = np.abs(times - target_time)
    if np.min(diffs) > tolerance:
        return np.nan

    # Linear interpolation
    idx = np.searchsorted(times, target_time)
    if idx == 0:
        return fluxes[0]
    if idx >= len(times):
        return fluxes[-1]

    t1, t2 = times[idx-1], times[idx]
    f1, f2 = fluxes[idx-1], fluxes[idx]
    if t2 == t1:
        return f1
    w = (target_time - t1) / (t2 - t1)
    return f1 + w * (f2 - f1)


def extract_temp_fwhm_features(obj_id, lc_data):
    """Extract temperature at FWHM features."""
    features = {'object_id': obj_id}

    obj_lc = lc_data[lc_data['object_id'] == obj_id]
    if obj_lc.empty:
        return features

    # Get g and r band data
    g_lc = obj_lc[obj_lc['Filter'] == 'g'].sort_values('Time (MJD)')
    r_lc = obj_lc[obj_lc['Filter'] == 'r'].sort_values('Time (MJD)')

    if len(g_lc) < 5 or len(r_lc) < 5:
        return features

    # Use r-band as reference for peak and half-max times
    r_times = r_lc['Time (MJD)'].values
    r_fluxes = r_lc['Flux'].values
    g_times = g_lc['Time (MJD)'].values
    g_fluxes = g_lc['Flux'].values

    # Find peak
    peak_idx = np.argmax(r_fluxes)
    peak_time = r_times[peak_idx]
    peak_flux = r_fluxes[peak_idx]

    if peak_flux <= 0:
        return features

    half_max = peak_flux / 2.0

    # Find rise half-max time
    rise_hm_time = np.nan
    for i in range(peak_idx):
        if r_fluxes[i] < half_max <= r_fluxes[i+1]:
            # Linear interpolation
            t1, t2 = r_times[i], r_times[i+1]
            f1, f2 = r_fluxes[i], r_fluxes[i+1]
            if f2 != f1:
                rise_hm_time = t1 + (half_max - f1) * (t2 - t1) / (f2 - f1)
            break

    # Find fall half-max time
    fall_hm_time = np.nan
    for i in range(peak_idx, len(r_times)-1):
        if r_fluxes[i] >= half_max > r_fluxes[i+1]:
            t1, t2 = r_times[i], r_times[i+1]
            f1, f2 = r_fluxes[i], r_fluxes[i+1]
            if f2 != f1:
                fall_hm_time = t1 + (half_max - f1) * (t2 - t1) / (f2 - f1)
            break

    # Temperature at peak
    g_at_peak = find_flux_at_time(g_times, g_fluxes, peak_time)
    r_at_peak = find_flux_at_time(r_times, r_fluxes, peak_time)
    temp_at_peak = estimate_temperature_from_gr(g_at_peak, r_at_peak)
    features['temp_at_peak'] = temp_at_peak

    # Temperature at rise half-max
    if not np.isnan(rise_hm_time):
        g_at_rise = find_flux_at_time(g_times, g_fluxes, rise_hm_time)
        r_at_rise = find_flux_at_time(r_times, r_fluxes, rise_hm_time)
        temp_at_rise_hm = estimate_temperature_from_gr(g_at_rise, r_at_rise)
        features['temp_at_rise_hm'] = temp_at_rise_hm
    else:
        features['temp_at_rise_hm'] = np.nan

    # Temperature at fall half-max
    if not np.isnan(fall_hm_time):
        g_at_fall = find_flux_at_time(g_times, g_fluxes, fall_hm_time)
        r_at_fall = find_flux_at_time(r_times, r_fluxes, fall_hm_time)
        temp_at_fall_hm = estimate_temperature_from_gr(g_at_fall, r_at_fall)
        features['temp_at_fall_hm'] = temp_at_fall_hm
    else:
        features['temp_at_fall_hm'] = np.nan

    # Temperature evolution metrics
    if not np.isnan(features.get('temp_at_rise_hm')) and not np.isnan(features.get('temp_at_fall_hm')):
        features['temp_change_hm'] = features['temp_at_fall_hm'] - features['temp_at_rise_hm']

        if features['temp_at_rise_hm'] > 0:
            features['temp_ratio_fall_rise'] = features['temp_at_fall_hm'] / features['temp_at_rise_hm']
        else:
            features['temp_ratio_fall_rise'] = np.nan
    else:
        features['temp_change_hm'] = np.nan
        features['temp_ratio_fall_rise'] = np.nan

    # Peak to half-max temperature change
    if not np.isnan(temp_at_peak) and not np.isnan(features.get('temp_at_fall_hm')):
        features['temp_drop_peak_to_hm'] = temp_at_peak - features['temp_at_fall_hm']
    else:
        features['temp_drop_peak_to_hm'] = np.nan

    # Cooling rate (K per day from peak to fall half-max)
    if not np.isnan(fall_hm_time) and not np.isnan(temp_at_peak) and not np.isnan(features.get('temp_at_fall_hm')):
        dt = fall_hm_time - peak_time
        if dt > 0:
            features['cooling_rate_to_hm'] = (temp_at_peak - features['temp_at_fall_hm']) / dt
        else:
            features['cooling_rate_to_hm'] = np.nan
    else:
        features['cooling_rate_to_hm'] = np.nan

    return features


def main():
    print("=" * 80)
    print("MALLORN v59b: Temperature at FWHM Features")
    print("=" * 80)

    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    cache_dir = Path(__file__).parent.parent / 'data' / 'processed'

    # Load data
    print("\n1. Loading data...")
    train_meta = pd.read_csv(data_dir / 'train_log.csv')
    test_meta = pd.read_csv(data_dir / 'test_log.csv')

    train_lcs, test_lcs = [], []
    for i in range(1, 21):
        split_dir = data_dir / f'split_{i:02d}'
        if (split_dir / 'train_full_lightcurves.csv').exists():
            train_lcs.append(pd.read_csv(split_dir / 'train_full_lightcurves.csv'))
        if (split_dir / 'test_full_lightcurves.csv').exists():
            test_lcs.append(pd.read_csv(split_dir / 'test_full_lightcurves.csv'))
    train_lc = pd.concat(train_lcs, ignore_index=True)
    test_lc = pd.concat(test_lcs, ignore_index=True)

    print(f"   Training: {len(train_meta)}, Test: {len(test_meta)}")

    # Extract features
    print("\n2. Extracting temperature at FWHM features...")

    print("   Training set...")
    train_feats = []
    for i, obj_id in enumerate(train_meta['object_id']):
        if (i + 1) % 500 == 0:
            print(f"      {i+1}/{len(train_meta)}")
        train_feats.append(extract_temp_fwhm_features(obj_id, train_lc))
    train_temp_df = pd.DataFrame(train_feats)

    print("   Test set...")
    test_feats = []
    for i, obj_id in enumerate(test_meta['object_id']):
        if (i + 1) % 1000 == 0:
            print(f"      {i+1}/{len(test_meta)}")
        test_feats.append(extract_temp_fwhm_features(obj_id, test_lc))
    test_temp_df = pd.DataFrame(test_feats)

    temp_cols = [c for c in train_temp_df.columns if c != 'object_id']
    print(f"   Temperature features: {len(temp_cols)}")

    # Analyze by class
    print("\n3. Temperature Analysis by Class:")
    train_labeled = train_temp_df.merge(train_meta[['object_id', 'target', 'SpecType']], on='object_id')

    for spec in ['TDE', 'AGN', 'SN Ia', 'SN II']:
        mask = train_labeled['SpecType'] == spec
        if mask.sum() > 0:
            temp_peak = train_labeled.loc[mask, 'temp_at_peak'].mean()
            temp_fall = train_labeled.loc[mask, 'temp_at_fall_hm'].mean()
            cooling = train_labeled.loc[mask, 'cooling_rate_to_hm'].mean()
            print(f"   {spec:8s}: T_peak={temp_peak:,.0f}K, T_fall_hm={temp_fall:,.0f}K, cooling={cooling:.0f} K/day")

    # Load baseline features
    print("\n4. Loading baseline features...")
    with open(cache_dir / 'features_cache.pkl', 'rb') as f:
        base = pickle.load(f)
    train_features = base['train_features'].copy()
    test_features = base['test_features'].copy()

    with open(cache_dir / 'bazin_features_cache.pkl', 'rb') as f:
        bazin = pickle.load(f)
    train_features = train_features.merge(bazin['train'], on='object_id', how='left')
    test_features = test_features.merge(bazin['test'], on='object_id', how='left')

    with open(cache_dir / 'tde_physics_cache.pkl', 'rb') as f:
        tde = pickle.load(f)
    train_features = train_features.merge(tde['train'], on='object_id', how='left')
    test_features = test_features.merge(tde['test'], on='object_id', how='left')

    with open(cache_dir / 'multiband_gp_cache.pkl', 'rb') as f:
        gp = pickle.load(f)
    train_features = train_features.merge(gp['train'], on='object_id', how='left')
    test_features = test_features.merge(gp['test'], on='object_id', how='left')

    # Add temperature features
    train_features = train_features.merge(train_temp_df, on='object_id', how='left')
    test_features = test_features.merge(test_temp_df, on='object_id', how='left')

    print(f"   Total features: {len(train_features.columns) - 1}")

    # Train
    print("\n5. Training...")
    feature_cols = [c for c in train_features.columns if c != 'object_id']
    X_train = train_features[feature_cols].fillna(train_features[feature_cols].median())
    X_test = test_features[feature_cols].fillna(X_train.median())
    y_train = train_meta['target'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            scale_pos_weight=20, random_state=42+fold, n_jobs=-1, verbosity=0
        )
        model.fit(X_train.iloc[tr_idx], y_train[tr_idx],
                  eval_set=[(X_train.iloc[val_idx], y_train[val_idx])], verbose=False)
        oof_preds[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / 5

        f1 = max([f1_score(y_train[val_idx], (oof_preds[val_idx] >= t).astype(int))
                  for t in np.arange(0.05, 0.5, 0.01)])
        print(f"   Fold {fold+1}: F1={f1:.4f}")

    best_f1, best_thresh = 0, 0.1
    for t in np.arange(0.05, 0.5, 0.01):
        f1 = f1_score(y_train, (oof_preds >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"\n   >>> v59b OOF F1: {best_f1:.4f} @ thresh={best_thresh:.2f}")

    # Save submission
    sub_dir = Path(__file__).parent.parent / 'submissions'
    sub = pd.DataFrame({'object_id': test_meta['object_id'],
                        'target': (test_preds >= best_thresh).astype(int)})
    sub.to_csv(sub_dir / 'submission_v59b_temp_fwhm.csv', index=False)
    print(f"   Saved: submission_v59b_temp_fwhm.csv ({sub['target'].sum()} TDEs)")


if __name__ == '__main__':
    main()
