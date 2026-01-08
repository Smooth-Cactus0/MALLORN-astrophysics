"""
MALLORN v58: FWHM (Full Width at Half Maximum) Features

Physics motivation:
- TDEs have broad peaks (FWHM ~ weeks to months) due to slow accretion disk drainage
- SNe have sharper peaks (FWHM ~ days to weeks), especially Type Ia
- AGN lack well-defined peaks, FWHM is poorly constrained or very large

Features:
- FWHM per band (in days)
- Rise HWHM (half-width on rise side)
- Fall HWHM (half-width on fall side)
- Asymmetry ratio (fall/rise HWHM)
- Average FWHM across bands
- FWHM ratios between bands
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

BANDS = ['u', 'g', 'r', 'i', 'z', 'y']


def extract_fwhm_features(obj_id, lc_data):
    """
    Extract FWHM features for a single object.

    FWHM = Full Width at Half Maximum
    - Time duration where flux > 0.5 * peak_flux
    """
    features = {'object_id': obj_id}

    obj_lc = lc_data[lc_data['object_id'] == obj_id]
    if obj_lc.empty:
        return features

    band_fwhm = {}
    band_rise_hwhm = {}
    band_fall_hwhm = {}

    for band in BANDS:
        band_lc = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_lc) < 5:
            features[f'{band}_fwhm'] = np.nan
            features[f'{band}_rise_hwhm'] = np.nan
            features[f'{band}_fall_hwhm'] = np.nan
            features[f'{band}_fwhm_asymmetry'] = np.nan
            continue

        times = band_lc['Time (MJD)'].values
        fluxes = band_lc['Flux'].values

        # Find peak
        peak_idx = np.argmax(fluxes)
        peak_time = times[peak_idx]
        peak_flux = fluxes[peak_idx]

        # Half maximum threshold
        half_max = peak_flux / 2.0

        # If peak flux is negative or very small, skip
        if peak_flux <= 0:
            features[f'{band}_fwhm'] = np.nan
            features[f'{band}_rise_hwhm'] = np.nan
            features[f'{band}_fall_hwhm'] = np.nan
            features[f'{band}_fwhm_asymmetry'] = np.nan
            continue

        # Find rise HWHM (time from half-max to peak on rising side)
        rise_times = times[:peak_idx + 1]
        rise_fluxes = fluxes[:peak_idx + 1]

        rise_hwhm = np.nan
        if len(rise_times) >= 2:
            # Find where flux crosses half_max on rise
            above_half = rise_fluxes >= half_max
            if np.any(above_half) and np.any(~above_half):
                # Find first crossing point
                cross_idx = np.where(above_half)[0][0]
                if cross_idx > 0:
                    # Linear interpolation to find exact crossing time
                    t1, t2 = rise_times[cross_idx - 1], rise_times[cross_idx]
                    f1, f2 = rise_fluxes[cross_idx - 1], rise_fluxes[cross_idx]
                    if f2 != f1:
                        t_cross = t1 + (half_max - f1) * (t2 - t1) / (f2 - f1)
                        rise_hwhm = peak_time - t_cross
            elif np.all(above_half):
                # All points above half-max, use first observation
                rise_hwhm = peak_time - rise_times[0]

        # Find fall HWHM (time from peak to half-max on falling side)
        fall_times = times[peak_idx:]
        fall_fluxes = fluxes[peak_idx:]

        fall_hwhm = np.nan
        if len(fall_times) >= 2:
            # Find where flux drops below half_max
            above_half = fall_fluxes >= half_max
            if np.any(above_half) and np.any(~above_half):
                # Find first crossing point (flux drops below half_max)
                below_indices = np.where(~above_half)[0]
                if len(below_indices) > 0:
                    cross_idx = below_indices[0]
                    if cross_idx > 0:
                        # Linear interpolation
                        t1, t2 = fall_times[cross_idx - 1], fall_times[cross_idx]
                        f1, f2 = fall_fluxes[cross_idx - 1], fall_fluxes[cross_idx]
                        if f2 != f1:
                            t_cross = t1 + (half_max - f1) * (t2 - t1) / (f2 - f1)
                            fall_hwhm = t_cross - peak_time
            elif np.all(above_half):
                # All points above half-max, use last observation
                fall_hwhm = fall_times[-1] - peak_time

        # Store features
        features[f'{band}_rise_hwhm'] = rise_hwhm
        features[f'{band}_fall_hwhm'] = fall_hwhm

        # FWHM = rise_hwhm + fall_hwhm
        if not np.isnan(rise_hwhm) and not np.isnan(fall_hwhm):
            features[f'{band}_fwhm'] = rise_hwhm + fall_hwhm
            band_fwhm[band] = rise_hwhm + fall_hwhm
            band_rise_hwhm[band] = rise_hwhm
            band_fall_hwhm[band] = fall_hwhm
        else:
            features[f'{band}_fwhm'] = np.nan

        # Asymmetry (fall/rise ratio)
        if not np.isnan(rise_hwhm) and not np.isnan(fall_hwhm) and rise_hwhm > 0:
            features[f'{band}_fwhm_asymmetry'] = fall_hwhm / rise_hwhm
        else:
            features[f'{band}_fwhm_asymmetry'] = np.nan

    # Aggregate features across bands
    if band_fwhm:
        features['fwhm_mean'] = np.nanmean(list(band_fwhm.values()))
        features['fwhm_std'] = np.nanstd(list(band_fwhm.values()))
        features['fwhm_max'] = np.nanmax(list(band_fwhm.values()))
        features['fwhm_min'] = np.nanmin(list(band_fwhm.values()))
    else:
        features['fwhm_mean'] = np.nan
        features['fwhm_std'] = np.nan
        features['fwhm_max'] = np.nan
        features['fwhm_min'] = np.nan

    # FWHM ratios between key bands
    if 'g' in band_fwhm and 'r' in band_fwhm and band_fwhm['r'] > 0:
        features['fwhm_g_over_r'] = band_fwhm['g'] / band_fwhm['r']
    else:
        features['fwhm_g_over_r'] = np.nan

    if 'r' in band_fwhm and 'i' in band_fwhm and band_fwhm['i'] > 0:
        features['fwhm_r_over_i'] = band_fwhm['r'] / band_fwhm['i']
    else:
        features['fwhm_r_over_i'] = np.nan

    # Average asymmetry
    asymmetries = [features.get(f'{b}_fwhm_asymmetry', np.nan) for b in BANDS]
    valid_asym = [a for a in asymmetries if not np.isnan(a)]
    if valid_asym:
        features['fwhm_asymmetry_mean'] = np.mean(valid_asym)
    else:
        features['fwhm_asymmetry_mean'] = np.nan

    return features


def load_data():
    """Load lightcurve and metadata."""
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'

    train_meta = pd.read_csv(data_dir / 'train_log.csv')
    test_meta = pd.read_csv(data_dir / 'test_log.csv')

    train_lcs, test_lcs = [], []
    for i in range(1, 21):
        split_dir = data_dir / f'split_{i:02d}'
        train_path = split_dir / 'train_full_lightcurves.csv'
        test_path = split_dir / 'test_full_lightcurves.csv'
        if train_path.exists():
            train_lcs.append(pd.read_csv(train_path))
        if test_path.exists():
            test_lcs.append(pd.read_csv(test_path))

    train_lc = pd.concat(train_lcs, ignore_index=True)
    test_lc = pd.concat(test_lcs, ignore_index=True)

    return train_meta, test_meta, train_lc, test_lc


def main():
    print("=" * 80)
    print("MALLORN v58: FWHM (Full Width at Half Maximum) Features")
    print("=" * 80)

    # =========================================================================
    # 1. Load data
    # =========================================================================
    print("\n1. Loading data...")
    train_meta, test_meta, train_lc, test_lc = load_data()
    print(f"   Training objects: {len(train_meta)}")
    print(f"   Test objects: {len(test_meta)}")
    print(f"   TDEs: {train_meta['target'].sum()} ({100*train_meta['target'].mean():.1f}%)")

    # =========================================================================
    # 2. Extract FWHM features
    # =========================================================================
    print("\n2. Extracting FWHM features...")

    print("   Training set...")
    train_fwhm = []
    for i, obj_id in enumerate(train_meta['object_id']):
        if (i + 1) % 500 == 0:
            print(f"      Progress: {i+1}/{len(train_meta)}")
        feats = extract_fwhm_features(obj_id, train_lc)
        train_fwhm.append(feats)
    train_fwhm_df = pd.DataFrame(train_fwhm)

    print("   Test set...")
    test_fwhm = []
    for i, obj_id in enumerate(test_meta['object_id']):
        if (i + 1) % 1000 == 0:
            print(f"      Progress: {i+1}/{len(test_meta)}")
        feats = extract_fwhm_features(obj_id, test_lc)
        test_fwhm.append(feats)
    test_fwhm_df = pd.DataFrame(test_fwhm)

    fwhm_cols = [c for c in train_fwhm_df.columns if c != 'object_id']
    print(f"   FWHM features: {len(fwhm_cols)}")

    # =========================================================================
    # 3. Analyze FWHM by class
    # =========================================================================
    print("\n3. FWHM Analysis by Class:")

    train_fwhm_with_labels = train_fwhm_df.merge(
        train_meta[['object_id', 'target', 'SpecType']], on='object_id'
    )

    # Compare TDE vs non-TDE
    tde_mask = train_fwhm_with_labels['target'] == 1

    print("\n   Mean FWHM (days) by class:")
    print(f"   {'Band':<10} {'TDE':>12} {'non-TDE':>12} {'Ratio':>10}")
    print("   " + "-" * 46)

    for band in ['r', 'g', 'i']:
        col = f'{band}_fwhm'
        tde_mean = train_fwhm_with_labels.loc[tde_mask, col].mean()
        non_tde_mean = train_fwhm_with_labels.loc[~tde_mask, col].mean()
        ratio = tde_mean / non_tde_mean if non_tde_mean > 0 else np.nan
        print(f"   {band:<10} {tde_mean:>12.1f} {non_tde_mean:>12.1f} {ratio:>10.2f}x")

    # Overall mean FWHM
    tde_mean_all = train_fwhm_with_labels.loc[tde_mask, 'fwhm_mean'].mean()
    non_tde_mean_all = train_fwhm_with_labels.loc[~tde_mask, 'fwhm_mean'].mean()
    ratio_all = tde_mean_all / non_tde_mean_all if non_tde_mean_all > 0 else np.nan
    print(f"   {'mean':<10} {tde_mean_all:>12.1f} {non_tde_mean_all:>12.1f} {ratio_all:>10.2f}x")

    # Asymmetry comparison
    print("\n   Mean Asymmetry (fall/rise) by class:")
    for band in ['r', 'g']:
        col = f'{band}_fwhm_asymmetry'
        tde_asym = train_fwhm_with_labels.loc[tde_mask, col].mean()
        non_tde_asym = train_fwhm_with_labels.loc[~tde_mask, col].mean()
        print(f"   {band}: TDE={tde_asym:.2f}, non-TDE={non_tde_asym:.2f}")

    # =========================================================================
    # 4. Load baseline features (v34a structure)
    # =========================================================================
    print("\n4. Loading baseline features...")
    cache_dir = Path(__file__).parent.parent / 'data' / 'processed'

    with open(cache_dir / 'features_cache.pkl', 'rb') as f:
        base_cache = pickle.load(f)
    train_features = base_cache['train_features'].copy()
    test_features = base_cache['test_features'].copy()

    with open(cache_dir / 'bazin_features_cache.pkl', 'rb') as f:
        bazin_cache = pickle.load(f)
    train_features = train_features.merge(bazin_cache['train'], on='object_id', how='left')
    test_features = test_features.merge(bazin_cache['test'], on='object_id', how='left')

    with open(cache_dir / 'tde_physics_cache.pkl', 'rb') as f:
        tde_cache = pickle.load(f)
    train_features = train_features.merge(tde_cache['train'], on='object_id', how='left')
    test_features = test_features.merge(tde_cache['test'], on='object_id', how='left')

    with open(cache_dir / 'multiband_gp_cache.pkl', 'rb') as f:
        gp_cache = pickle.load(f)
    train_features = train_features.merge(gp_cache['train'], on='object_id', how='left')
    test_features = test_features.merge(gp_cache['test'], on='object_id', how='left')

    print(f"   Baseline features: {len(train_features.columns) - 1}")

    # Add FWHM features
    train_features = train_features.merge(train_fwhm_df, on='object_id', how='left')
    test_features = test_features.merge(test_fwhm_df, on='object_id', how='left')

    print(f"   + FWHM features: {len(fwhm_cols)}")
    print(f"   = Total: {len(train_features.columns) - 1}")

    # =========================================================================
    # 5. Train model
    # =========================================================================
    print("\n5. Training XGBoost...")

    feature_cols = [c for c in train_features.columns if c != 'object_id']
    X_train = train_features[feature_cols].copy()
    X_test = test_features[feature_cols].copy()
    y_train = train_meta['target'].values

    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    feature_importance = np.zeros(len(feature_cols))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=len(y_tr[y_tr==0]) / max(1, len(y_tr[y_tr==1])),
            random_state=42 + fold, n_jobs=-1, verbosity=0
        )

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.05, 0.5, 0.01):
            f1 = f1_score(y_val, (model.predict_proba(X_val)[:, 1] >= thresh).astype(int))
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        print(f"   Fold {fold+1}/5: F1={best_f1:.4f} @ thresh={best_thresh:.2f}")

        test_preds += model.predict_proba(X_test)[:, 1] / 5
        feature_importance += model.feature_importances_ / 5

    # Find optimal threshold
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.05, 0.5, 0.01):
        f1 = f1_score(y_train, (oof_preds >= thresh).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    print(f"\n   >>> v58 OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}")

    # =========================================================================
    # 6. Feature importance - FWHM features
    # =========================================================================
    print("\n6. FWHM Feature Importance:")

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print("\n   Top 20 features:")
    for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
        marker = "[FWHM]" if 'fwhm' in row['feature'].lower() or 'hwhm' in row['feature'].lower() else ""
        print(f"   #{i+1:2d}: {row['feature']:<35s} {row['importance']:>8.1f} {marker}")

    print("\n   FWHM feature rankings:")
    for feat in fwhm_cols:
        if feat in list(importance_df['feature']):
            rank = list(importance_df['feature']).index(feat) + 1
            imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
            print(f"      {feat:<30s}: rank #{rank:3d}, importance={imp:.1f}")

    # =========================================================================
    # 7. Results comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(f"\n   v34a baseline:    OOF F1=0.6667, LB F1=0.6907")
    print(f"   v57a extinction:  OOF F1=0.6292, LB F1=0.643")
    print(f"   v58 +FWHM:        OOF F1={best_f1:.4f}")

    # =========================================================================
    # 8. Create submission
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUBMISSION")
    print("=" * 80)

    sub_dir = Path(__file__).parent.parent / 'submissions'
    sub_dir.mkdir(exist_ok=True)

    submission = pd.DataFrame({
        'object_id': test_meta['object_id'],
        'target': (test_preds >= best_thresh).astype(int)
    })
    submission.to_csv(sub_dir / 'submission_v58_fwhm.csv', index=False)
    print(f"   Saved: submission_v58_fwhm.csv")
    print(f"   Predicted TDEs: {submission['target'].sum()}")

    # Save artifacts
    artifacts = {
        'oof_preds': oof_preds,
        'test_preds': test_preds,
        'best_threshold': best_thresh,
        'oof_f1': best_f1,
        'feature_importance': importance_df,
        'fwhm_features': fwhm_cols
    }
    with open(cache_dir / 'v58_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
