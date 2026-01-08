"""
MALLORN v59c: Stetson Variability Indices

Classic variability indices from Stetson (1996) designed for stellar variability:
- Stetson J: Measures correlated variability (same direction deviations in pairs)
- Stetson K: Kurtosis-based, sensitive to outliers
- Stetson L: Combined J*K measure

Physics:
- AGN: Stochastic, uncorrelated variability -> low J
- TDE/SN: Coherent rise/fall -> high J (correlated deviations)
- Periodic variables: Very high J

Also adding:
- Eta (von Neumann ratio): Measures smoothness of lightcurve
- Weighted mean and std
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


def compute_stetson_indices(times, fluxes, errors):
    """
    Compute Stetson variability indices.
    """
    n = len(fluxes)
    if n < 3:
        return {'J': np.nan, 'K': np.nan, 'L': np.nan}

    # Weighted mean
    weights = 1.0 / (errors ** 2 + 1e-10)
    weighted_mean = np.sum(weights * fluxes) / np.sum(weights)

    # Normalized residuals (delta)
    delta = np.sqrt(n / (n - 1)) * (fluxes - weighted_mean) / errors

    # Stetson J: sum of sign(P_i * P_{i+1}) * sqrt(|P_i * P_{i+1}|)
    # where P_i = delta_i for single-band
    J = 0
    n_pairs = 0
    for i in range(n - 1):
        p = delta[i] * delta[i + 1]
        J += np.sign(p) * np.sqrt(np.abs(p))
        n_pairs += 1

    if n_pairs > 0:
        J = J / n_pairs
    else:
        J = np.nan

    # Stetson K: kurtosis measure
    K = np.sum(np.abs(delta)) / np.sqrt(np.sum(delta ** 2)) / np.sqrt(n)

    # Stetson L: J * K
    if not np.isnan(J) and not np.isnan(K):
        L = J * K
    else:
        L = np.nan

    return {'J': J, 'K': K, 'L': L}


def compute_eta(times, fluxes):
    """
    Compute von Neumann ratio (eta).
    Measures smoothness: low eta = smooth, high eta = noisy
    """
    n = len(fluxes)
    if n < 3:
        return np.nan

    # Sort by time
    sort_idx = np.argsort(times)
    fluxes_sorted = fluxes[sort_idx]

    # Von Neumann ratio
    mean_flux = np.mean(fluxes_sorted)
    numerator = np.sum((fluxes_sorted[1:] - fluxes_sorted[:-1]) ** 2)
    denominator = np.sum((fluxes_sorted - mean_flux) ** 2)

    if denominator > 0:
        eta = numerator / denominator
    else:
        eta = np.nan

    return eta


def extract_stetson_features(obj_id, lc_data):
    """Extract Stetson indices for one object."""
    features = {'object_id': obj_id}

    obj_lc = lc_data[lc_data['object_id'] == obj_id]
    if obj_lc.empty:
        return features

    all_J, all_K, all_L, all_eta = [], [], [], []

    for band in BANDS:
        band_lc = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_lc) < 5:
            features[f'{band}_stetson_J'] = np.nan
            features[f'{band}_stetson_K'] = np.nan
            features[f'{band}_stetson_L'] = np.nan
            features[f'{band}_eta'] = np.nan
            continue

        times = band_lc['Time (MJD)'].values
        fluxes = band_lc['Flux'].values
        errors = band_lc['Flux_err'].values

        # Stetson indices
        stetson = compute_stetson_indices(times, fluxes, errors)
        features[f'{band}_stetson_J'] = stetson['J']
        features[f'{band}_stetson_K'] = stetson['K']
        features[f'{band}_stetson_L'] = stetson['L']

        if not np.isnan(stetson['J']):
            all_J.append(stetson['J'])
        if not np.isnan(stetson['K']):
            all_K.append(stetson['K'])
        if not np.isnan(stetson['L']):
            all_L.append(stetson['L'])

        # Eta (von Neumann ratio)
        eta = compute_eta(times, fluxes)
        features[f'{band}_eta'] = eta
        if not np.isnan(eta):
            all_eta.append(eta)

    # Aggregate across bands
    features['stetson_J_mean'] = np.mean(all_J) if all_J else np.nan
    features['stetson_K_mean'] = np.mean(all_K) if all_K else np.nan
    features['stetson_L_mean'] = np.mean(all_L) if all_L else np.nan
    features['eta_mean'] = np.mean(all_eta) if all_eta else np.nan

    features['stetson_J_std'] = np.std(all_J) if len(all_J) > 1 else np.nan
    features['eta_std'] = np.std(all_eta) if len(all_eta) > 1 else np.nan

    # Cross-band Stetson J (correlation between bands)
    # Using r and g bands which are usually best sampled
    r_lc = obj_lc[obj_lc['Filter'] == 'r'].sort_values('Time (MJD)')
    g_lc = obj_lc[obj_lc['Filter'] == 'g'].sort_values('Time (MJD)')

    if len(r_lc) >= 5 and len(g_lc) >= 5:
        # Find common epochs (within 1 day)
        r_times = r_lc['Time (MJD)'].values
        r_fluxes = r_lc['Flux'].values
        r_errors = r_lc['Flux_err'].values
        g_times = g_lc['Time (MJD)'].values
        g_fluxes = g_lc['Flux'].values
        g_errors = g_lc['Flux_err'].values

        # Match observations
        matched_r_delta = []
        matched_g_delta = []

        r_mean = np.mean(r_fluxes)
        g_mean = np.mean(g_fluxes)
        n_r = len(r_fluxes)
        n_g = len(g_fluxes)

        for i, t_r in enumerate(r_times):
            # Find closest g observation
            dt = np.abs(g_times - t_r)
            if np.min(dt) < 1.0:  # Within 1 day
                j = np.argmin(dt)
                delta_r = np.sqrt(n_r / (n_r - 1)) * (r_fluxes[i] - r_mean) / r_errors[i]
                delta_g = np.sqrt(n_g / (n_g - 1)) * (g_fluxes[j] - g_mean) / g_errors[j]
                matched_r_delta.append(delta_r)
                matched_g_delta.append(delta_g)

        if len(matched_r_delta) >= 3:
            # Cross-band Stetson J
            J_cross = 0
            for dr, dg in zip(matched_r_delta, matched_g_delta):
                p = dr * dg
                J_cross += np.sign(p) * np.sqrt(np.abs(p))
            J_cross /= len(matched_r_delta)
            features['stetson_J_cross_gr'] = J_cross
        else:
            features['stetson_J_cross_gr'] = np.nan
    else:
        features['stetson_J_cross_gr'] = np.nan

    return features


def main():
    print("=" * 80)
    print("MALLORN v59c: Stetson Variability Indices")
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
    print("\n2. Extracting Stetson indices...")

    print("   Training set...")
    train_feats = []
    for i, obj_id in enumerate(train_meta['object_id']):
        if (i + 1) % 500 == 0:
            print(f"      {i+1}/{len(train_meta)}")
        train_feats.append(extract_stetson_features(obj_id, train_lc))
    train_stetson_df = pd.DataFrame(train_feats)

    print("   Test set...")
    test_feats = []
    for i, obj_id in enumerate(test_meta['object_id']):
        if (i + 1) % 1000 == 0:
            print(f"      {i+1}/{len(test_meta)}")
        test_feats.append(extract_stetson_features(obj_id, test_lc))
    test_stetson_df = pd.DataFrame(test_feats)

    stetson_cols = [c for c in train_stetson_df.columns if c != 'object_id']
    print(f"   Stetson features: {len(stetson_cols)}")

    # Analyze by class
    print("\n3. Stetson J Analysis by Class:")
    train_labeled = train_stetson_df.merge(train_meta[['object_id', 'target', 'SpecType']], on='object_id')

    print(f"   {'Class':8s} {'J_mean':>10s} {'J_cross_gr':>12s} {'eta':>10s}")
    print("   " + "-" * 42)
    for spec in ['TDE', 'AGN', 'SN Ia', 'SN II']:
        mask = train_labeled['SpecType'] == spec
        if mask.sum() > 0:
            j_mean = train_labeled.loc[mask, 'stetson_J_mean'].mean()
            j_cross = train_labeled.loc[mask, 'stetson_J_cross_gr'].mean()
            eta = train_labeled.loc[mask, 'eta_mean'].mean()
            print(f"   {spec:8s} {j_mean:>10.3f} {j_cross:>12.3f} {eta:>10.3f}")

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

    # Add Stetson features
    train_features = train_features.merge(train_stetson_df, on='object_id', how='left')
    test_features = test_features.merge(test_stetson_df, on='object_id', how='left')

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

    print(f"\n   >>> v59c OOF F1: {best_f1:.4f} @ thresh={best_thresh:.2f}")

    # Save submission
    sub_dir = Path(__file__).parent.parent / 'submissions'
    sub = pd.DataFrame({'object_id': test_meta['object_id'],
                        'target': (test_preds >= best_thresh).astype(int)})
    sub.to_csv(sub_dir / 'submission_v59c_stetson.csv', index=False)
    print(f"   Saved: submission_v59c_stetson.csv ({sub['target'].sum()} TDEs)")


if __name__ == '__main__':
    main()
