"""
MALLORN v59a: Variability Structure Function Features

The structure function SF(tau) measures how flux varies over different timescales:
    SF(tau) = sqrt(mean((flux(t+tau) - flux(t))^2))

Physics:
- AGN follow damped random walk: SF ~ tau^0.5 at short timescales
- Transients (TDE/SN) have coherent rise/fall: different SF shape
- Structure function slope and amplitude discriminate stochastic vs coherent variability
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


def compute_structure_function(times, fluxes, tau_bins=[5, 10, 20, 50, 100, 200]):
    """
    Compute structure function at different time lags.

    SF(tau) = sqrt(<(flux(t+tau) - flux(t))^2>)
    """
    sf_values = {}

    for tau in tau_bins:
        diffs_squared = []
        for i in range(len(times)):
            for j in range(i+1, len(times)):
                dt = abs(times[j] - times[i])
                if tau * 0.7 <= dt <= tau * 1.3:  # Allow 30% tolerance
                    diff = fluxes[j] - fluxes[i]
                    diffs_squared.append(diff ** 2)

        if len(diffs_squared) >= 3:
            sf_values[tau] = np.sqrt(np.mean(diffs_squared))
        else:
            sf_values[tau] = np.nan

    return sf_values


def extract_sf_features(obj_id, lc_data):
    """Extract structure function features for one object."""
    features = {'object_id': obj_id}

    obj_lc = lc_data[lc_data['object_id'] == obj_id]
    if obj_lc.empty:
        return features

    tau_bins = [5, 10, 20, 50, 100, 200]

    all_sf = {tau: [] for tau in tau_bins}

    for band in ['r', 'g', 'i']:  # Focus on well-sampled bands
        band_lc = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_lc) < 10:
            continue

        times = band_lc['Time (MJD)'].values
        fluxes = band_lc['Flux'].values

        sf = compute_structure_function(times, fluxes, tau_bins)

        for tau in tau_bins:
            features[f'{band}_sf_{tau}d'] = sf.get(tau, np.nan)
            if not np.isnan(sf.get(tau, np.nan)):
                all_sf[tau].append(sf[tau])

    # Aggregate SF across bands
    for tau in tau_bins:
        if all_sf[tau]:
            features[f'sf_mean_{tau}d'] = np.mean(all_sf[tau])
        else:
            features[f'sf_mean_{tau}d'] = np.nan

    # Structure function slope (log-log)
    # SF ~ tau^gamma, where gamma~0.5 for AGN (DRW)
    valid_taus = []
    valid_sfs = []
    for tau in tau_bins:
        sf_val = features.get(f'sf_mean_{tau}d', np.nan)
        if not np.isnan(sf_val) and sf_val > 0:
            valid_taus.append(np.log10(tau))
            valid_sfs.append(np.log10(sf_val))

    if len(valid_taus) >= 3:
        # Linear fit in log-log space
        coeffs = np.polyfit(valid_taus, valid_sfs, 1)
        features['sf_slope'] = coeffs[0]  # gamma
        features['sf_intercept'] = coeffs[1]
    else:
        features['sf_slope'] = np.nan
        features['sf_intercept'] = np.nan

    # SF ratio (short vs long timescale)
    sf_short = features.get('sf_mean_10d', np.nan)
    sf_long = features.get('sf_mean_100d', np.nan)
    if not np.isnan(sf_short) and not np.isnan(sf_long) and sf_long > 0:
        features['sf_ratio_10_100'] = sf_short / sf_long
    else:
        features['sf_ratio_10_100'] = np.nan

    return features


def main():
    print("=" * 80)
    print("MALLORN v59a: Structure Function Features")
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

    # Extract SF features
    print("\n2. Extracting structure function features...")

    print("   Training set...")
    train_sf = []
    for i, obj_id in enumerate(train_meta['object_id']):
        if (i + 1) % 500 == 0:
            print(f"      {i+1}/{len(train_meta)}")
        train_sf.append(extract_sf_features(obj_id, train_lc))
    train_sf_df = pd.DataFrame(train_sf)

    print("   Test set...")
    test_sf = []
    for i, obj_id in enumerate(test_meta['object_id']):
        if (i + 1) % 1000 == 0:
            print(f"      {i+1}/{len(test_meta)}")
        test_sf.append(extract_sf_features(obj_id, test_lc))
    test_sf_df = pd.DataFrame(test_sf)

    sf_cols = [c for c in train_sf_df.columns if c != 'object_id']
    print(f"   SF features: {len(sf_cols)}")

    # Analyze SF slope by class
    print("\n3. SF Analysis by Class:")
    train_sf_labeled = train_sf_df.merge(train_meta[['object_id', 'target', 'SpecType']], on='object_id')

    for spec in ['TDE', 'AGN', 'SN Ia', 'SN II']:
        mask = train_sf_labeled['SpecType'] == spec
        if mask.sum() > 0:
            slope = train_sf_labeled.loc[mask, 'sf_slope'].mean()
            print(f"   {spec:8s}: SF slope = {slope:.3f} (n={mask.sum()})")

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

    # Add SF features
    train_features = train_features.merge(train_sf_df, on='object_id', how='left')
    test_features = test_features.merge(test_sf_df, on='object_id', how='left')

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

    print(f"\n   >>> v59a OOF F1: {best_f1:.4f} @ thresh={best_thresh:.2f}")

    # Save submission
    sub_dir = Path(__file__).parent.parent / 'submissions'
    sub = pd.DataFrame({'object_id': test_meta['object_id'],
                        'target': (test_preds >= best_thresh).astype(int)})
    sub.to_csv(sub_dir / 'submission_v59a_sf.csv', index=False)
    print(f"   Saved: submission_v59a_sf.csv ({sub['target'].sum()} TDEs)")


if __name__ == '__main__':
    main()
