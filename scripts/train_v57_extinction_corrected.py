"""
MALLORN v57: Proper Extinction Correction using Fitzpatrick99 + Selective Features

Key improvements:
1. Apply PROPER dust extinction correction using the official method from MALLORN notebook:
   - extinction package with fitzpatrick99 law
   - R_V = 3.1 (standard Milky Way)
   - Effective wavelengths from SVO Filter Profile Service
2. Add only proven valuable features:
   - agn_probability (top feature from v56!)
   - peak_flux_g_over_r
   - g_peaks_last
3. Two versions: (a) all features, (b) top 200 features

Physics: The fitzpatrick99 extinction law models how dust absorbs/scatters
light as a function of wavelength. Bluer light is affected more than red.
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
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# =============================================================================
# EXTINCTION CORRECTION - Official Method from MALLORN notebook
# =============================================================================
# Effective wavelengths in Angstroms from SVO Filter Profile Service
EFFECTIVE_WAVELENGTHS = {
    'u': np.array([3641.0]),
    'g': np.array([4704.0]),
    'r': np.array([6155.0]),
    'i': np.array([7504.0]),
    'z': np.array([8695.0]),
    'y': np.array([10056.0])
}

# R_V = 3.1 is standard Milky Way value
R_V = 3.1

BANDS = ['u', 'g', 'r', 'i', 'z', 'y']

# Try to import extinction package
try:
    from extinction import fitzpatrick99
    HAS_EXTINCTION = True
    print("   [OK] extinction package loaded successfully")
except ImportError:
    HAS_EXTINCTION = False
    print("   [WARN] extinction package not found, will use fallback coefficients")
    # Fallback coefficients if extinction package not available
    # These are A_lambda/E(B-V) values for R_V=3.1
    FALLBACK_COEFFS = {
        'u': 4.81, 'g': 3.64, 'r': 2.70,
        'i': 2.06, 'z': 1.58, 'y': 1.31
    }


def get_extinction_mag(ebv, band):
    """
    Get extinction in magnitudes (A_lambda) for a given E(B-V) and band.
    Uses Fitzpatrick99 law with R_V = 3.1.
    """
    if pd.isna(ebv) or ebv <= 0:
        return 0.0

    if HAS_EXTINCTION:
        wavelength = EFFECTIVE_WAVELENGTHS[band]
        A_lambda = fitzpatrick99(wavelength, float(ebv) * R_V)
        return float(A_lambda[0])
    else:
        return float(ebv) * FALLBACK_COEFFS[band]


def correct_flux_for_extinction(flux, ebv, band):
    """
    Apply extinction correction to flux.

    Formula from MALLORN notebook:
        flux_corrected = flux * 10^(A_lambda / 2.5)

    This converts from observed (extincted) flux to intrinsic flux.
    """
    if pd.isna(flux) or pd.isna(ebv):
        return flux

    A_lambda = get_extinction_mag(ebv, band)
    correction_factor = 10 ** (A_lambda / 2.5)
    return flux * correction_factor


def correct_color_for_extinction(observed_color, ebv, band1, band2):
    """
    Correct observed color for extinction.

    Intrinsic_color = Observed_color - E(band1-band2)
    where E(band1-band2) = A_band1 - A_band2
    """
    if pd.isna(ebv) or pd.isna(observed_color):
        return observed_color

    A1 = get_extinction_mag(ebv, band1)
    A2 = get_extinction_mag(ebv, band2)
    color_excess = A1 - A2

    return observed_color - color_excess


def load_data():
    """Load lightcurve and metadata."""
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'

    train_meta = pd.read_csv(data_dir / 'train_log.csv')
    test_meta = pd.read_csv(data_dir / 'test_log.csv')

    # Load lightcurves
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


def add_extinction_corrected_color_features(features_df, meta_df):
    """
    Add extinction-corrected versions of color features using proper Fitzpatrick99.
    """
    # Merge EBV from metadata
    ebv_dict = dict(zip(meta_df['object_id'], meta_df['EBV']))

    # Find all color columns to correct
    color_pairs = [('g', 'r'), ('r', 'i'), ('u', 'g'), ('i', 'z')]

    new_features = {}

    for col in features_df.columns:
        for band1, band2 in color_pairs:
            color_key = f'{band1}_{band2}'
            if color_key in col and '_dered' not in col:
                # Create corrected version
                new_col_name = col.replace(color_key, f'{color_key}_dered')

                corrected_values = []
                for idx, row in features_df.iterrows():
                    obj_id = row['object_id']
                    ebv = ebv_dict.get(obj_id, 0)
                    observed = row[col]
                    corrected = correct_color_for_extinction(observed, ebv, band1, band2)
                    corrected_values.append(corrected)

                new_features[new_col_name] = corrected_values
                break

    # Add new columns
    for col_name, values in new_features.items():
        features_df[col_name] = values

    return features_df


def train_agn_classifier(X, meta_df):
    """Train AGN vs Rest classifier and return OOF probabilities."""
    y_agn = (meta_df['SpecType'] == 'AGN').astype(int).values
    oof_probs = np.zeros(len(X))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_agn)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y_agn[train_idx]

        model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0
        )
        model.fit(X_train, y_train)
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]

    # Train final model for test predictions
    final_model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, n_jobs=-1, verbosity=0
    )
    final_model.fit(X, y_agn)

    return oof_probs, final_model


def extract_peak_features_with_extinction(obj_id, lc_data, ebv):
    """Extract peak ordering features with proper extinction correction."""
    features = {'object_id': obj_id}

    obj_lc = lc_data[lc_data['object_id'] == obj_id]
    if obj_lc.empty:
        return features

    peak_times = {}
    peak_fluxes = {}

    for band in BANDS:
        band_lc = obj_lc[obj_lc['Filter'] == band]
        if len(band_lc) >= 3:
            fluxes = band_lc['Flux'].values.copy()
            times = band_lc['Time (MJD)'].values

            # Apply proper extinction correction
            if not pd.isna(ebv):
                fluxes = np.array([correct_flux_for_extinction(f, ebv, band) for f in fluxes])

            peak_idx = np.argmax(fluxes)
            peak_times[band] = times[peak_idx]
            peak_fluxes[band] = fluxes[peak_idx]

    # Feature 1: g_to_r_peak_delay
    if 'g' in peak_times and 'r' in peak_times:
        features['g_to_r_peak_delay'] = peak_times['r'] - peak_times['g']
    else:
        features['g_to_r_peak_delay'] = np.nan

    # Feature 2: peak_flux_g_over_r (extinction-corrected!)
    if 'g' in peak_fluxes and 'r' in peak_fluxes and peak_fluxes['r'] > 0:
        features['peak_flux_g_over_r'] = peak_fluxes['g'] / peak_fluxes['r']
    else:
        features['peak_flux_g_over_r'] = np.nan

    # Feature 3: g_peaks_last
    if len(peak_times) >= 2:
        sorted_bands = sorted(peak_times.keys(), key=lambda b: peak_times[b])
        features['g_peaks_last'] = 1 if sorted_bands[-1] == 'g' else 0
    else:
        features['g_peaks_last'] = np.nan

    return features


def main():
    print("=" * 80)
    print("MALLORN v57: Proper Fitzpatrick99 Extinction + Selective Features")
    print("=" * 80)

    # =========================================================================
    # 1. Load data
    # =========================================================================
    print("\n1. Loading data...")
    train_meta, test_meta, train_lc, test_lc = load_data()
    print(f"   Training objects: {len(train_meta)}")
    print(f"   Test objects: {len(test_meta)}")
    print(f"   TDEs: {train_meta['target'].sum()} ({100*train_meta['target'].mean():.1f}%)")

    # Show extinction impact
    print(f"\n   E(B-V) statistics:")
    print(f"   Train: mean={train_meta['EBV'].mean():.4f}, max={train_meta['EBV'].max():.4f}")

    # Example extinction values for median E(B-V)
    median_ebv = train_meta['EBV'].median()
    print(f"\n   Extinction at median E(B-V)={median_ebv:.3f}:")
    for band in BANDS:
        A = get_extinction_mag(median_ebv, band)
        print(f"      A_{band} = {A:.3f} mag")

    # =========================================================================
    # 2. Load cached baseline features (matching v34a structure)
    # =========================================================================
    print("\n2. Loading cached features...")
    cache_dir = Path(__file__).parent.parent / 'data' / 'processed'

    # Base statistical features (128 columns including object_id)
    with open(cache_dir / 'features_cache.pkl', 'rb') as f:
        base_cache = pickle.load(f)
    train_features = base_cache['train_features'].copy()
    test_features = base_cache['test_features'].copy()
    print(f"   Base statistical: {len(train_features.columns) - 1}")

    # Bazin features
    with open(cache_dir / 'bazin_features_cache.pkl', 'rb') as f:
        bazin_cache = pickle.load(f)
    train_features = train_features.merge(bazin_cache['train'], on='object_id', how='left')
    test_features = test_features.merge(bazin_cache['test'], on='object_id', how='left')
    print(f"   + Bazin: {len(bazin_cache['train'].columns) - 1}")

    # TDE physics features
    with open(cache_dir / 'tde_physics_cache.pkl', 'rb') as f:
        tde_cache = pickle.load(f)
    train_features = train_features.merge(tde_cache['train'], on='object_id', how='left')
    test_features = test_features.merge(tde_cache['test'], on='object_id', how='left')
    print(f"   + TDE physics: {len(tde_cache['train'].columns) - 1}")

    # Multiband GP features (used by v55)
    with open(cache_dir / 'multiband_gp_cache.pkl', 'rb') as f:
        gp_cache = pickle.load(f)
    train_features = train_features.merge(gp_cache['train'], on='object_id', how='left')
    test_features = test_features.merge(gp_cache['test'], on='object_id', how='left')
    print(f"   + Multiband GP: {len(gp_cache['train'].columns) - 1}")

    print(f"   = Total baseline: {len(train_features.columns) - 1}")

    # =========================================================================
    # 3. Apply proper extinction correction to color features
    # =========================================================================
    print("\n3. Applying Fitzpatrick99 extinction correction to colors...")

    train_features = add_extinction_corrected_color_features(train_features, train_meta)
    test_features = add_extinction_corrected_color_features(test_features, test_meta)

    dered_cols = [c for c in train_features.columns if '_dered' in c]
    print(f"   Added {len(dered_cols)} dereddened color features")

    # =========================================================================
    # 4. Train AGN classifier
    # =========================================================================
    print("\n4. Training AGN classifier...")

    baseline_cols = [c for c in base_cache['train_features'].columns if c != 'object_id']
    X_agn_train = train_features[baseline_cols].fillna(train_features[baseline_cols].median())
    X_agn_test = test_features[baseline_cols].fillna(X_agn_train.median())

    train_agn_probs, agn_model = train_agn_classifier(X_agn_train, train_meta)
    test_agn_probs = agn_model.predict_proba(X_agn_test)[:, 1]

    train_features['agn_probability'] = train_agn_probs
    test_features['agn_probability'] = test_agn_probs

    agn_acc = ((train_agn_probs > 0.5) == (train_meta['SpecType'] == 'AGN')).mean()
    print(f"   AGN classifier accuracy: {agn_acc:.4f}")

    # =========================================================================
    # 5. Extract peak features with extinction correction
    # =========================================================================
    print("\n5. Extracting peak features (extinction-corrected)...")

    train_ebv = dict(zip(train_meta['object_id'], train_meta['EBV']))
    test_ebv = dict(zip(test_meta['object_id'], test_meta['EBV']))

    train_peak_features = []
    for i, obj_id in enumerate(train_meta['object_id']):
        if (i + 1) % 500 == 0:
            print(f"   Train: {i+1}/{len(train_meta)}")
        feats = extract_peak_features_with_extinction(obj_id, train_lc, train_ebv.get(obj_id))
        train_peak_features.append(feats)

    test_peak_features = []
    for i, obj_id in enumerate(test_meta['object_id']):
        if (i + 1) % 1000 == 0:
            print(f"   Test: {i+1}/{len(test_meta)}")
        feats = extract_peak_features_with_extinction(obj_id, test_lc, test_ebv.get(obj_id))
        test_peak_features.append(feats)

    train_peak_df = pd.DataFrame(train_peak_features)
    test_peak_df = pd.DataFrame(test_peak_features)

    train_features = train_features.merge(train_peak_df, on='object_id', how='left')
    test_features = test_features.merge(test_peak_df, on='object_id', how='left')

    # =========================================================================
    # 6. Prepare features
    # =========================================================================
    print("\n6. Preparing features...")

    feature_cols = [c for c in train_features.columns if c != 'object_id']
    print(f"   Total features (v57a): {len(feature_cols)}")

    X_train = train_features[feature_cols].copy()
    X_test = test_features[feature_cols].copy()
    y_train = train_meta['target'].values

    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())

    # =========================================================================
    # 7. Train v57a (all features)
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"VERSION A: All {len(feature_cols)} features")
    print("=" * 80)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_a = np.zeros(len(X_train))
    test_preds_a = np.zeros(len(X_test))
    feature_importance_a = np.zeros(len(feature_cols))

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
        oof_preds_a[val_idx] = model.predict_proba(X_val)[:, 1]

        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.05, 0.5, 0.01):
            f1 = f1_score(y_val, (model.predict_proba(X_val)[:, 1] >= thresh).astype(int))
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        print(f"   Fold {fold+1}/5: F1={best_f1:.4f} @ thresh={best_thresh:.2f}")

        test_preds_a += model.predict_proba(X_test)[:, 1] / 5
        feature_importance_a += model.feature_importances_ / 5

    best_f1_a, best_thresh_a = 0, 0.5
    for thresh in np.arange(0.05, 0.5, 0.01):
        f1 = f1_score(y_train, (oof_preds_a >= thresh).astype(int))
        if f1 > best_f1_a:
            best_f1_a, best_thresh_a = f1, thresh

    print(f"\n   >>> v57a OOF F1: {best_f1_a:.4f} @ threshold={best_thresh_a:.2f}")

    # =========================================================================
    # 8. Feature selection for v57b (top 200)
    # =========================================================================
    print("\n" + "=" * 80)
    print("VERSION B: Top 200 features")
    print("=" * 80)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance_a
    }).sort_values('importance', ascending=False)

    top_200_features = importance_df.head(200)['feature'].tolist()
    X_train_b = X_train[top_200_features]
    X_test_b = X_test[top_200_features]

    oof_preds_b = np.zeros(len(X_train_b))
    test_preds_b = np.zeros(len(X_test_b))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_b, y_train)):
        X_tr, X_val = X_train_b.iloc[train_idx], X_train_b.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=len(y_tr[y_tr==0]) / max(1, len(y_tr[y_tr==1])),
            random_state=42 + fold, n_jobs=-1, verbosity=0
        )

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds_b[val_idx] = model.predict_proba(X_val)[:, 1]

        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.05, 0.5, 0.01):
            f1 = f1_score(y_val, (model.predict_proba(X_val)[:, 1] >= thresh).astype(int))
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh
        print(f"   Fold {fold+1}/5: F1={best_f1:.4f} @ thresh={best_thresh:.2f}")

        test_preds_b += model.predict_proba(X_test_b)[:, 1] / 5

    best_f1_b, best_thresh_b = 0, 0.5
    for thresh in np.arange(0.05, 0.5, 0.01):
        f1 = f1_score(y_train, (oof_preds_b >= thresh).astype(int))
        if f1 > best_f1_b:
            best_f1_b, best_thresh_b = f1, thresh

    print(f"\n   >>> v57b OOF F1: {best_f1_b:.4f} @ threshold={best_thresh_b:.2f}")

    # =========================================================================
    # 9. Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(f"\n   v34a baseline:     OOF F1=0.6667, LB F1=0.6907 (best)")
    print(f"   v55 powerlaw:      OOF F1=0.6751, LB F1=0.6873")
    print(f"   v56 AGN+peak:      OOF F1=0.6648")
    print(f"   v57a (all feat):   OOF F1={best_f1_a:.4f}")
    print(f"   v57b (top 200):    OOF F1={best_f1_b:.4f}")

    # =========================================================================
    # 10. Feature importance
    # =========================================================================
    print("\n" + "=" * 80)
    print("TOP 20 FEATURES")
    print("=" * 80)

    for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
        marker = ""
        if '_dered' in row['feature']:
            marker = "[DERED]"
        elif row['feature'] in ['agn_probability', 'g_to_r_peak_delay', 'peak_flux_g_over_r', 'g_peaks_last']:
            marker = "[NEW]"
        print(f"   #{i+1:2d}: {row['feature']:<40s} {row['importance']:>8.1f} {marker}")

    # New features specifically
    print("\n   New feature rankings:")
    new_feats = ['agn_probability', 'g_to_r_peak_delay', 'peak_flux_g_over_r', 'g_peaks_last']
    dered_feats = [c for c in feature_cols if '_dered' in c][:5]

    for feat in new_feats + dered_feats:
        if feat in list(importance_df['feature']):
            rank = list(importance_df['feature']).index(feat) + 1
            imp = importance_df[importance_df['feature'] == feat]['importance'].values[0]
            print(f"      {feat:<40s}: rank #{rank:3d}, importance={imp:.1f}")

    # =========================================================================
    # 11. Create submissions
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUBMISSIONS")
    print("=" * 80)

    sub_dir = Path(__file__).parent.parent / 'submissions'
    sub_dir.mkdir(exist_ok=True)

    sub_a = pd.DataFrame({
        'object_id': test_meta['object_id'],
        'target': (test_preds_a >= best_thresh_a).astype(int)
    })
    sub_a.to_csv(sub_dir / 'submission_v57a_extinction.csv', index=False)
    print(f"   v57a saved: {sub_a['target'].sum()} predicted TDEs")

    sub_b = pd.DataFrame({
        'object_id': test_meta['object_id'],
        'target': (test_preds_b >= best_thresh_b).astype(int)
    })
    sub_b.to_csv(sub_dir / 'submission_v57b_top200.csv', index=False)
    print(f"   v57b saved: {sub_b['target'].sum()} predicted TDEs")

    print("\n" + "=" * 80)
    print("COMPLETE - Ready for leaderboard testing!")
    print("=" * 80)


if __name__ == '__main__':
    main()
