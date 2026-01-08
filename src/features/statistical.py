"""
Statistical feature extraction for MALLORN lightcurve classification.

This module extracts per-band and aggregate statistics from lightcurve data.
These features form the baseline for gradient boosting models.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import warnings


def skewness(x):
    """Compute skewness without scipy."""
    n = len(x)
    if n < 3:
        return 0
    mean = np.mean(x)
    std = np.std(x, ddof=0)
    if std == 0:
        return 0
    return np.mean(((x - mean) / std) ** 3)


def kurtosis(x):
    """Compute excess kurtosis without scipy."""
    n = len(x)
    if n < 4:
        return 0
    mean = np.mean(x)
    std = np.std(x, ddof=0)
    if std == 0:
        return 0
    return np.mean(((x - mean) / std) ** 4) - 3

# LSST bands in wavelength order
LSST_BANDS = ["u", "g", "r", "i", "z", "y"]


def compute_band_statistics(flux: np.ndarray, flux_err: np.ndarray, times: np.ndarray) -> dict:
    """
    Compute statistical features for a single band's lightcurve.

    Args:
        flux: Array of flux values
        flux_err: Array of flux errors
        times: Array of observation times (MJD)

    Returns:
        Dictionary of computed features
    """
    features = {}
    n = len(flux)

    if n == 0:
        # Return NaN for missing bands
        return {
            'n_obs': 0,
            'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
            'median': np.nan, 'skew': np.nan, 'kurtosis': np.nan,
            'amplitude': np.nan, 'mad': np.nan, 'iqr': np.nan,
            'beyond_1std': np.nan, 'beyond_2std': np.nan,
            'max_slope': np.nan, 'mean_snr': np.nan,
            'time_span': np.nan, 'cadence_mean': np.nan
        }

    # Basic statistics
    features['n_obs'] = n
    features['mean'] = np.mean(flux)
    features['std'] = np.std(flux) if n > 1 else 0
    features['min'] = np.min(flux)
    features['max'] = np.max(flux)
    features['median'] = np.median(flux)

    # Higher moments
    if n > 2:
        features['skew'] = skewness(flux)
        features['kurtosis'] = kurtosis(flux)
    else:
        features['skew'] = 0
        features['kurtosis'] = 0

    # Amplitude and variability
    features['amplitude'] = features['max'] - features['min']
    features['mad'] = np.median(np.abs(flux - features['median']))
    features['iqr'] = np.percentile(flux, 75) - np.percentile(flux, 25) if n > 1 else 0

    # Beyond N sigma (fraction of points beyond N std from mean)
    if features['std'] > 0:
        z_scores = np.abs(flux - features['mean']) / features['std']
        features['beyond_1std'] = np.mean(z_scores > 1)
        features['beyond_2std'] = np.mean(z_scores > 2)
    else:
        features['beyond_1std'] = 0
        features['beyond_2std'] = 0

    # Maximum slope between consecutive observations
    if n > 1:
        sorted_idx = np.argsort(times)
        sorted_flux = flux[sorted_idx]
        sorted_times = times[sorted_idx]
        dt = np.diff(sorted_times)
        df = np.diff(sorted_flux)
        # Avoid division by zero
        valid = dt > 0
        if np.any(valid):
            slopes = np.abs(df[valid] / dt[valid])
            features['max_slope'] = np.max(slopes)
        else:
            features['max_slope'] = 0
    else:
        features['max_slope'] = 0

    # Signal-to-noise ratio
    valid_err = flux_err > 0
    if np.any(valid_err):
        features['mean_snr'] = np.mean(np.abs(flux[valid_err]) / flux_err[valid_err])
    else:
        features['mean_snr'] = np.nan

    # Time-based features
    if n > 1:
        features['time_span'] = np.max(times) - np.min(times)
        sorted_times = np.sort(times)
        cadences = np.diff(sorted_times)
        features['cadence_mean'] = np.mean(cadences)
    else:
        features['time_span'] = 0
        features['cadence_mean'] = 0

    return features


def extract_statistical_features(
    lightcurves: pd.DataFrame,
    object_ids: Optional[List[str]] = None,
    bands: List[str] = LSST_BANDS
) -> pd.DataFrame:
    """
    Extract statistical features for all objects.

    Args:
        lightcurves: DataFrame with columns [object_id, Time (MJD), Flux, Flux_err, Filter]
        object_ids: Optional list of object IDs to process (defaults to all)
        bands: List of bands to extract features for

    Returns:
        DataFrame with one row per object and columns for each feature
    """
    if object_ids is None:
        object_ids = lightcurves['object_id'].unique()

    # Pre-group for O(1) lookup per object
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if (i + 1) % 500 == 0:
            print(f"    Stats: {i+1}/{len(object_ids)} objects processed")

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue
        obj_features = {'object_id': obj_id}

        # Per-band features
        for band in bands:
            band_data = obj_lc[obj_lc['Filter'] == band]

            if len(band_data) > 0:
                flux = band_data['Flux'].values
                flux_err = band_data['Flux_err'].values
                times = band_data['Time (MJD)'].values
                band_stats = compute_band_statistics(flux, flux_err, times)
            else:
                band_stats = compute_band_statistics(
                    np.array([]), np.array([]), np.array([])
                )

            for feat_name, feat_val in band_stats.items():
                obj_features[f'{band}_{feat_name}'] = feat_val

        # Aggregate features across all bands
        all_flux = obj_lc['Flux'].values
        all_flux_err = obj_lc['Flux_err'].values
        all_times = obj_lc['Time (MJD)'].values

        agg_stats = compute_band_statistics(all_flux, all_flux_err, all_times)
        for feat_name, feat_val in agg_stats.items():
            obj_features[f'all_{feat_name}'] = feat_val

        # Cross-band features
        band_means = {band: obj_features.get(f'{band}_mean', np.nan) for band in bands}
        band_maxes = {band: obj_features.get(f'{band}_max', np.nan) for band in bands}

        # Color-like ratios (flux ratios between bands)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not np.isnan(band_means['g']) and band_means['r'] > 0:
                obj_features['flux_ratio_g_r'] = band_means['g'] / band_means['r']
            else:
                obj_features['flux_ratio_g_r'] = np.nan

            if not np.isnan(band_means['r']) and band_means['i'] > 0:
                obj_features['flux_ratio_r_i'] = band_means['r'] / band_means['i']
            else:
                obj_features['flux_ratio_r_i'] = np.nan

            if not np.isnan(band_means['i']) and band_means['z'] > 0:
                obj_features['flux_ratio_i_z'] = band_means['i'] / band_means['z']
            else:
                obj_features['flux_ratio_i_z'] = np.nan

        # Peak band (which band has maximum flux)
        valid_maxes = {b: m for b, m in band_maxes.items() if not np.isnan(m)}
        if valid_maxes:
            peak_band = max(valid_maxes, key=valid_maxes.get)
            obj_features['peak_band'] = bands.index(peak_band)
        else:
            obj_features['peak_band'] = -1

        all_features.append(obj_features)

    return pd.DataFrame(all_features)


def add_metadata_features(
    features: pd.DataFrame,
    metadata: pd.DataFrame
) -> pd.DataFrame:
    """
    Add metadata features (redshift, extinction) to extracted features.

    Args:
        features: DataFrame from extract_statistical_features
        metadata: DataFrame with object_id, Z, EBV columns

    Returns:
        Merged DataFrame with additional features
    """
    result = features.merge(
        metadata[['object_id', 'Z', 'EBV']],
        on='object_id',
        how='left'
    )

    # Redshift-derived features
    result['luminosity_distance'] = result['Z'] * 4280  # Approximate Mpc for z < 1
    result['time_dilation'] = 1 + result['Z']

    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_all_data

    print("Loading data...")
    data = load_all_data()

    print("Extracting features for first 10 objects...")
    sample_ids = data['train_meta']['object_id'].head(10).tolist()
    features = extract_statistical_features(data['train_lc'], sample_ids)

    print(f"\nExtracted {len(features.columns)} features for {len(features)} objects")
    print("\nFeature columns:")
    print(features.columns.tolist()[:20], "...")
    print("\nSample:")
    print(features.head())
