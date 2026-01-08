"""
Cesium-Style Astronomy Features for MALLORN.

Port of key features from the Cesium library used by PLAsTiCC winners.
These features characterize lightcurve variability, shape, and temporal patterns.

References:
- Cesium: https://github.com/cesium-ml/cesium
- PLAsTiCC 1st place: Used extensive Cesium features
- PLAsTiCC 2nd place: Stetson indices, percentile ratios
- PLAsTiCC 3rd place: Beyond-n-std features

Key Features Implemented:
1. Stetson J, K: Correlated variability across bands
2. Beyond N-std: Outlier detection (fraction of points beyond σ thresholds)
3. Flux percentile ratios: Shape characterization
4. Maximum slope: Fastest rise/fall rate
5. Linear trend: Overall flux evolution
6. Percent amplitude: Peak-to-median ratio
7. Anderson-Darling: Normality test for flux distribution
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Optional

LSST_BANDS = ["u", "g", "r", "i", "z", "y"]


def stetson_j(times: np.ndarray, fluxes: np.ndarray, flux_errors: np.ndarray) -> float:
    """
    Stetson J statistic - measures correlated variability within a single band.

    J = sum( w_i * δ_i * sign(δ_i) ) / sum(w_i)
    where δ_i = sqrt(n/(n-1)) * (flux_i - mean) / error_i

    High J indicates real variability (not just noise).

    Args:
        times: Observation times (MJD)
        fluxes: Flux measurements
        flux_errors: Flux uncertainties

    Returns:
        Stetson J statistic (typically 0-5, higher = more variable)
    """
    if len(fluxes) < 2:
        return np.nan

    n = len(fluxes)
    mean_flux = np.mean(fluxes)

    # Normalized residuals
    delta = np.sqrt(n / (n - 1)) * (fluxes - mean_flux) / np.where(flux_errors > 0, flux_errors, 1.0)

    # Weights (inverse variance)
    weights = 1.0 / np.where(flux_errors > 0, flux_errors**2, 1.0)

    # Stetson J: weighted sum of signed squared residuals
    numerator = np.sum(weights * delta * np.sign(delta))
    denominator = np.sum(weights)

    if denominator == 0:
        return np.nan

    return numerator / denominator


def stetson_k(times: np.ndarray, fluxes: np.ndarray, flux_errors: np.ndarray) -> float:
    """
    Stetson K statistic - measures kurtosis of residual distribution.

    K = (1/N) * sum( |δ_i| ) / sqrt( (1/N) * sum(δ_i²) )

    K ~ 0.798 for normally distributed noise (Gaussian).
    K > 0.798 indicates peaked distribution (outliers).
    K < 0.798 indicates flat distribution.

    Args:
        times: Observation times
        fluxes: Flux measurements
        flux_errors: Flux uncertainties

    Returns:
        Stetson K statistic (typically 0.5-1.5)
    """
    if len(fluxes) < 2:
        return np.nan

    n = len(fluxes)
    mean_flux = np.mean(fluxes)

    # Normalized residuals
    delta = np.sqrt(n / (n - 1)) * (fluxes - mean_flux) / np.where(flux_errors > 0, flux_errors, 1.0)

    numerator = np.mean(np.abs(delta))
    denominator = np.sqrt(np.mean(delta**2))

    if denominator == 0:
        return np.nan

    return numerator / denominator


def beyond_n_std(fluxes: np.ndarray, n_std: float = 1.0) -> float:
    """
    Fraction of observations beyond n standard deviations from the mean.

    For Gaussian noise:
    - beyond_1_std ~ 0.32 (32% outside ±1σ)
    - beyond_2_std ~ 0.05 (5% outside ±2σ)

    Higher values indicate real variability or outliers.

    Args:
        fluxes: Flux measurements
        n_std: Number of standard deviations (default 1.0)

    Returns:
        Fraction of points beyond n*std from mean
    """
    if len(fluxes) < 3:
        return np.nan

    mean = np.mean(fluxes)
    std = np.std(fluxes)

    if std == 0:
        return 0.0

    deviations = np.abs(fluxes - mean) / std
    fraction = np.sum(deviations > n_std) / len(fluxes)

    return fraction


def flux_percentile_ratio(fluxes: np.ndarray, lower: int, upper: int) -> float:
    """
    Ratio of flux percentiles, characterizes lightcurve shape.

    Common ratios:
    - flux_percentile_ratio_mid20: (60th - 40th) / (95th - 5th)
    - flux_percentile_ratio_mid35: (67.5th - 32.5th) / (95th - 5th)
    - flux_percentile_ratio_mid50: (75th - 25th) / (95th - 5th)
    - flux_percentile_ratio_mid65: (82.5th - 17.5th) / (95th - 5th)
    - flux_percentile_ratio_mid80: (90th - 10th) / (95th - 5th)

    Ratio close to 1 → symmetric distribution.
    Ratio < 1 → concentrated around median (peaked).

    Args:
        fluxes: Flux measurements
        lower: Lower percentile (e.g., 40 for mid20)
        upper: Upper percentile (e.g., 60 for mid20)

    Returns:
        Percentile ratio
    """
    if len(fluxes) < 5:
        return np.nan

    p_lower = np.percentile(fluxes, lower)
    p_upper = np.percentile(fluxes, upper)
    p_5 = np.percentile(fluxes, 5)
    p_95 = np.percentile(fluxes, 95)

    denominator = p_95 - p_5

    if denominator == 0:
        return np.nan

    return (p_upper - p_lower) / denominator


def percent_amplitude(fluxes: np.ndarray) -> float:
    """
    Percent amplitude: (max - median) / median.

    Measures peak brightness relative to typical flux.
    High for transients with strong peaks.

    Args:
        fluxes: Flux measurements

    Returns:
        Percent amplitude (fractional)
    """
    if len(fluxes) < 2:
        return np.nan

    median_flux = np.median(fluxes)

    if median_flux == 0:
        return np.nan

    return (np.max(fluxes) - median_flux) / np.abs(median_flux)


def maximum_slope(times: np.ndarray, fluxes: np.ndarray) -> float:
    """
    Maximum absolute slope between consecutive observations.

    Slope = |Δflux| / Δtime

    High for rapid transients (TDEs, SNe).
    Low for slow variables (AGN).

    Args:
        times: Observation times (MJD)
        fluxes: Flux measurements

    Returns:
        Maximum slope (flux units per day)
    """
    if len(times) < 2:
        return np.nan

    # Sort by time
    sort_idx = np.argsort(times)
    times_sorted = times[sort_idx]
    fluxes_sorted = fluxes[sort_idx]

    # Compute slopes between consecutive points
    delta_time = np.diff(times_sorted)
    delta_flux = np.diff(fluxes_sorted)

    # Avoid division by zero
    delta_time = np.where(delta_time > 0, delta_time, 1.0)

    slopes = np.abs(delta_flux / delta_time)

    return np.max(slopes)


def linear_trend(times: np.ndarray, fluxes: np.ndarray, flux_errors: np.ndarray) -> float:
    """
    Slope of weighted linear fit to lightcurve.

    Positive → rising over time.
    Negative → declining over time.
    Near zero → stable or oscillating.

    Args:
        times: Observation times (MJD)
        fluxes: Flux measurements
        flux_errors: Flux uncertainties

    Returns:
        Linear trend slope (flux units per day)
    """
    if len(times) < 3:
        return np.nan

    # Weights (inverse variance)
    weights = 1.0 / np.where(flux_errors > 0, flux_errors**2, 1.0)

    # Weighted linear regression
    try:
        # Normalize times for numerical stability
        t_mean = np.mean(times)
        t = times - t_mean

        # Weighted least squares: flux = a + b*t
        w_sum = np.sum(weights)
        t_weighted = np.sum(weights * t) / w_sum
        flux_weighted = np.sum(weights * fluxes) / w_sum

        numerator = np.sum(weights * (t - t_weighted) * (fluxes - flux_weighted))
        denominator = np.sum(weights * (t - t_weighted)**2)

        if denominator == 0:
            return np.nan

        slope = numerator / denominator
        return slope

    except:
        return np.nan


def anderson_darling_statistic(fluxes: np.ndarray) -> float:
    """
    Anderson-Darling test for normality.

    Tests if flux distribution is Gaussian.
    Higher values → non-Gaussian (real variability, not noise).

    Args:
        fluxes: Flux measurements

    Returns:
        Anderson-Darling statistic
    """
    if len(fluxes) < 5:
        return np.nan

    try:
        # Standardize fluxes
        fluxes_norm = (fluxes - np.mean(fluxes)) / np.std(fluxes)

        # Anderson-Darling test
        result = stats.anderson(fluxes_norm, dist='norm')

        return result.statistic

    except:
        return np.nan


def extract_cesium_features_single_band(times: np.ndarray, fluxes: np.ndarray,
                                        flux_errors: np.ndarray) -> Dict[str, float]:
    """
    Extract all Cesium features for a single band.

    Args:
        times: Observation times (MJD)
        fluxes: Flux measurements
        flux_errors: Flux uncertainties

    Returns:
        Dictionary of Cesium features
    """
    features = {}

    # Stetson statistics
    features['cesium_stetson_j'] = stetson_j(times, fluxes, flux_errors)
    features['cesium_stetson_k'] = stetson_k(times, fluxes, flux_errors)

    # Beyond N-std features
    features['cesium_beyond_1std'] = beyond_n_std(fluxes, n_std=1.0)
    features['cesium_beyond_2std'] = beyond_n_std(fluxes, n_std=2.0)

    # Flux percentile ratios
    features['cesium_flux_percentile_ratio_mid20'] = flux_percentile_ratio(fluxes, 40, 60)
    features['cesium_flux_percentile_ratio_mid35'] = flux_percentile_ratio(fluxes, 32.5, 67.5)
    features['cesium_flux_percentile_ratio_mid50'] = flux_percentile_ratio(fluxes, 25, 75)
    features['cesium_flux_percentile_ratio_mid65'] = flux_percentile_ratio(fluxes, 17.5, 82.5)
    features['cesium_flux_percentile_ratio_mid80'] = flux_percentile_ratio(fluxes, 10, 90)

    # Shape features
    features['cesium_percent_amplitude'] = percent_amplitude(fluxes)
    features['cesium_maximum_slope'] = maximum_slope(times, fluxes)
    features['cesium_linear_trend'] = linear_trend(times, fluxes, flux_errors)

    # Distribution test
    features['cesium_anderson_darling'] = anderson_darling_statistic(fluxes)

    return features


def extract_cesium_features_single(obj_lc: pd.DataFrame) -> Dict[str, float]:
    """
    Extract Cesium features for a single object across all bands.

    Args:
        obj_lc: Lightcurve DataFrame for one object

    Returns:
        Dictionary with Cesium features for all bands
    """
    features = {}

    for band in LSST_BANDS:
        band_lc = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_lc) < 5:
            # Insufficient data - fill with NaN
            for feat_name in ['cesium_stetson_j', 'cesium_stetson_k',
                             'cesium_beyond_1std', 'cesium_beyond_2std',
                             'cesium_flux_percentile_ratio_mid20', 'cesium_flux_percentile_ratio_mid35',
                             'cesium_flux_percentile_ratio_mid50', 'cesium_flux_percentile_ratio_mid65',
                             'cesium_flux_percentile_ratio_mid80', 'cesium_percent_amplitude',
                             'cesium_maximum_slope', 'cesium_linear_trend', 'cesium_anderson_darling']:
                features[f'{band}_{feat_name}'] = np.nan
            continue

        times = band_lc['Time (MJD)'].values
        fluxes = band_lc['Flux'].values
        flux_errors = band_lc['Flux_err'].values

        # Extract features for this band
        band_features = extract_cesium_features_single_band(times, fluxes, flux_errors)

        # Add band prefix
        for key, val in band_features.items():
            features[f'{band}_{key}'] = val

    # Cross-band features
    # Stetson J consistency across bands (similar to Bazin consistency)
    stetson_j_values = []
    for band in ['g', 'r', 'i']:  # Focus on well-sampled bands
        key = f'{band}_cesium_stetson_j'
        if key in features and not np.isnan(features[key]):
            stetson_j_values.append(features[key])

    if len(stetson_j_values) >= 2:
        features['cesium_stetson_j_consistency'] = np.std(stetson_j_values) / np.mean(np.abs(stetson_j_values))
    else:
        features['cesium_stetson_j_consistency'] = np.nan

    # Average variability across bands
    beyond_1std_values = []
    for band in LSST_BANDS:
        key = f'{band}_cesium_beyond_1std'
        if key in features and not np.isnan(features[key]):
            beyond_1std_values.append(features[key])

    if len(beyond_1std_values) > 0:
        features['cesium_avg_beyond_1std'] = np.mean(beyond_1std_values)
    else:
        features['cesium_avg_beyond_1std'] = np.nan

    return features


def extract_cesium_features(
    lightcurves: pd.DataFrame,
    object_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract Cesium features for multiple objects.

    Args:
        lightcurves: DataFrame with lightcurve data
        object_ids: Optional list of object IDs

    Returns:
        DataFrame with Cesium features
    """
    if object_ids is None:
        object_ids = lightcurves['object_id'].unique()

    # Pre-group for efficiency
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if (i + 1) % 500 == 0:
            print(f"    Cesium: {i+1}/{len(object_ids)} objects processed")

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue

        features = extract_cesium_features_single(obj_lc)
        features['object_id'] = obj_id
        all_features.append(features)

    return pd.DataFrame(all_features)


if __name__ == "__main__":
    # Test Cesium features
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_all_data

    print("Loading data...")
    data = load_all_data()

    print("\nTesting Cesium features on first 10 objects...")
    sample_ids = data['train_meta']['object_id'].head(10).tolist()
    cesium_features = extract_cesium_features(data['train_lc'], sample_ids)

    print(f"\nExtracted {len(cesium_features.columns)-1} Cesium features")
    print("\nFeature columns (first 20):")
    print([c for c in cesium_features.columns if c != 'object_id'][:20])

    print("\nSample values for r-band:")
    r_cols = [c for c in cesium_features.columns if c.startswith('r_cesium')]
    print(cesium_features[['object_id'] + r_cols[:5]].head())

    # Check feature coverage
    print("\nFeature coverage (non-NaN):")
    for col in r_cols[:5]:
        coverage = cesium_features[col].notna().sum() / len(cesium_features)
        print(f"  {col}: {100*coverage:.1f}%")
