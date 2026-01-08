"""
Lightcurve shape feature extraction for MALLORN classification.

These features capture the temporal evolution of transients:
- Rise time to peak
- Fade time from peak
- Asymmetry (rise/fade ratio)
- Decay rate characteristics
- Duration above flux thresholds

Physics motivation:
- TDEs: t^(-5/3) power-law decay from mass fallback
- SNe Ia: ~17 day rise, exponential decay from Co-56
- Core-collapse SNe: Variable, often with plateau
- AGN: Stochastic, no characteristic timescales
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
import warnings

LSST_BANDS = ["u", "g", "r", "i", "z", "y"]


def find_peak_info(times: np.ndarray, fluxes: np.ndarray) -> Tuple[float, float, int]:
    """Find peak time, flux, and index."""
    if len(times) == 0:
        return np.nan, np.nan, -1
    peak_idx = np.argmax(fluxes)
    return times[peak_idx], fluxes[peak_idx], peak_idx


def compute_rise_time(times: np.ndarray, fluxes: np.ndarray,
                      peak_time: float, peak_flux: float,
                      threshold_frac: float = 0.1) -> float:
    """
    Compute rise time from threshold to peak.

    Args:
        threshold_frac: Fraction of peak flux to use as detection threshold
    """
    if np.isnan(peak_time) or np.isnan(peak_flux) or len(times) < 2:
        return np.nan

    # Find observations before peak
    pre_peak_mask = times < peak_time
    if not np.any(pre_peak_mask):
        return np.nan

    pre_times = times[pre_peak_mask]
    pre_fluxes = fluxes[pre_peak_mask]

    # Find first time flux exceeds threshold
    threshold = threshold_frac * peak_flux
    above_threshold = pre_fluxes > threshold

    if not np.any(above_threshold):
        # Use first observation as proxy
        return peak_time - pre_times[0]

    first_detection_idx = np.argmax(above_threshold)
    first_detection_time = pre_times[first_detection_idx]

    return peak_time - first_detection_time


def compute_fade_time(times: np.ndarray, fluxes: np.ndarray,
                      peak_time: float, peak_flux: float,
                      fade_frac: float = 0.5) -> float:
    """
    Compute time to fade to fraction of peak flux.

    Args:
        fade_frac: Fraction of peak to measure fade time to (default 0.5 = half-peak)
    """
    if np.isnan(peak_time) or np.isnan(peak_flux) or len(times) < 2:
        return np.nan

    # Find observations after peak
    post_peak_mask = times > peak_time
    if not np.any(post_peak_mask):
        return np.nan

    post_times = times[post_peak_mask]
    post_fluxes = fluxes[post_peak_mask]

    # Sort by time
    sort_idx = np.argsort(post_times)
    post_times = post_times[sort_idx]
    post_fluxes = post_fluxes[sort_idx]

    # Find first time flux drops below threshold
    threshold = fade_frac * peak_flux
    below_threshold = post_fluxes < threshold

    if not np.any(below_threshold):
        # Never fades to threshold - use last observation
        return post_times[-1] - peak_time

    fade_idx = np.argmax(below_threshold)
    fade_time = post_times[fade_idx]

    return fade_time - peak_time


def fit_power_law_decay(times: np.ndarray, fluxes: np.ndarray,
                        peak_time: float, peak_flux: float) -> Tuple[float, float]:
    """
    Fit power-law f(t) = A * (t - t0)^alpha to post-peak decay.

    Returns:
        (alpha, fit_residual): Power-law index and goodness of fit
    """
    if np.isnan(peak_time) or np.isnan(peak_flux):
        return np.nan, np.nan

    # Get post-peak data
    post_mask = (times > peak_time + 5) & (fluxes > 0)  # Start 5 days after peak
    if np.sum(post_mask) < 5:
        return np.nan, np.nan

    post_times = times[post_mask]
    post_fluxes = fluxes[post_mask]

    # Transform to log space: log(f) = log(A) + alpha * log(t - t0)
    dt = post_times - peak_time
    dt = np.maximum(dt, 1.0)  # Avoid log(0)

    log_dt = np.log10(dt)
    log_flux = np.log10(np.maximum(post_fluxes, 1e-10))

    # Simple linear regression in log space
    try:
        coeffs = np.polyfit(log_dt, log_flux, 1)
        alpha = coeffs[0]  # Power-law index

        # Compute residual
        predicted = coeffs[0] * log_dt + coeffs[1]
        residual = np.sqrt(np.mean((log_flux - predicted) ** 2))

        return alpha, residual
    except:
        return np.nan, np.nan


def compute_duration_above_threshold(times: np.ndarray, fluxes: np.ndarray,
                                     threshold_frac: float = 0.5) -> float:
    """Compute duration flux stays above fraction of peak."""
    if len(times) < 2:
        return np.nan

    peak_flux = np.max(fluxes)
    threshold = threshold_frac * peak_flux

    above = fluxes > threshold
    if not np.any(above):
        return 0

    above_times = times[above]
    return np.max(above_times) - np.min(above_times)


def compute_flux_percentiles(fluxes: np.ndarray) -> Dict[str, float]:
    """Compute flux percentiles for shape characterization."""
    if len(fluxes) == 0:
        return {'p10': np.nan, 'p25': np.nan, 'p75': np.nan, 'p90': np.nan}

    return {
        'p10': np.percentile(fluxes, 10),
        'p25': np.percentile(fluxes, 25),
        'p75': np.percentile(fluxes, 75),
        'p90': np.percentile(fluxes, 90)
    }


def extract_shape_features_single(obj_lc: pd.DataFrame) -> Dict[str, float]:
    """
    Extract lightcurve shape features for a single object.

    Args:
        obj_lc: DataFrame with columns [Time (MJD), Flux, Flux_err, Filter]

    Returns:
        Dictionary of shape features
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

    # Per-band shape features
    peak_times = {}
    peak_fluxes = {}

    for band in LSST_BANDS:
        if band not in band_data:
            features[f'{band}_rise_time'] = np.nan
            features[f'{band}_fade_time_50'] = np.nan
            features[f'{band}_fade_time_25'] = np.nan
            features[f'{band}_asymmetry'] = np.nan
            features[f'{band}_duration_50'] = np.nan
            features[f'{band}_duration_25'] = np.nan
            features[f'{band}_power_law_alpha'] = np.nan
            features[f'{band}_power_law_residual'] = np.nan
            continue

        times = band_data[band]['times']
        fluxes = band_data[band]['fluxes']

        # Peak info
        peak_time, peak_flux, _ = find_peak_info(times, fluxes)
        peak_times[band] = peak_time
        peak_fluxes[band] = peak_flux

        # Rise time
        rise_time = compute_rise_time(times, fluxes, peak_time, peak_flux)
        features[f'{band}_rise_time'] = rise_time

        # Fade times (to 50% and 25% of peak)
        fade_50 = compute_fade_time(times, fluxes, peak_time, peak_flux, 0.5)
        fade_25 = compute_fade_time(times, fluxes, peak_time, peak_flux, 0.25)
        features[f'{band}_fade_time_50'] = fade_50
        features[f'{band}_fade_time_25'] = fade_25

        # Asymmetry (rise/fade ratio)
        if not np.isnan(rise_time) and not np.isnan(fade_50) and fade_50 > 0:
            features[f'{band}_asymmetry'] = rise_time / fade_50
        else:
            features[f'{band}_asymmetry'] = np.nan

        # Duration above thresholds
        features[f'{band}_duration_50'] = compute_duration_above_threshold(times, fluxes, 0.5)
        features[f'{band}_duration_25'] = compute_duration_above_threshold(times, fluxes, 0.25)

        # Power-law decay fit
        alpha, residual = fit_power_law_decay(times, fluxes, peak_time, peak_flux)
        features[f'{band}_power_law_alpha'] = alpha
        features[f'{band}_power_law_residual'] = residual

    # Cross-band features

    # Peak time spread (do all bands peak at similar times?)
    valid_peak_times = [t for t in peak_times.values() if not np.isnan(t)]
    if len(valid_peak_times) >= 2:
        features['peak_time_spread'] = np.max(valid_peak_times) - np.min(valid_peak_times)
        features['peak_time_std'] = np.std(valid_peak_times)
    else:
        features['peak_time_spread'] = np.nan
        features['peak_time_std'] = np.nan

    # Average shape features across optical bands (g, r, i)
    optical_bands = ['g', 'r', 'i']

    rise_times = [features.get(f'{b}_rise_time', np.nan) for b in optical_bands]
    fade_times = [features.get(f'{b}_fade_time_50', np.nan) for b in optical_bands]
    alphas = [features.get(f'{b}_power_law_alpha', np.nan) for b in optical_bands]

    valid_rises = [r for r in rise_times if not np.isnan(r)]
    valid_fades = [f for f in fade_times if not np.isnan(f)]
    valid_alphas = [a for a in alphas if not np.isnan(a)]

    features['optical_mean_rise_time'] = np.mean(valid_rises) if valid_rises else np.nan
    features['optical_mean_fade_time'] = np.mean(valid_fades) if valid_fades else np.nan
    features['optical_mean_power_alpha'] = np.mean(valid_alphas) if valid_alphas else np.nan

    # Rise/fade consistency across bands
    if len(valid_rises) >= 2:
        features['rise_time_consistency'] = np.std(valid_rises) / (np.mean(valid_rises) + 1e-6)
    else:
        features['rise_time_consistency'] = np.nan

    if len(valid_fades) >= 2:
        features['fade_time_consistency'] = np.std(valid_fades) / (np.mean(valid_fades) + 1e-6)
    else:
        features['fade_time_consistency'] = np.nan

    # Overall shape from all observations
    all_times = obj_lc['Time (MJD)'].values
    all_fluxes = obj_lc['Flux'].values

    if len(all_times) >= 5:
        peak_time_all, peak_flux_all, _ = find_peak_info(all_times, all_fluxes)

        features['all_rise_time'] = compute_rise_time(all_times, all_fluxes, peak_time_all, peak_flux_all)
        features['all_fade_time_50'] = compute_fade_time(all_times, all_fluxes, peak_time_all, peak_flux_all, 0.5)

        if not np.isnan(features['all_rise_time']) and not np.isnan(features['all_fade_time_50']):
            if features['all_fade_time_50'] > 0:
                features['all_asymmetry'] = features['all_rise_time'] / features['all_fade_time_50']
            else:
                features['all_asymmetry'] = np.nan
        else:
            features['all_asymmetry'] = np.nan

        alpha, residual = fit_power_law_decay(all_times, all_fluxes, peak_time_all, peak_flux_all)
        features['all_power_law_alpha'] = alpha
        features['all_power_law_residual'] = residual

        # Flux distribution shape
        percentiles = compute_flux_percentiles(all_fluxes)
        features['flux_p10'] = percentiles['p10']
        features['flux_p25'] = percentiles['p25']
        features['flux_p75'] = percentiles['p75']
        features['flux_p90'] = percentiles['p90']

        # Concentration: what fraction of flux is in the peak?
        if peak_flux_all > 0:
            features['flux_concentration'] = peak_flux_all / (np.sum(all_fluxes) + 1e-6)
        else:
            features['flux_concentration'] = np.nan
    else:
        features['all_rise_time'] = np.nan
        features['all_fade_time_50'] = np.nan
        features['all_asymmetry'] = np.nan
        features['all_power_law_alpha'] = np.nan
        features['all_power_law_residual'] = np.nan
        features['flux_p10'] = np.nan
        features['flux_p25'] = np.nan
        features['flux_p75'] = np.nan
        features['flux_p90'] = np.nan
        features['flux_concentration'] = np.nan

    return features


def extract_shape_features(
    lightcurves: pd.DataFrame,
    object_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract shape features for multiple objects.

    Args:
        lightcurves: DataFrame with columns [object_id, Time (MJD), Flux, Flux_err, Filter]
        object_ids: Optional list of object IDs to process

    Returns:
        DataFrame with one row per object and shape feature columns
    """
    if object_ids is None:
        object_ids = lightcurves['object_id'].unique()

    # Pre-group for O(1) lookup per object
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if (i + 1) % 500 == 0:
            print(f"    Shapes: {i+1}/{len(object_ids)} objects processed")

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue
        features = extract_shape_features_single(obj_lc)
        features['object_id'] = obj_id
        all_features.append(features)

    return pd.DataFrame(all_features)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_all_data

    print("Loading data...")
    data = load_all_data()

    print("\nExtracting shape features for first 20 objects...")
    sample_ids = data['train_meta']['object_id'].head(20).tolist()
    shape_features = extract_shape_features(data['train_lc'], sample_ids)

    print(f"\nExtracted {len(shape_features.columns)-1} shape features")
    print("\nFeature columns:")
    print([c for c in shape_features.columns if c != 'object_id'])
    print("\nSample values (r-band):")
    r_cols = [c for c in shape_features.columns if c.startswith('r_') or c.startswith('all_')]
    print(shape_features[['object_id'] + r_cols[:8]].head())
