"""
Time-to-Decline Features for TDE Classification

Based on PLAsTiCC 1st place (Kyle Boone):
- "computed the time to decline to 20% of the maximum light of the transient"
- This was THE key feature that gave Boone 1st place

Physics basis:
- TDEs: Slow decline (months), low decline velocity
- SNe: Fast decline (weeks), high decline velocity
- AGN: Stochastic, no clear decline pattern

Features:
- Time from peak to [80%, 60%, 40%, 20%, 10%] of peak flux
- Per band (u, g, r, i, z, y) = 30 features
- Decline velocity: flux drop per unit time
"""

import numpy as np
import pandas as pd
from typing import List, Dict


def find_peak_time_and_flux(times: np.ndarray, fluxes: np.ndarray) -> tuple:
    """
    Find time and flux at peak brightness.

    Args:
        times: Observation times (MJD)
        fluxes: Flux measurements

    Returns:
        (peak_time, peak_flux) tuple, or (np.nan, np.nan) if insufficient data
    """
    if len(times) < 3 or len(fluxes) < 3:
        return np.nan, np.nan

    # Find peak flux
    peak_idx = np.argmax(fluxes)
    peak_time = times[peak_idx]
    peak_flux = fluxes[peak_idx]

    return peak_time, peak_flux


def compute_time_to_decline(times: np.ndarray, fluxes: np.ndarray,
                            peak_time: float, peak_flux: float,
                            threshold_fraction: float) -> float:
    """
    Compute time from peak to when flux declines to threshold fraction of peak.

    Args:
        times: Observation times (MJD)
        fluxes: Flux measurements
        peak_time: Time of peak brightness
        peak_flux: Flux at peak
        threshold_fraction: Fraction of peak flux (e.g., 0.2 for 20%)

    Returns:
        Time (in days) from peak to threshold, or np.nan if not reached
    """
    # Only consider observations after peak
    post_peak_mask = times > peak_time

    if not np.any(post_peak_mask):
        return np.nan

    times_post = times[post_peak_mask]
    fluxes_post = fluxes[post_peak_mask]

    # Sort by time
    sort_idx = np.argsort(times_post)
    times_sorted = times_post[sort_idx]
    fluxes_sorted = fluxes_post[sort_idx]

    # Target flux
    target_flux = peak_flux * threshold_fraction

    # Find first time flux drops below target
    below_threshold = fluxes_sorted < target_flux

    if not np.any(below_threshold):
        return np.nan

    # Get first crossing
    first_crossing_idx = np.where(below_threshold)[0][0]

    # If we have observations on both sides of threshold, interpolate
    if first_crossing_idx > 0:
        # Linear interpolation between points
        t1 = times_sorted[first_crossing_idx - 1]
        t2 = times_sorted[first_crossing_idx]
        f1 = fluxes_sorted[first_crossing_idx - 1]
        f2 = fluxes_sorted[first_crossing_idx]

        # Interpolate time at target flux
        if f1 != f2:
            crossing_time = t1 + (target_flux - f1) * (t2 - t1) / (f2 - f1)
        else:
            crossing_time = t2
    else:
        crossing_time = times_sorted[first_crossing_idx]

    # Time from peak to crossing
    decline_time = crossing_time - peak_time

    return decline_time


def extract_time_to_decline_single(obj_lc: pd.DataFrame) -> Dict[str, float]:
    """
    Extract time-to-decline features for a single object.

    Args:
        obj_lc: Lightcurve DataFrame (must have Filter, Time (MJD), Flux)

    Returns:
        Dictionary of time-to-decline features
    """
    features = {}

    # Decline thresholds (% of peak flux)
    thresholds = [0.8, 0.6, 0.4, 0.2, 0.1]

    # LSST bands
    bands = ['u', 'g', 'r', 'i', 'z', 'y']

    for band in bands:
        band_lc = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_lc) < 3:
            # Not enough observations
            for thresh in thresholds:
                features[f'{band}_decline_to_{int(thresh*100)}pct'] = np.nan
            features[f'{band}_decline_velocity'] = np.nan
            continue

        times = band_lc['Time (MJD)'].values
        fluxes = band_lc['Flux'].values

        # Find peak
        peak_time, peak_flux = find_peak_time_and_flux(times, fluxes)

        if np.isnan(peak_time) or np.isnan(peak_flux):
            for thresh in thresholds:
                features[f'{band}_decline_to_{int(thresh*100)}pct'] = np.nan
            features[f'{band}_decline_velocity'] = np.nan
            continue

        # Compute time to decline for each threshold
        decline_times = []
        for thresh in thresholds:
            decline_time = compute_time_to_decline(times, fluxes, peak_time,
                                                   peak_flux, thresh)
            features[f'{band}_decline_to_{int(thresh*100)}pct'] = decline_time

            if np.isfinite(decline_time):
                decline_times.append(decline_time)

        # Compute decline velocity (average flux drop per day)
        if len(decline_times) >= 2:
            # From 80% to 20% is typical metric
            t_80 = features.get(f'{band}_decline_to_80pct', np.nan)
            t_20 = features.get(f'{band}_decline_to_20pct', np.nan)

            if np.isfinite(t_80) and np.isfinite(t_20) and t_20 > t_80:
                # Velocity: fraction of flux lost per day
                velocity = (0.8 - 0.2) / (t_20 - t_80)
                features[f'{band}_decline_velocity'] = velocity
            else:
                features[f'{band}_decline_velocity'] = np.nan
        else:
            features[f'{band}_decline_velocity'] = np.nan

    return features


def fill_nan_decline() -> Dict[str, float]:
    """Return dictionary with all decline features set to NaN"""
    features = {}
    thresholds = [0.8, 0.6, 0.4, 0.2, 0.1]
    bands = ['u', 'g', 'r', 'i', 'z', 'y']

    for band in bands:
        for thresh in thresholds:
            features[f'{band}_decline_to_{int(thresh*100)}pct'] = np.nan
        features[f'{band}_decline_velocity'] = np.nan

    return features


def extract_time_to_decline(lightcurves: pd.DataFrame,
                            object_ids: List[str]) -> pd.DataFrame:
    """
    Extract time-to-decline features for multiple objects.

    Args:
        lightcurves: DataFrame with lightcurve data (object_id, Time (MJD), Flux, Filter)
        object_ids: List of object IDs to process

    Returns:
        DataFrame with time-to-decline features
    """
    all_features = []

    # Pre-group for efficiency
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    for i, obj_id in enumerate(object_ids):
        if (i + 1) % 500 == 0:
            print(f"    Time-to-decline: {i+1}/{len(object_ids)} objects processed", flush=True)

        obj_lc = grouped.get(obj_id, pd.DataFrame())

        if obj_lc.empty:
            features = fill_nan_decline()
            features['object_id'] = obj_id
            all_features.append(features)
            continue

        features = extract_time_to_decline_single(obj_lc)
        features['object_id'] = obj_id
        all_features.append(features)

    df = pd.DataFrame(all_features)

    # Fill remaining NaNs with median
    for col in df.columns:
        if col != 'object_id':
            median_val = df[col].median()
            if np.isnan(median_val):
                median_val = 0.0
            df[col].fillna(median_val, inplace=True)

    return df


if __name__ == "__main__":
    # Test time-to-decline features
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_all_data

    print("Loading data...")
    data = load_all_data()

    print("\nTesting time-to-decline on first 10 objects...")
    sample_ids = data['train_meta']['object_id'].head(10).tolist()
    decline_features = extract_time_to_decline(data['train_lc'], sample_ids)

    print(f"\nExtracted {len(decline_features.columns)-1} time-to-decline features")
    print("\nFeature columns:")
    print([c for c in decline_features.columns if c != 'object_id'])

    print("\nSample decline times (g-band):")
    g_cols = [c for c in decline_features.columns if c.startswith('g_decline_')]
    print(decline_features[['object_id'] + g_cols].head())

    # Check feature coverage
    print("\nFeature coverage (non-NaN):")
    for col in ['g_decline_to_80pct', 'g_decline_to_20pct', 'g_decline_velocity']:
        if col in decline_features.columns:
            coverage = decline_features[col].notna().sum() / len(decline_features)
            print(f"  {col}: {100*coverage:.1f}%")
