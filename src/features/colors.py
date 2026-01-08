"""
Color feature extraction for MALLORN lightcurve classification.

Color features exploit the key physical difference between transient types:
- TDEs maintain hot blackbody temperatures (blue colors)
- Supernovae cool over time (colors redden)
- AGN have stochastic color variability

Key features:
- Colors at peak (g-r, r-i, etc.)
- Colors at post-peak epochs (+20, +50, +100 days)
- Color evolution slopes
- Color variability (std of colors over time)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import warnings

# LSST bands in wavelength order (blue to red)
LSST_BANDS = ["u", "g", "r", "i", "z", "y"]

# Central wavelengths in nm
BAND_WAVELENGTHS = {
    "u": 367.0, "g": 482.5, "r": 622.2,
    "i": 754.5, "z": 869.1, "y": 971.0
}

# Common color combinations
COLOR_PAIRS = [
    ("g", "r"),  # Most important - sensitive to temperature
    ("r", "i"),  # Red optical
    ("u", "g"),  # Blue/UV - sensitive to hot sources
    ("i", "z"),  # Near-IR
]


def find_peak_time(times: np.ndarray, fluxes: np.ndarray) -> float:
    """Find the time of peak flux."""
    if len(times) == 0:
        return np.nan
    peak_idx = np.argmax(fluxes)
    return times[peak_idx]


def interpolate_flux(times: np.ndarray, fluxes: np.ndarray,
                     target_time: float, max_gap: float = 50.0) -> float:
    """
    Interpolate flux at a target time using linear interpolation.

    Args:
        times: Array of observation times
        fluxes: Array of flux values
        target_time: Time at which to interpolate
        max_gap: Maximum gap (days) for interpolation to be valid

    Returns:
        Interpolated flux or NaN if not possible
    """
    if len(times) < 2:
        return np.nan

    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    fluxes = fluxes[sort_idx]

    # Check if target is within range
    if target_time < times[0] or target_time > times[-1]:
        return np.nan

    # Find bracketing points
    idx = np.searchsorted(times, target_time)
    if idx == 0:
        return fluxes[0]
    if idx == len(times):
        return fluxes[-1]

    t1, t2 = times[idx-1], times[idx]
    f1, f2 = fluxes[idx-1], fluxes[idx]

    # Check gap size
    if t2 - t1 > max_gap:
        return np.nan

    # Linear interpolation
    weight = (target_time - t1) / (t2 - t1)
    return f1 + weight * (f2 - f1)


def compute_color(flux1: float, flux2: float) -> float:
    """
    Compute color as flux ratio (or log ratio ~ magnitude difference).

    For AB magnitudes: color = -2.5 * log10(f1/f2)
    We use log ratio as a proxy.
    """
    if np.isnan(flux1) or np.isnan(flux2):
        return np.nan
    if flux1 <= 0 or flux2 <= 0:
        return np.nan

    # Use log flux ratio (proportional to magnitude difference)
    return -2.5 * np.log10(flux1 / flux2)


def extract_color_features_single(obj_lc: pd.DataFrame) -> Dict[str, float]:
    """
    Extract color features for a single object's lightcurve.

    Args:
        obj_lc: DataFrame with columns [Time (MJD), Flux, Flux_err, Filter]

    Returns:
        Dictionary of color features
    """
    features = {}

    # Organize data by band
    band_data = {}
    for band in LSST_BANDS:
        band_lc = obj_lc[obj_lc['Filter'] == band]
        if len(band_lc) > 0:
            band_data[band] = {
                'times': band_lc['Time (MJD)'].values,
                'fluxes': band_lc['Flux'].values,
                'errors': band_lc['Flux_err'].values
            }

    # Find global peak time (using r-band or brightest band)
    peak_times = {}
    for band in ['r', 'g', 'i']:  # Prefer these bands for peak
        if band in band_data and len(band_data[band]['fluxes']) > 0:
            peak_times[band] = find_peak_time(
                band_data[band]['times'],
                band_data[band]['fluxes']
            )

    # Use r-band peak as reference, fallback to g or i
    if 'r' in peak_times and not np.isnan(peak_times['r']):
        ref_peak_time = peak_times['r']
    elif 'g' in peak_times and not np.isnan(peak_times['g']):
        ref_peak_time = peak_times['g']
    elif 'i' in peak_times and not np.isnan(peak_times['i']):
        ref_peak_time = peak_times['i']
    else:
        ref_peak_time = np.nan

    features['peak_mjd'] = ref_peak_time

    # Epochs relative to peak (in days) - ENHANCED with more time points
    # Based on 2025 TDE paper: post-peak colors are highly predictive
    epochs = {
        'peak': 0,
        'post_10d': 10,      # NEW: Early evolution
        'post_20d': 20,
        'post_30d': 30,      # NEW
        'post_50d': 50,
        'post_75d': 75,      # NEW
        'post_100d': 100,
        'post_150d': 150,    # NEW: Late-time behavior
        'pre_10d': -10,      # NEW
        'pre_20d': -20,
    }

    # Compute colors at each epoch
    for epoch_name, delta_t in epochs.items():
        target_time = ref_peak_time + delta_t if not np.isnan(ref_peak_time) else np.nan

        # Interpolate flux at target time for each band
        epoch_fluxes = {}
        for band in LSST_BANDS:
            if band in band_data:
                flux = interpolate_flux(
                    band_data[band]['times'],
                    band_data[band]['fluxes'],
                    target_time
                )
                epoch_fluxes[band] = flux
            else:
                epoch_fluxes[band] = np.nan

        # Compute colors for each pair
        for band1, band2 in COLOR_PAIRS:
            color_name = f'{band1}_{band2}_{epoch_name}'
            color = compute_color(epoch_fluxes.get(band1, np.nan),
                                  epoch_fluxes.get(band2, np.nan))
            features[color_name] = color

    # Compute color evolution slopes
    for band1, band2 in COLOR_PAIRS:
        color_peak = features.get(f'{band1}_{band2}_peak', np.nan)
        color_post_50d = features.get(f'{band1}_{band2}_post_50d', np.nan)
        color_post_100d = features.get(f'{band1}_{band2}_post_100d', np.nan)

        # Slope from peak to +50 days
        if not np.isnan(color_peak) and not np.isnan(color_post_50d):
            features[f'{band1}_{band2}_slope_50d'] = (color_post_50d - color_peak) / 50.0
        else:
            features[f'{band1}_{band2}_slope_50d'] = np.nan

        # Slope from peak to +100 days
        if not np.isnan(color_peak) and not np.isnan(color_post_100d):
            features[f'{band1}_{band2}_slope_100d'] = (color_post_100d - color_peak) / 100.0
        else:
            features[f'{band1}_{band2}_slope_100d'] = np.nan

    # Color variability (std of instantaneous colors)
    # Compute colors at each observation time where both bands have data
    for band1, band2 in COLOR_PAIRS:
        if band1 in band_data and band2 in band_data:
            colors = []
            times1 = band_data[band1]['times']
            fluxes1 = band_data[band1]['fluxes']

            for t, f1 in zip(times1, fluxes1):
                f2 = interpolate_flux(band_data[band2]['times'],
                                      band_data[band2]['fluxes'], t, max_gap=5.0)
                c = compute_color(f1, f2)
                if not np.isnan(c):
                    colors.append(c)

            if len(colors) >= 3:
                features[f'{band1}_{band2}_std'] = np.std(colors)
                features[f'{band1}_{band2}_range'] = np.max(colors) - np.min(colors)
            else:
                features[f'{band1}_{band2}_std'] = np.nan
                features[f'{band1}_{band2}_range'] = np.nan
        else:
            features[f'{band1}_{band2}_std'] = np.nan
            features[f'{band1}_{band2}_range'] = np.nan

    # Peak fluxes and flux ratios
    for band in LSST_BANDS:
        if band in band_data:
            features[f'{band}_peak_flux'] = np.max(band_data[band]['fluxes'])
        else:
            features[f'{band}_peak_flux'] = np.nan

    # Flux ratios at peak
    for band1, band2 in COLOR_PAIRS:
        f1 = features.get(f'{band1}_peak_flux', np.nan)
        f2 = features.get(f'{band2}_peak_flux', np.nan)
        if not np.isnan(f1) and not np.isnan(f2) and f2 > 0:
            features[f'{band1}_{band2}_peak_flux_ratio'] = f1 / f2
        else:
            features[f'{band1}_{band2}_peak_flux_ratio'] = np.nan

    # Time lag between peaks in different bands
    # (TDEs often peak at similar times across bands, SNe show delays)
    for band1, band2 in [('g', 'r'), ('r', 'i')]:
        if band1 in peak_times and band2 in peak_times:
            lag = peak_times.get(band1, np.nan) - peak_times.get(band2, np.nan)
            features[f'{band1}_{band2}_peak_lag'] = lag
        else:
            features[f'{band1}_{band2}_peak_lag'] = np.nan

    # === ENHANCED FEATURES (Phase A additions) ===

    # Color curvature: is evolution linear or curved?
    # Second derivative of color evolution
    for band1, band2 in [('g', 'r'), ('r', 'i')]:
        c_peak = features.get(f'{band1}_{band2}_peak', np.nan)
        c_30 = features.get(f'{band1}_{band2}_post_30d', np.nan)
        c_75 = features.get(f'{band1}_{band2}_post_75d', np.nan)

        if not any(np.isnan([c_peak, c_30, c_75])):
            # Approximate second derivative
            slope1 = (c_30 - c_peak) / 30.0
            slope2 = (c_75 - c_30) / 45.0
            curvature = (slope2 - slope1) / 37.5  # average time span
            features[f'{band1}_{band2}_curvature'] = curvature
        else:
            features[f'{band1}_{band2}_curvature'] = np.nan

    # Color stability: how constant is the color during late-time?
    # Low variance = stable (TDE-like), high variance = variable (AGN-like)
    for band1, band2 in [('g', 'r'), ('r', 'i')]:
        late_colors = [
            features.get(f'{band1}_{band2}_post_50d', np.nan),
            features.get(f'{band1}_{band2}_post_75d', np.nan),
            features.get(f'{band1}_{band2}_post_100d', np.nan),
            features.get(f'{band1}_{band2}_post_150d', np.nan),
        ]
        valid_colors = [c for c in late_colors if not np.isnan(c)]

        if len(valid_colors) >= 2:
            features[f'{band1}_{band2}_late_stability'] = np.std(valid_colors)
            features[f'{band1}_{band2}_late_mean'] = np.mean(valid_colors)
        else:
            features[f'{band1}_{band2}_late_stability'] = np.nan
            features[f'{band1}_{band2}_late_mean'] = np.nan

    # Temperature estimation at multiple epochs
    # Using Wien's approximation: bluer colors = hotter
    def estimate_temp_from_gr(g_r_color):
        """Estimate blackbody temperature from g-r color."""
        if np.isnan(g_r_color):
            return np.nan
        # Empirical calibration: T ~ 7000K / (g-r + 0.6)
        # Valid range: -0.5 < g-r < 2.0
        if g_r_color < -0.5:
            return 50000  # Very hot
        elif g_r_color > 2.0:
            return 3000   # Cool
        else:
            return 7000 / (g_r_color + 0.6)

    for epoch in ['peak', 'post_30d', 'post_75d', 'post_150d']:
        gr_color = features.get(f'g_r_{epoch}', np.nan)
        temp = estimate_temp_from_gr(gr_color)
        features[f'temp_{epoch}'] = temp

    # Temperature evolution rate (dT/dt)
    t_peak = features.get('temp_peak', np.nan)
    t_30 = features.get('temp_post_30d', np.nan)
    t_75 = features.get('temp_post_75d', np.nan)
    t_150 = features.get('temp_post_150d', np.nan)

    if not np.isnan(t_peak) and not np.isnan(t_30):
        features['temp_slope_early'] = (t_30 - t_peak) / 30.0  # K/day
    else:
        features['temp_slope_early'] = np.nan

    if not np.isnan(t_30) and not np.isnan(t_75):
        features['temp_slope_mid'] = (t_75 - t_30) / 45.0
    else:
        features['temp_slope_mid'] = np.nan

    if not np.isnan(t_75) and not np.isnan(t_150):
        features['temp_slope_late'] = (t_150 - t_75) / 75.0
    else:
        features['temp_slope_late'] = np.nan

    # Temperature stability (TDEs maintain ~constant temperature)
    temps = [t_peak, t_30, t_75, t_150]
    valid_temps = [t for t in temps if not np.isnan(t)]
    if len(valid_temps) >= 2:
        features['temp_stability'] = np.std(valid_temps) / np.mean(valid_temps)  # CV
    else:
        features['temp_stability'] = np.nan

    return features


def extract_color_features(
    lightcurves: pd.DataFrame,
    object_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract color features for multiple objects.

    Args:
        lightcurves: DataFrame with columns [object_id, Time (MJD), Flux, Flux_err, Filter]
        object_ids: Optional list of object IDs to process

    Returns:
        DataFrame with one row per object and color feature columns
    """
    if object_ids is None:
        object_ids = lightcurves['object_id'].unique()

    # Pre-group for O(1) lookup per object
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if (i + 1) % 500 == 0:
            print(f"    Colors: {i+1}/{len(object_ids)} objects processed")

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue
        features = extract_color_features_single(obj_lc)
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

    print("\nExtracting color features for first 20 objects...")
    sample_ids = data['train_meta']['object_id'].head(20).tolist()
    color_features = extract_color_features(data['train_lc'], sample_ids)

    print(f"\nExtracted {len(color_features.columns)-1} color features")
    print("\nFeature columns:")
    print([c for c in color_features.columns if c != 'object_id'])
    print("\nSample values (g-r colors):")
    gr_cols = [c for c in color_features.columns if c.startswith('g_r')]
    print(color_features[['object_id'] + gr_cols].head())
