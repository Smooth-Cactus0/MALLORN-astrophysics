"""
Enhanced Post-Peak Color Features for TDE Classification

Based on 2025 TDE research (arxiv.org/abs/2509.25902):
- Post-peak colors are THE #1 discriminator for TDEs
- TDEs stay blue (hot accretion disk), SNe cool rapidly
- Color at peak + evolution over 0-150 days critical

Current v34a: 4 color features (g-r, r-i at 20d, 50d)
v47 adds: 40+ color features at 8 time points, 4 color pairs

Expected gain: +2-3% based on research
Target: Push toward LB 1.0
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.interpolate import interp1d


def get_flux_at_time(times: np.ndarray, fluxes: np.ndarray, target_time: float,
                     window: float = 5.0) -> float:
    """
    Get interpolated flux at target time using nearby observations.

    Args:
        times: Observation times (MJD)
        fluxes: Flux measurements
        target_time: Target time to extract flux
        window: Time window (±days) to include observations

    Returns:
        Interpolated flux at target_time, or np.nan if insufficient data
    """
    # Find observations within window
    mask = (times >= target_time - window) & (times <= target_time + window)

    if np.sum(mask) < 2:
        return np.nan

    t_window = times[mask]
    f_window = fluxes[mask]

    # Sort by time
    sort_idx = np.argsort(t_window)
    t_sorted = t_window[sort_idx]
    f_sorted = f_window[sort_idx]

    # Linear interpolation
    try:
        interp_func = interp1d(t_sorted, f_sorted, kind='linear',
                              bounds_error=False, fill_value=np.nan)
        return float(interp_func(target_time))
    except:
        return np.nan


def compute_color(band1_flux: float, band2_flux: float) -> float:
    """
    Compute color (magnitude difference) from flux ratio.

    Color = -2.5 * log10(flux1 / flux2)

    Args:
        band1_flux: Flux in band 1
        band2_flux: Flux in band 2

    Returns:
        Color (mag1 - mag2), or np.nan if invalid
    """
    if band1_flux <= 0 or band2_flux <= 0:
        return np.nan

    if not np.isfinite(band1_flux) or not np.isfinite(band2_flux):
        return np.nan

    return -2.5 * np.log10(band1_flux / band2_flux)


def extract_enhanced_colors_single(obj_lc: pd.DataFrame,
                                   peak_time: float = None) -> Dict[str, float]:
    """
    Extract comprehensive color features for single object.

    Args:
        obj_lc: Lightcurve DataFrame for one object (must have Filter, Time (MJD), Flux)
        peak_time: Peak time in MJD. If None, computed from g-band

    Returns:
        Dictionary of color features
    """
    features = {}

    # Determine peak time if not provided
    if peak_time is None:
        g_lc = obj_lc[obj_lc['Filter'] == 'g'].sort_values('Time (MJD)')
        if len(g_lc) > 0:
            peak_time = g_lc.loc[g_lc['Flux'].idxmax(), 'Time (MJD)']
        else:
            # Fallback to r-band
            r_lc = obj_lc[obj_lc['Filter'] == 'r'].sort_values('Time (MJD)')
            if len(r_lc) > 0:
                peak_time = r_lc.loc[r_lc['Flux'].idxmax(), 'Time (MJD)']
            else:
                # Can't determine peak - return NaNs
                return fill_nan_colors()

    # Define time points (days after peak)
    time_offsets = [0, 10, 20, 30, 50, 75, 100, 150]

    # Define color pairs (bluer - redder)
    color_pairs = [
        ('u', 'g'),  # UV - blue
        ('g', 'r'),  # Blue - red (most discriminative)
        ('r', 'i'),  # Red - near-IR
        ('i', 'z')   # Near-IR - IR
    ]

    # Extract flux data per band
    band_data = {}
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        band_lc = obj_lc[obj_lc['Filter'] == band]
        if len(band_lc) > 0:
            band_data[band] = (band_lc['Time (MJD)'].values, band_lc['Flux'].values)
        else:
            band_data[band] = (np.array([]), np.array([]))

    # Extract colors at each time point
    all_colors = {pair: [] for pair in color_pairs}

    for offset in time_offsets:
        target_time = peak_time + offset

        for (band1, band2) in color_pairs:
            # Get fluxes at target time
            if len(band_data[band1][0]) > 0:
                flux1 = get_flux_at_time(band_data[band1][0], band_data[band1][1], target_time)
            else:
                flux1 = np.nan

            if len(band_data[band2][0]) > 0:
                flux2 = get_flux_at_time(band_data[band2][0], band_data[band2][1], target_time)
            else:
                flux2 = np.nan

            # Compute color
            color = compute_color(flux1, flux2)

            # Store feature
            feature_name = f'{band1}{band2}_color_{offset}d'
            features[feature_name] = color

            # Track for dispersion calculation
            if np.isfinite(color):
                all_colors[(band1, band2)].append(color)

    # Add color dispersion features (consistency over time)
    for (band1, band2) in color_pairs:
        colors = all_colors[(band1, band2)]
        if len(colors) >= 3:
            features[f'{band1}{band2}_color_dispersion'] = np.std(colors)
            features[f'{band1}{band2}_color_range'] = np.max(colors) - np.min(colors)
            features[f'{band1}{band2}_color_mean'] = np.mean(colors)
        else:
            features[f'{band1}{band2}_color_dispersion'] = np.nan
            features[f'{band1}{band2}_color_range'] = np.nan
            features[f'{band1}{band2}_color_mean'] = np.nan

    # Add cross-color features
    # TDEs: g-r stays blue AND r-i stays blue (hot throughout)
    # SNe: Both colors redden over time
    gr_colors = all_colors[('g', 'r')]
    ri_colors = all_colors[('r', 'i')]

    if len(gr_colors) >= 2 and len(ri_colors) >= 2:
        # Color correlation (TDEs: both stay blue → high correlation)
        valid_pairs = [(gr, ri) for gr, ri in zip(gr_colors, ri_colors)
                      if np.isfinite(gr) and np.isfinite(ri)]
        if len(valid_pairs) >= 3:
            gr_vals = np.array([p[0] for p in valid_pairs])
            ri_vals = np.array([p[1] for p in valid_pairs])
            features['gr_ri_color_correlation'] = np.corrcoef(gr_vals, ri_vals)[0, 1]
        else:
            features['gr_ri_color_correlation'] = np.nan
    else:
        features['gr_ri_color_correlation'] = np.nan

    return features


def fill_nan_colors() -> Dict[str, float]:
    """Return dictionary with all color features set to NaN"""
    features = {}

    time_offsets = [0, 10, 20, 30, 50, 75, 100, 150]
    color_pairs = [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z')]

    for offset in time_offsets:
        for (band1, band2) in color_pairs:
            features[f'{band1}{band2}_color_{offset}d'] = np.nan

    for (band1, band2) in color_pairs:
        features[f'{band1}{band2}_color_dispersion'] = np.nan
        features[f'{band1}{band2}_color_range'] = np.nan
        features[f'{band1}{band2}_color_mean'] = np.nan

    features['gr_ri_color_correlation'] = np.nan

    return features


def extract_enhanced_colors(lightcurves: pd.DataFrame,
                            object_ids: List[str],
                            peak_times: Dict[str, float] = None) -> pd.DataFrame:
    """
    Extract enhanced color features for multiple objects.

    Args:
        lightcurves: DataFrame with lightcurve data (object_id, Time (MJD), Flux, Filter)
        object_ids: List of object IDs to process
        peak_times: Optional dict mapping object_id -> peak_time (MJD)

    Returns:
        DataFrame with enhanced color features
    """
    all_features = []

    # Pre-group for efficiency
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    for i, obj_id in enumerate(object_ids):
        if (i + 1) % 500 == 0:
            print(f"    Enhanced colors: {i+1}/{len(object_ids)} objects processed", flush=True)

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            features = fill_nan_colors()
            features['object_id'] = obj_id
            all_features.append(features)
            continue

        # Get peak time if provided
        peak_time = None
        if peak_times is not None and obj_id in peak_times:
            peak_time = peak_times[obj_id]

        features = extract_enhanced_colors_single(obj_lc, peak_time)
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
    # Test enhanced color features
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_all_data

    print("Loading data...")
    data = load_all_data()

    print("\nTesting enhanced colors on first 10 objects...")
    sample_ids = data['train_meta']['object_id'].head(10).tolist()
    color_features = extract_enhanced_colors(data['train_lc'], sample_ids)

    print(f"\nExtracted {len(color_features.columns)-1} enhanced color features")
    print("\nFeature columns (first 20):")
    print([c for c in color_features.columns if c != 'object_id'][:20])

    print("\nSample g-r colors at different time points:")
    gr_cols = [c for c in color_features.columns if c.startswith('gr_color_')]
    print(color_features[['object_id'] + gr_cols].head())

    # Check feature coverage
    print("\nFeature coverage (non-NaN):")
    for col in gr_cols[:5]:
        coverage = color_features[col].notna().sum() / len(color_features)
        print(f"  {col}: {100*coverage:.1f}%")
