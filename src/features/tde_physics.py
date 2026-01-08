"""
TDE-specific physics features based on literature review.

Key insights from papers:
1. Gezari 2021 (arxiv:2104.14580): TDE light curves show power-law decline scaling with BH mass
2. van Velzen 2020 (arxiv:2008.05461): Optical TDEs have shared photometric properties
3. ALeRCE classifier (arxiv:2503.19698): Color variance is highly discriminative

Key physics:
- TDEs decay as t^-5/3 (bolometric) or t^-5/12 (monochromatic optical)
- TDEs maintain constant temperature ~12,000-13,000K
- TDEs have nearly constant g-r color (blue continuum)
- SNe cool and redden over time
- AGN have stochastic color variations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings

LSST_BANDS = ["u", "g", "r", "i", "z", "y"]


def compute_color_variance(obj_lc: pd.DataFrame, color_pairs: List[tuple] = [('g', 'r'), ('r', 'i')]) -> Dict[str, float]:
    """
    Compute color variance over time.

    TDEs have nearly CONSTANT colors (key discriminator!)
    SNe redden over time (increasing g-r)
    AGN have stochastic color changes

    Returns variance and range of colors.
    """
    features = {}

    for b1, b2 in color_pairs:
        # Get band data
        band1 = obj_lc[obj_lc['Filter'] == b1].sort_values('Time (MJD)')
        band2 = obj_lc[obj_lc['Filter'] == b2].sort_values('Time (MJD)')

        if len(band1) < 3 or len(band2) < 3:
            features[f'{b1}_{b2}_color_var'] = np.nan
            features[f'{b1}_{b2}_color_range'] = np.nan
            features[f'{b1}_{b2}_color_trend'] = np.nan
            continue

        # Interpolate to common time grid
        colors = []
        times = []

        for _, row1 in band1.iterrows():
            t1 = row1['Time (MJD)']
            f1 = row1['Flux']

            # Find closest band2 observation
            dt = np.abs(band2['Time (MJD)'].values - t1)
            min_idx = np.argmin(dt)

            if dt[min_idx] < 5:  # Within 5 days
                f2 = band2.iloc[min_idx]['Flux']
                if f1 > 0 and f2 > 0:
                    # Color in magnitude
                    color = -2.5 * np.log10(f1 / f2)
                    colors.append(color)
                    times.append(t1)

        if len(colors) >= 3:
            colors = np.array(colors)
            times = np.array(times)

            # Variance (low for TDEs!)
            features[f'{b1}_{b2}_color_var'] = np.var(colors)

            # Range
            features[f'{b1}_{b2}_color_range'] = np.max(colors) - np.min(colors)

            # Trend (slope of color vs time)
            # Positive = reddening (SNe), ~0 = TDE, variable = AGN
            if len(times) >= 3:
                coeffs = np.polyfit(times - times[0], colors, 1)
                features[f'{b1}_{b2}_color_trend'] = coeffs[0] * 100  # per 100 days
            else:
                features[f'{b1}_{b2}_color_trend'] = np.nan
        else:
            features[f'{b1}_{b2}_color_var'] = np.nan
            features[f'{b1}_{b2}_color_range'] = np.nan
            features[f'{b1}_{b2}_color_trend'] = np.nan

    return features


def compute_late_time_behavior(obj_lc: pd.DataFrame, bands: List[str] = ['g', 'r', 'i']) -> Dict[str, float]:
    """
    Compute late-time lightcurve behavior.

    TDEs: slow power-law decline (t^-5/12 monochromatic)
    SNe: faster exponential decline, often plateau then cliff
    AGN: irregular, may brighten again
    """
    features = {}

    for band in bands:
        band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_data) < 5:
            features[f'{band}_late_slope'] = np.nan
            features[f'{band}_late_flux_ratio'] = np.nan
            features[f'{band}_rebrightening'] = np.nan
            continue

        times = band_data['Time (MJD)'].values
        fluxes = band_data['Flux'].values

        # Find peak
        peak_idx = np.argmax(fluxes)
        peak_time = times[peak_idx]
        peak_flux = fluxes[peak_idx]

        # Late time = >50 days after peak
        late_mask = times > (peak_time + 50)
        late_times = times[late_mask]
        late_fluxes = fluxes[late_mask]

        if len(late_times) >= 3 and peak_flux > 0:
            # Late-time decay slope (in log space)
            # TDEs: shallow slope (~-5/12 = -0.42)
            # SNe: steeper slope
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                log_t = np.log10(late_times - peak_time + 1)
                log_f = np.log10(np.maximum(late_fluxes, 1e-10))

                if np.std(log_t) > 0:
                    coeffs = np.polyfit(log_t, log_f, 1)
                    features[f'{band}_late_slope'] = coeffs[0]
                else:
                    features[f'{band}_late_slope'] = np.nan

            # Flux ratio: late / peak
            features[f'{band}_late_flux_ratio'] = np.mean(late_fluxes) / peak_flux

            # Check for rebrightening (AGN signature)
            if len(late_fluxes) >= 3:
                late_max = np.max(late_fluxes)
                late_mean = np.mean(late_fluxes)
                features[f'{band}_rebrightening'] = late_max / late_mean if late_mean > 0 else 1.0
            else:
                features[f'{band}_rebrightening'] = np.nan
        else:
            features[f'{band}_late_slope'] = np.nan
            features[f'{band}_late_flux_ratio'] = np.nan
            features[f'{band}_rebrightening'] = np.nan

    return features


def compute_rise_characteristics(obj_lc: pd.DataFrame, bands: List[str] = ['g', 'r']) -> Dict[str, float]:
    """
    Compute rise phase characteristics.

    TDEs: Rise time 29-60 days, can be linear/quadratic/Gaussian
    SNe: Generally faster rise (days to ~2 weeks)
    """
    features = {}

    for band in bands:
        band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_data) < 5:
            features[f'{band}_rise_shape'] = np.nan
            features[f'{band}_rise_rate'] = np.nan
            continue

        times = band_data['Time (MJD)'].values
        fluxes = band_data['Flux'].values

        peak_idx = np.argmax(fluxes)
        peak_flux = fluxes[peak_idx]

        # Rise phase data
        rise_times = times[:peak_idx + 1]
        rise_fluxes = fluxes[:peak_idx + 1]

        if len(rise_times) >= 3 and peak_flux > 0:
            # Normalize flux to 0-1
            norm_flux = rise_fluxes / peak_flux
            norm_time = (rise_times - rise_times[0]) / (rise_times[-1] - rise_times[0] + 1e-6)

            # Rise shape: compare to linear
            # < 1: concave (accelerating) - more TDE-like
            # > 1: convex (decelerating)
            linear_flux = norm_time
            shape_ratio = np.mean(norm_flux) / np.mean(linear_flux) if np.mean(linear_flux) > 0 else 1.0
            features[f'{band}_rise_shape'] = shape_ratio

            # Rise rate: flux increase per day
            if rise_times[-1] > rise_times[0]:
                features[f'{band}_rise_rate'] = peak_flux / (rise_times[-1] - rise_times[0])
            else:
                features[f'{band}_rise_rate'] = np.nan
        else:
            features[f'{band}_rise_shape'] = np.nan
            features[f'{band}_rise_rate'] = np.nan

    return features


def compute_temperature_stability(obj_lc: pd.DataFrame) -> Dict[str, float]:
    """
    Estimate temperature stability over time.

    TDEs: Constant ~12,000-13,000K
    SNe: Start hot, cool over time
    AGN: Variable

    We use g-r color as temperature proxy.
    """
    features = {}

    g_data = obj_lc[obj_lc['Filter'] == 'g'].sort_values('Time (MJD)')
    r_data = obj_lc[obj_lc['Filter'] == 'r'].sort_values('Time (MJD)')

    if len(g_data) < 3 or len(r_data) < 3:
        features['temp_stability'] = np.nan
        features['temp_trend'] = np.nan
        features['temp_late_vs_peak'] = np.nan
        return features

    # Compute temperature proxy at multiple epochs
    temps = []
    times = []

    for _, g_row in g_data.iterrows():
        t = g_row['Time (MJD)']
        g_flux = g_row['Flux']

        # Find closest r observation
        dt = np.abs(r_data['Time (MJD)'].values - t)
        min_idx = np.argmin(dt)

        if dt[min_idx] < 3 and g_flux > 0 and r_data.iloc[min_idx]['Flux'] > 0:
            r_flux = r_data.iloc[min_idx]['Flux']

            # Simple temperature proxy from g-r color
            # Bluer (higher g/r) = hotter
            g_r = -2.5 * np.log10(g_flux / r_flux)

            # Convert to rough temperature (calibrated for TDE range)
            if g_r < -0.5:
                temp = 40000
            elif g_r > 1.5:
                temp = 5000
            else:
                temp = 7000 / (g_r + 0.5)

            temps.append(temp)
            times.append(t)

    if len(temps) >= 3:
        temps = np.array(temps)
        times = np.array(times)

        # Temperature stability (low std = TDE-like)
        features['temp_stability'] = np.std(temps) / np.mean(temps)  # Coefficient of variation

        # Temperature trend (negative = cooling like SN)
        coeffs = np.polyfit(times - times[0], temps, 1)
        features['temp_trend'] = coeffs[0] * 100  # Change per 100 days

        # Late vs peak temperature
        peak_idx = len(temps) // 4  # Approximate peak
        if len(temps) > 4:
            peak_temp = np.mean(temps[:max(2, peak_idx)])
            late_temp = np.mean(temps[-3:])
            features['temp_late_vs_peak'] = late_temp / peak_temp
        else:
            features['temp_late_vs_peak'] = np.nan
    else:
        features['temp_stability'] = np.nan
        features['temp_trend'] = np.nan
        features['temp_late_vs_peak'] = np.nan

    return features


def compute_decay_power_law(obj_lc: pd.DataFrame, bands: List[str] = ['r']) -> Dict[str, float]:
    """
    Fit power-law decay to post-peak lightcurve.

    TDE expected: t^-5/3 ≈ -1.67 (bolometric) or t^-5/12 ≈ -0.42 (monochromatic)
    SN: Often steeper exponential decay
    """
    features = {}

    for band in bands:
        band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_data) < 5:
            features[f'{band}_decay_alpha'] = np.nan
            features[f'{band}_decay_alpha_late'] = np.nan
            features[f'{band}_decay_residual'] = np.nan
            continue

        times = band_data['Time (MJD)'].values
        fluxes = band_data['Flux'].values

        peak_idx = np.argmax(fluxes)
        peak_time = times[peak_idx]
        peak_flux = fluxes[peak_idx]

        # Post-peak data
        post_mask = times > peak_time
        post_times = times[post_mask]
        post_fluxes = fluxes[post_mask]

        if len(post_times) >= 4 and peak_flux > 0:
            # Fit: log(flux) = alpha * log(t - t_peak) + const
            dt = post_times - peak_time
            dt = np.maximum(dt, 1)  # Avoid log(0)

            valid = post_fluxes > 0
            if np.sum(valid) >= 3:
                log_t = np.log10(dt[valid])
                log_f = np.log10(post_fluxes[valid])

                coeffs = np.polyfit(log_t, log_f, 1)
                features[f'{band}_decay_alpha'] = coeffs[0]

                # Residuals from power-law fit
                predicted = coeffs[0] * log_t + coeffs[1]
                features[f'{band}_decay_residual'] = np.std(log_f - predicted)

                # Late-time only (>50 days)
                late_valid = (dt > 50) & valid
                if np.sum(late_valid) >= 3:
                    log_t_late = np.log10(dt[late_valid])
                    log_f_late = np.log10(post_fluxes[late_valid])
                    coeffs_late = np.polyfit(log_t_late, log_f_late, 1)
                    features[f'{band}_decay_alpha_late'] = coeffs_late[0]
                else:
                    features[f'{band}_decay_alpha_late'] = np.nan
            else:
                features[f'{band}_decay_alpha'] = np.nan
                features[f'{band}_decay_residual'] = np.nan
                features[f'{band}_decay_alpha_late'] = np.nan
        else:
            features[f'{band}_decay_alpha'] = np.nan
            features[f'{band}_decay_residual'] = np.nan
            features[f'{band}_decay_alpha_late'] = np.nan

    return features


def extract_tde_physics_features_single(obj_lc: pd.DataFrame) -> Dict[str, float]:
    """Extract all TDE-specific physics features for a single object."""
    features = {}

    # Color variance (KEY discriminator!)
    features.update(compute_color_variance(obj_lc))

    # Late-time behavior
    features.update(compute_late_time_behavior(obj_lc))

    # Rise characteristics
    features.update(compute_rise_characteristics(obj_lc))

    # Temperature stability
    features.update(compute_temperature_stability(obj_lc))

    # Power-law decay
    features.update(compute_decay_power_law(obj_lc))

    return features


def extract_tde_physics_features(
    lightcurves: pd.DataFrame,
    object_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract TDE-specific physics features for multiple objects.

    Args:
        lightcurves: DataFrame with lightcurve data
        object_ids: Optional list of object IDs

    Returns:
        DataFrame with TDE physics features
    """
    if object_ids is None:
        object_ids = lightcurves['object_id'].unique()

    # Pre-group for efficiency
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if (i + 1) % 500 == 0:
            print(f"    TDE Physics: {i+1}/{len(object_ids)} objects processed")

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue

        features = extract_tde_physics_features_single(obj_lc)
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

    print("\nExtracting TDE physics features for first 20 objects...")
    sample_ids = data['train_meta']['object_id'].head(20).tolist()
    tde_features = extract_tde_physics_features(data['train_lc'], sample_ids)

    print(f"\nExtracted {len(tde_features.columns)-1} TDE physics features")
    print("\nFeature columns:")
    print([c for c in tde_features.columns if c != 'object_id'])
    print("\nSample values:")
    print(tde_features[['object_id', 'g_r_color_var', 'temp_stability', 'r_decay_alpha']].head())
