"""
Advanced physics features for MALLORN v30.

Building on existing physics features, this module adds:
1. Multi-epoch blackbody cooling curves (6-8 temperature measurements)
2. Late-time color evolution (100-200 days post-peak)
3. SED fitting quality metrics (chi-squared, residuals)
4. Cross-band temporal asymmetry comparisons

Key physics insights:
- TDEs: Slow cooling (dT/dt ~ -100 K/day), stay hot for months
- SNe: Fast adiabatic cooling (dT/dt ~ -500 K/day)
- AGN: No systematic cooling, stochastic temperature variations

References:
- 2025 TDE Paper (arxiv:2509.25902): Gaussian Process features, post-peak colors
- van Velzen+ 2020: TDE blackbody evolution
- Gezari 2021: TDE cooling timescales
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.optimize import curve_fit
import warnings

LSST_BANDS = ["u", "g", "r", "i", "z", "y"]

# Effective wavelengths in Angstroms
BAND_WAVELENGTHS = {
    "u": 3670, "g": 4825, "r": 6222,
    "i": 7545, "z": 8691, "y": 9710
}

# Planck's constant, speed of light, Boltzmann constant (CGS)
H_PLANCK = 6.626e-27  # erg·s
C_LIGHT = 2.998e10    # cm/s
K_BOLTZ = 1.381e-16   # erg/K


def planck_function(wavelength_angstrom: float, temperature: float) -> float:
    """
    Planck blackbody function B_λ(T).

    Args:
        wavelength_angstrom: Wavelength in Angstroms
        temperature: Temperature in Kelvin

    Returns:
        Specific intensity (arbitrary units for fitting)
    """
    if temperature <= 0 or wavelength_angstrom <= 0:
        return np.nan

    lam_cm = wavelength_angstrom * 1e-8  # Convert to cm

    try:
        exponent = (H_PLANCK * C_LIGHT) / (lam_cm * K_BOLTZ * temperature)
        if exponent > 700:  # Avoid overflow
            return 0.0

        intensity = (2 * H_PLANCK * C_LIGHT**2 / lam_cm**5) / (np.exp(exponent) - 1)
        return intensity
    except:
        return np.nan


def estimate_temperature_sed(fluxes: Dict[str, float], bands: List[str]) -> Tuple[float, float]:
    """
    Estimate blackbody temperature by fitting Planck function to multi-band fluxes.

    Args:
        fluxes: Dictionary mapping band to flux
        bands: List of bands to use in fit

    Returns:
        (temperature in K, reduced chi-squared of fit)
    """
    # Filter valid fluxes
    valid_bands = [b for b in bands if b in fluxes and fluxes[b] > 0 and not np.isnan(fluxes[b])]

    if len(valid_bands) < 3:
        return np.nan, np.nan

    wavelengths = np.array([BAND_WAVELENGTHS[b] for b in valid_bands])
    observed_fluxes = np.array([fluxes[b] for b in valid_bands])

    # Normalize to avoid numerical issues
    flux_norm = np.median(observed_fluxes)
    if flux_norm <= 0:
        return np.nan, np.nan
    observed_fluxes_norm = observed_fluxes / flux_norm

    # Fit: flux = A * B_λ(T)
    def model(lam, temp, amplitude):
        return amplitude * np.array([planck_function(l, temp) for l in lam])

    try:
        # Initial guess: T ~ 15000 K (typical for optical transients)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                model,
                wavelengths,
                observed_fluxes_norm,
                p0=[15000, 1.0],
                bounds=([3000, 0], [100000, 1e10]),
                maxfev=500
            )

        temp_fit = popt[0]

        # Compute reduced chi-squared
        predicted = model(wavelengths, *popt)
        residuals = observed_fluxes_norm - predicted
        chi2 = np.sum(residuals**2)
        dof = len(valid_bands) - 2  # 2 parameters
        reduced_chi2 = chi2 / dof if dof > 0 else np.nan

        return temp_fit, reduced_chi2

    except:
        return np.nan, np.nan


def compute_multi_epoch_temperatures(obj_lc: pd.DataFrame,
                                    epochs: List[int] = [0, 20, 50, 75, 100, 150, 200]) -> Dict[str, float]:
    """
    Compute blackbody temperature at multiple epochs relative to peak.

    Args:
        obj_lc: Lightcurve DataFrame
        epochs: Days relative to peak (0 = peak, positive = post-peak)

    Returns:
        Features including temperatures, cooling rates, and fit quality
    """
    features = {}

    # Find peak time in r-band
    r_data = obj_lc[obj_lc['Filter'] == 'r'].sort_values('Time (MJD)')
    if len(r_data) < 3:
        # Return NaN for all features
        for epoch in epochs:
            features[f'temp_epoch_{epoch}d'] = np.nan
            features[f'temp_chi2_epoch_{epoch}d'] = np.nan
        features['cooling_rate_early'] = np.nan
        features['cooling_rate_late'] = np.nan
        features['cooling_rate_overall'] = np.nan
        features['temp_dispersion_early'] = np.nan
        features['temp_dispersion_late'] = np.nan
        features['sed_quality_mean'] = np.nan
        features['sed_quality_trend'] = np.nan
        return features

    peak_time = r_data.iloc[r_data['Flux'].argmax()]['Time (MJD)']

    # Extract temperatures at each epoch
    temps = []
    chi2s = []
    valid_epochs = []

    for epoch in epochs:
        target_time = peak_time + epoch

        # Get fluxes near this epoch (within 10 days)
        epoch_lc = obj_lc[np.abs(obj_lc['Time (MJD)'] - target_time) < 10]

        # Average fluxes per band
        band_fluxes = {}
        for band in ['g', 'r', 'i', 'z']:
            band_data = epoch_lc[epoch_lc['Filter'] == band]
            if len(band_data) > 0:
                band_fluxes[band] = band_data['Flux'].median()

        # Fit temperature
        temp, chi2 = estimate_temperature_sed(band_fluxes, ['g', 'r', 'i', 'z'])

        features[f'temp_epoch_{epoch}d'] = temp
        features[f'temp_chi2_epoch_{epoch}d'] = chi2

        if not np.isnan(temp):
            temps.append(temp)
            chi2s.append(chi2 if not np.isnan(chi2) else 0)
            valid_epochs.append(epoch)

    # Cooling rate features
    if len(temps) >= 3:
        temps_arr = np.array(temps)
        epochs_arr = np.array(valid_epochs)

        # Overall cooling rate (linear fit)
        coeffs = np.polyfit(epochs_arr, temps_arr, 1)
        features['cooling_rate_overall'] = coeffs[0]  # K/day

        # Early cooling (first half of epochs)
        mid_idx = len(temps) // 2
        if mid_idx >= 2:
            early_epochs = epochs_arr[:mid_idx]
            early_temps = temps_arr[:mid_idx]
            coeffs_early = np.polyfit(early_epochs, early_temps, 1)
            features['cooling_rate_early'] = coeffs_early[0]

            # Temperature dispersion in early phase
            features['temp_dispersion_early'] = np.std(early_temps)
        else:
            features['cooling_rate_early'] = np.nan
            features['temp_dispersion_early'] = np.nan

        # Late cooling (second half)
        if len(temps) - mid_idx >= 2:
            late_epochs = epochs_arr[mid_idx:]
            late_temps = temps_arr[mid_idx:]
            coeffs_late = np.polyfit(late_epochs - late_epochs[0], late_temps, 1)
            features['cooling_rate_late'] = coeffs_late[0]

            # Temperature dispersion in late phase
            features['temp_dispersion_late'] = np.std(late_temps)
        else:
            features['cooling_rate_late'] = np.nan
            features['temp_dispersion_late'] = np.nan

        # SED fit quality metrics
        if len(chi2s) >= 3:
            features['sed_quality_mean'] = np.mean(chi2s)

            # Does fit quality degrade over time? (positive = degrading)
            coeffs_chi2 = np.polyfit(epochs_arr, chi2s, 1)
            features['sed_quality_trend'] = coeffs_chi2[0]
        else:
            features['sed_quality_mean'] = np.nan
            features['sed_quality_trend'] = np.nan
    else:
        features['cooling_rate_early'] = np.nan
        features['cooling_rate_late'] = np.nan
        features['cooling_rate_overall'] = np.nan
        features['temp_dispersion_early'] = np.nan
        features['temp_dispersion_late'] = np.nan
        features['sed_quality_mean'] = np.nan
        features['sed_quality_trend'] = np.nan

    return features


def compute_late_time_colors(obj_lc: pd.DataFrame,
                             late_epochs: List[int] = [100, 150, 200]) -> Dict[str, float]:
    """
    Compute color evolution in the late-time phase (>100 days post-peak).

    TDEs should maintain blue colors even at late times.
    SNe redden significantly by 100+ days.

    Args:
        obj_lc: Lightcurve DataFrame
        late_epochs: Epochs to measure colors (days post-peak)

    Returns:
        Late-time color features
    """
    features = {}

    # Find peak time
    r_data = obj_lc[obj_lc['Filter'] == 'r'].sort_values('Time (MJD)')
    if len(r_data) < 3:
        for epoch in late_epochs:
            features[f'g_r_late_{epoch}d'] = np.nan
            features[f'r_i_late_{epoch}d'] = np.nan
        features['g_r_late_slope'] = np.nan
        features['r_i_late_slope'] = np.nan
        features['g_r_late_dispersion'] = np.nan
        features['color_accel_g_r'] = np.nan
        return features

    peak_time = r_data.iloc[r_data['Flux'].argmax()]['Time (MJD)']

    # Extract colors at late epochs
    g_r_colors = []
    r_i_colors = []
    g_r_epochs = []
    r_i_epochs = []

    for epoch in late_epochs:
        target_time = peak_time + epoch

        # Get fluxes near this epoch (within 15 days for late times)
        epoch_lc = obj_lc[np.abs(obj_lc['Time (MJD)'] - target_time) < 15]

        g_flux = epoch_lc[epoch_lc['Filter'] == 'g']['Flux'].median() if len(epoch_lc[epoch_lc['Filter'] == 'g']) > 0 else np.nan
        r_flux = epoch_lc[epoch_lc['Filter'] == 'r']['Flux'].median() if len(epoch_lc[epoch_lc['Filter'] == 'r']) > 0 else np.nan
        i_flux = epoch_lc[epoch_lc['Filter'] == 'i']['Flux'].median() if len(epoch_lc[epoch_lc['Filter'] == 'i']) > 0 else np.nan

        # Compute colors
        if g_flux > 0 and r_flux > 0:
            g_r = -2.5 * np.log10(g_flux / r_flux)
            features[f'g_r_late_{epoch}d'] = g_r
            g_r_colors.append(g_r)
            g_r_epochs.append(epoch)
        else:
            features[f'g_r_late_{epoch}d'] = np.nan

        if r_flux > 0 and i_flux > 0:
            r_i = -2.5 * np.log10(r_flux / i_flux)
            features[f'r_i_late_{epoch}d'] = r_i
            r_i_colors.append(r_i)
            r_i_epochs.append(epoch)
        else:
            features[f'r_i_late_{epoch}d'] = np.nan

    # Late-time color evolution
    if len(g_r_colors) >= 2:
        # Slope (reddening rate)
        coeffs_gr = np.polyfit(g_r_epochs, g_r_colors, 1)
        features['g_r_late_slope'] = coeffs_gr[0] * 100  # per 100 days

        # Dispersion
        features['g_r_late_dispersion'] = np.std(g_r_colors)

        # Color acceleration (curvature in color evolution)
        if len(g_r_colors) >= 3:
            coeffs_quad = np.polyfit(g_r_epochs, g_r_colors, 2)
            features['color_accel_g_r'] = coeffs_quad[0] * 10000  # per 100 days²
        else:
            features['color_accel_g_r'] = np.nan
    else:
        features['g_r_late_slope'] = np.nan
        features['g_r_late_dispersion'] = np.nan
        features['color_accel_g_r'] = np.nan

    if len(r_i_colors) >= 2:
        coeffs_ri = np.polyfit(r_i_epochs, r_i_colors, 1)
        features['r_i_late_slope'] = coeffs_ri[0] * 100  # per 100 days
    else:
        features['r_i_late_slope'] = np.nan

    return features


def compute_cross_band_asymmetry(obj_lc: pd.DataFrame,
                                 bands: List[str] = ['g', 'r', 'i']) -> Dict[str, float]:
    """
    Compute temporal asymmetry differences across bands.

    TDEs should show consistent rise/fade asymmetry across all bands (achromatic evolution).
    SNe may show band-dependent asymmetry due to changing opacity.

    Args:
        obj_lc: Lightcurve DataFrame
        bands: Bands to compare

    Returns:
        Cross-band asymmetry features
    """
    features = {}

    # Compute asymmetry per band
    band_asymmetries = {}
    band_rise_times = {}
    band_fade_times = {}
    band_peak_times = {}

    for band in bands:
        band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_data) < 5:
            band_asymmetries[band] = np.nan
            continue

        times = band_data['Time (MJD)'].values
        fluxes = band_data['Flux'].values

        # Find peak
        peak_idx = np.argmax(fluxes)
        peak_time = times[peak_idx]
        band_peak_times[band] = peak_time

        # Rise and fade times
        rise_time = peak_time - times[0] if peak_idx > 0 else np.nan
        fade_time = times[-1] - peak_time if peak_idx < len(times) - 1 else np.nan

        band_rise_times[band] = rise_time
        band_fade_times[band] = fade_time

        # Asymmetry ratio
        if not np.isnan(rise_time) and not np.isnan(fade_time) and fade_time > 0:
            asymmetry = rise_time / fade_time
            band_asymmetries[band] = asymmetry
            features[f'{band}_asymmetry'] = asymmetry
        else:
            band_asymmetries[band] = np.nan
            features[f'{band}_asymmetry'] = np.nan

    # Cross-band comparisons
    valid_bands = [b for b in bands if not np.isnan(band_asymmetries.get(b, np.nan))]

    if len(valid_bands) >= 2:
        asymm_values = [band_asymmetries[b] for b in valid_bands]

        # Dispersion of asymmetry across bands (low = achromatic)
        features['asymmetry_dispersion'] = np.std(asymm_values)

        # Specific band comparisons
        if 'g' in valid_bands and 'r' in valid_bands:
            features['asymmetry_diff_g_r'] = band_asymmetries['g'] - band_asymmetries['r']
        else:
            features['asymmetry_diff_g_r'] = np.nan

        if 'r' in valid_bands and 'i' in valid_bands:
            features['asymmetry_diff_r_i'] = band_asymmetries['r'] - band_asymmetries['i']
        else:
            features['asymmetry_diff_r_i'] = np.nan
    else:
        features['asymmetry_dispersion'] = np.nan
        features['asymmetry_diff_g_r'] = np.nan
        features['asymmetry_diff_r_i'] = np.nan

    # Peak time lags between bands
    if len(band_peak_times) >= 2:
        if 'g' in band_peak_times and 'r' in band_peak_times:
            features['peak_lag_g_r'] = band_peak_times['g'] - band_peak_times['r']
        else:
            features['peak_lag_g_r'] = np.nan

        if 'r' in band_peak_times and 'i' in band_peak_times:
            features['peak_lag_r_i'] = band_peak_times['r'] - band_peak_times['i']
        else:
            features['peak_lag_r_i'] = np.nan

        # Peak time dispersion across all bands
        peak_times = [band_peak_times[b] for b in valid_bands if b in band_peak_times]
        if len(peak_times) >= 2:
            features['peak_time_dispersion'] = np.std(peak_times)
        else:
            features['peak_time_dispersion'] = np.nan
    else:
        features['peak_lag_g_r'] = np.nan
        features['peak_lag_r_i'] = np.nan
        features['peak_time_dispersion'] = np.nan

    # Rise time correlation (do bands rise together?)
    valid_rise = [b for b in bands if not np.isnan(band_rise_times.get(b, np.nan))]
    if len(valid_rise) >= 2:
        rise_values = [band_rise_times[b] for b in valid_rise]
        features['rise_time_dispersion'] = np.std(rise_values) / np.mean(rise_values) if np.mean(rise_values) > 0 else np.nan
    else:
        features['rise_time_dispersion'] = np.nan

    return features


def extract_advanced_physics_features_single(obj_lc: pd.DataFrame) -> Dict[str, float]:
    """
    Extract all advanced physics features for a single object.

    Args:
        obj_lc: Lightcurve DataFrame for one object

    Returns:
        Dictionary of advanced physics features
    """
    features = {}

    # Multi-epoch temperature evolution and cooling
    features.update(compute_multi_epoch_temperatures(obj_lc))

    # Late-time color evolution (100-200 days)
    features.update(compute_late_time_colors(obj_lc))

    # Cross-band temporal asymmetry
    features.update(compute_cross_band_asymmetry(obj_lc))

    return features


def extract_advanced_physics_features(
    lightcurves: pd.DataFrame,
    object_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract advanced physics features for multiple objects.

    Args:
        lightcurves: DataFrame with lightcurve data
        object_ids: Optional list of object IDs

    Returns:
        DataFrame with advanced physics features
    """
    if object_ids is None:
        object_ids = lightcurves['object_id'].unique()

    # Pre-group for efficiency
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if (i + 1) % 500 == 0:
            print(f"    Advanced Physics: {i+1}/{len(object_ids)} objects processed")

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue

        features = extract_advanced_physics_features_single(obj_lc)
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

    print("\nExtracting advanced physics features for first 10 objects...")
    sample_ids = data['train_meta']['object_id'].head(10).tolist()
    adv_features = extract_advanced_physics_features(data['train_lc'], sample_ids)

    print(f"\nExtracted {len(adv_features.columns)-1} advanced physics features")
    print("\nFeature columns:")
    print([c for c in adv_features.columns if c != 'object_id'][:20], "...")
    print("\nSample values:")
    print(adv_features[['object_id', 'cooling_rate_overall', 'g_r_late_100d', 'asymmetry_dispersion']].head())
