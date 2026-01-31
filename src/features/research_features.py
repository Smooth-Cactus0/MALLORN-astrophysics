"""
Research-based features from PLAsTiCC winners and TDE classification literature.

Features implemented based on:
1. Avocado (PLAsTiCC 1st place) - Kyle Boone 2019
2. ALeRCE TDE classifier - 2025 paper
3. FLEET algorithm - 2022 paper
4. PLAsTiCC 3rd place solution - nyanp

Five key feature categories:
1. Power law fit quality (t^-5/3 TDE decay)
2. Nuclear position proxy (host galaxy center indicator)
3. Color at peak (g-r, r-i at maximum brightness)
4. Multi-timescale variability (MHPS - Mexican Hat Power Spectra)
5. Absolute luminosity (using redshift)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import optimize
from scipy.signal import convolve
import warnings

LSST_BANDS = ["u", "g", "r", "i", "z", "y"]

# Cosmology constants for luminosity distance
H0 = 70.0  # km/s/Mpc
c = 299792.458  # km/s
OMEGA_M = 0.3
OMEGA_L = 0.7


# =============================================================================
# 1. POWER LAW FIT QUALITY (t^-5/3 TDE decay)
# =============================================================================

def power_law_model(t, A, alpha, t0):
    """Power law: F(t) = A * (t - t0)^alpha"""
    dt = np.maximum(t - t0, 0.1)
    return A * np.power(dt, alpha)


def fit_power_law_decay(times: np.ndarray, fluxes: np.ndarray,
                        flux_errs: np.ndarray) -> Dict[str, float]:
    """
    Fit power law to post-peak decay and assess quality.

    TDE theoretical: alpha = -5/3 ≈ -1.67 (bolometric)
    Monochromatic optical: alpha ≈ -5/12 ≈ -0.42

    Returns:
        - alpha: fitted power law exponent
        - alpha_deviation: |alpha - (-5/3)|
        - fit_chi2: reduced chi-squared of fit
        - fit_residual_std: standard deviation of residuals
    """
    features = {
        'powerlaw_alpha': np.nan,
        'powerlaw_alpha_deviation_53': np.nan,
        'powerlaw_alpha_deviation_512': np.nan,
        'powerlaw_chi2': np.nan,
        'powerlaw_residual_std': np.nan,
        'powerlaw_fit_success': 0
    }

    if len(times) < 5:
        return features

    # Find peak
    peak_idx = np.argmax(fluxes)
    peak_time = times[peak_idx]
    peak_flux = fluxes[peak_idx]

    # Post-peak data (at least 10 days after peak)
    post_mask = (times > peak_time + 10) & (fluxes > 0)
    post_times = times[post_mask]
    post_fluxes = fluxes[post_mask]
    post_errs = flux_errs[post_mask] if flux_errs is not None else np.ones_like(post_fluxes)

    if len(post_times) < 4:
        return features

    # Log-space fitting is more stable
    dt = post_times - peak_time
    log_t = np.log10(dt)
    log_f = np.log10(post_fluxes)

    # Simple linear regression in log-log space: log(F) = alpha * log(t) + c
    try:
        coeffs, cov = np.polyfit(log_t, log_f, 1, cov=True)
        alpha = coeffs[0]

        features['powerlaw_alpha'] = alpha
        features['powerlaw_alpha_deviation_53'] = np.abs(alpha - (-5/3))  # Deviation from bolometric
        features['powerlaw_alpha_deviation_512'] = np.abs(alpha - (-5/12))  # Deviation from monochromatic

        # Compute residuals
        predicted = coeffs[0] * log_t + coeffs[1]
        residuals = log_f - predicted
        features['powerlaw_residual_std'] = np.std(residuals)

        # Reduced chi-squared
        if len(post_errs) > 2:
            # Convert errors to log space approximately
            log_errs = post_errs / (post_fluxes * np.log(10) + 1e-10)
            log_errs = np.clip(log_errs, 0.01, 1.0)
            chi2 = np.sum((residuals / log_errs) ** 2)
            dof = len(residuals) - 2
            features['powerlaw_chi2'] = chi2 / max(dof, 1)

        features['powerlaw_fit_success'] = 1

    except Exception:
        pass

    return features


def compute_power_law_features(obj_lc: pd.DataFrame,
                               bands: List[str] = ['g', 'r', 'i']) -> Dict[str, float]:
    """Extract power law fit features for multiple bands."""
    features = {}

    for band in bands:
        band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_data) < 5:
            for key in ['powerlaw_alpha', 'powerlaw_alpha_deviation_53',
                       'powerlaw_alpha_deviation_512', 'powerlaw_chi2',
                       'powerlaw_residual_std', 'powerlaw_fit_success']:
                features[f'{band}_{key}'] = np.nan
            continue

        times = band_data['Time (MJD)'].values
        fluxes = band_data['Flux'].values
        flux_errs = band_data['Flux_err'].values if 'Flux_err' in band_data.columns else None

        band_features = fit_power_law_decay(times, fluxes, flux_errs)
        for key, value in band_features.items():
            features[f'{band}_{key}'] = value

    # Combined features across optical bands
    alphas = [features.get(f'{b}_powerlaw_alpha', np.nan) for b in bands]
    alphas = [a for a in alphas if not np.isnan(a)]

    if len(alphas) >= 2:
        features['optical_mean_powerlaw_alpha'] = np.mean(alphas)
        features['optical_std_powerlaw_alpha'] = np.std(alphas)
        features['optical_mean_deviation_53'] = np.mean([np.abs(a - (-5/3)) for a in alphas])
    else:
        features['optical_mean_powerlaw_alpha'] = alphas[0] if alphas else np.nan
        features['optical_std_powerlaw_alpha'] = np.nan
        features['optical_mean_deviation_53'] = np.abs(alphas[0] - (-5/3)) if alphas else np.nan

    return features


# =============================================================================
# 2. NUCLEAR POSITION PROXY
# =============================================================================

def compute_nuclear_position_proxy(obj_lc: pd.DataFrame,
                                   metadata: Optional[Dict] = None) -> Dict[str, float]:
    """
    Compute proxy features for nuclear position (TDEs occur at galaxy centers).

    Without direct host galaxy data, we use:
    1. Light curve smoothness (nuclear sources tend to be smoother)
    2. Flux concentration ratio (nuclear = more concentrated)
    3. Variability structure function (AGN-like vs transient-like)

    Based on ALeRCE's mean_distnr concept adapted for photometry-only.
    """
    features = {
        'nuclear_smoothness': np.nan,
        'nuclear_concentration': np.nan,
        'nuclear_variability_ratio': np.nan,
        'nuclear_position_score': np.nan
    }

    # Use r-band as reference (best coverage typically)
    r_data = obj_lc[obj_lc['Filter'] == 'r'].sort_values('Time (MJD)')

    if len(r_data) < 10:
        return features

    times = r_data['Time (MJD)'].values
    fluxes = r_data['Flux'].values
    flux_errs = r_data['Flux_err'].values if 'Flux_err' in r_data.columns else np.ones_like(fluxes)

    # 1. Smoothness: ratio of flux changes to flux errors
    # Smooth = low ratio (changes are within noise)
    flux_diffs = np.abs(np.diff(fluxes))
    time_diffs = np.diff(times)
    rate_of_change = flux_diffs / (time_diffs + 0.1)

    # Normalize by typical flux error
    median_err = np.median(flux_errs)
    if median_err > 0:
        smoothness = np.median(rate_of_change) / median_err
        features['nuclear_smoothness'] = 1.0 / (1.0 + smoothness)  # Higher = smoother

    # 2. Concentration: ratio of peak flux to baseline flux
    # Nuclear transients have higher concentration
    peak_flux = np.max(fluxes)
    baseline_flux = np.percentile(fluxes, 10)

    if baseline_flux > 0:
        features['nuclear_concentration'] = peak_flux / baseline_flux
    elif peak_flux > 0:
        features['nuclear_concentration'] = peak_flux / np.median(np.abs(fluxes) + 1)

    # 3. Variability ratio: short-term vs long-term variability
    # AGN have persistent variability, TDEs have single-event structure
    if len(times) >= 20:
        # Short-term: within 10-day windows
        short_var = []
        for i in range(len(times) - 5):
            if times[i+5] - times[i] < 15:
                short_var.append(np.std(fluxes[i:i+5]))

        # Long-term: overall variance
        long_var = np.std(fluxes)

        if len(short_var) > 0 and long_var > 0:
            features['nuclear_variability_ratio'] = np.mean(short_var) / long_var

    # 4. Combined nuclear position score (higher = more likely nuclear)
    scores = []
    if not np.isnan(features['nuclear_smoothness']):
        scores.append(features['nuclear_smoothness'])
    if not np.isnan(features['nuclear_concentration']):
        # Normalize concentration to 0-1 range
        scores.append(min(1.0, features['nuclear_concentration'] / 100))
    if not np.isnan(features['nuclear_variability_ratio']):
        # Low variability ratio = more TDE-like
        scores.append(1.0 - min(1.0, features['nuclear_variability_ratio']))

    if scores:
        features['nuclear_position_score'] = np.mean(scores)

    return features


# =============================================================================
# 3. COLOR AT PEAK
# =============================================================================

def compute_color_at_peak(obj_lc: pd.DataFrame,
                          color_pairs: List[Tuple[str, str]] = [('g', 'r'), ('r', 'i')]) -> Dict[str, float]:
    """
    Compute color measurements specifically at peak brightness.

    Key insight from FLEET: (g-r) color at peak is one of the most
    discriminative features for TDE classification.

    TDEs: Blue at peak (g-r ~ -0.3 to 0.0)
    SNe: Variable, often redder
    AGN: Variable
    """
    features = {}

    for b1, b2 in color_pairs:
        features[f'{b1}_{b2}_color_at_peak'] = np.nan
        features[f'{b1}_{b2}_color_peak_to_late'] = np.nan

    # Find overall peak (use r-band or brightest available)
    r_data = obj_lc[obj_lc['Filter'] == 'r']
    if len(r_data) < 3:
        g_data = obj_lc[obj_lc['Filter'] == 'g']
        if len(g_data) < 3:
            return features
        peak_time = g_data.loc[g_data['Flux'].idxmax(), 'Time (MJD)']
    else:
        peak_time = r_data.loc[r_data['Flux'].idxmax(), 'Time (MJD)']

    for b1, b2 in color_pairs:
        band1 = obj_lc[obj_lc['Filter'] == b1]
        band2 = obj_lc[obj_lc['Filter'] == b2]

        if len(band1) < 2 or len(band2) < 2:
            continue

        # Find observations near peak (within 10 days)
        peak_window = 10

        b1_near_peak = band1[np.abs(band1['Time (MJD)'] - peak_time) < peak_window]
        b2_near_peak = band2[np.abs(band2['Time (MJD)'] - peak_time) < peak_window]

        if len(b1_near_peak) > 0 and len(b2_near_peak) > 0:
            # Get closest observations to peak
            b1_peak = b1_near_peak.iloc[np.argmin(np.abs(b1_near_peak['Time (MJD)'] - peak_time))]
            b2_peak = b2_near_peak.iloc[np.argmin(np.abs(b2_near_peak['Time (MJD)'] - peak_time))]

            f1 = b1_peak['Flux']
            f2 = b2_peak['Flux']

            if f1 > 0 and f2 > 0:
                # Color in magnitude (negative = bluer)
                color_at_peak = -2.5 * np.log10(f1 / f2)
                features[f'{b1}_{b2}_color_at_peak'] = color_at_peak

                # Compare to late-time color (50+ days after peak)
                b1_late = band1[band1['Time (MJD)'] > peak_time + 50]
                b2_late = band2[band2['Time (MJD)'] > peak_time + 50]

                if len(b1_late) > 0 and len(b2_late) > 0:
                    # Match closest observations
                    colors_late = []
                    for _, row1 in b1_late.iterrows():
                        t1 = row1['Time (MJD)']
                        f1_late = row1['Flux']

                        dt = np.abs(b2_late['Time (MJD)'].values - t1)
                        min_idx = np.argmin(dt)

                        if dt[min_idx] < 5:  # Within 5 days
                            f2_late = b2_late.iloc[min_idx]['Flux']
                            if f1_late > 0 and f2_late > 0:
                                colors_late.append(-2.5 * np.log10(f1_late / f2_late))

                    if colors_late:
                        # Change from peak to late (positive = reddening)
                        features[f'{b1}_{b2}_color_peak_to_late'] = np.mean(colors_late) - color_at_peak

    return features


# =============================================================================
# 4. MULTI-TIMESCALE VARIABILITY (MHPS - Mexican Hat Power Spectra)
# =============================================================================

def mexican_hat_wavelet(scale: float, length: int) -> np.ndarray:
    """
    Generate Mexican Hat (Ricker) wavelet.

    MH(t) = (1 - (t/scale)^2) * exp(-t^2 / (2*scale^2))
    """
    t = np.linspace(-length//2, length//2, length)
    normalized_t = t / scale
    wavelet = (1 - normalized_t**2) * np.exp(-normalized_t**2 / 2)
    # Normalize to unit energy
    wavelet = wavelet / np.sqrt(np.sum(wavelet**2))
    return wavelet


def compute_mhps_features(obj_lc: pd.DataFrame,
                         timescales: List[float] = [10, 30, 100],
                         band: str = 'r') -> Dict[str, float]:
    """
    Compute Mexican Hat Power Spectra at multiple timescales.

    Based on ALeRCE TDE classifier paper:
    - MHPS measures variability power at different frequencies
    - Ratios like MHPS_10/MHPS_100 distinguish transients from AGN

    TDEs: Strong power at intermediate timescales (single event)
    AGN: Power across all timescales (stochastic)
    SNe: Sharp peak at short timescales
    """
    features = {}

    # Initialize all features
    for ts in timescales:
        features[f'mhps_{int(ts)}d'] = np.nan
    features['mhps_10_100_ratio'] = np.nan
    features['mhps_30_100_ratio'] = np.nan
    features['mhps_dominant_scale'] = np.nan

    band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

    if len(band_data) < 20:
        return features

    times = band_data['Time (MJD)'].values
    fluxes = band_data['Flux'].values

    # Interpolate to regular grid (1-day spacing)
    time_span = times[-1] - times[0]
    if time_span < 50:
        return features

    # Create regular time grid
    t_regular = np.arange(times[0], times[-1], 1.0)

    # Linear interpolation
    f_regular = np.interp(t_regular, times, fluxes)

    # Remove mean (detrend)
    f_regular = f_regular - np.mean(f_regular)

    mhps_values = {}

    for scale in timescales:
        # Wavelet length (5 sigma coverage)
        wavelet_len = int(min(5 * scale, len(f_regular) // 2))
        if wavelet_len < 5:
            continue

        wavelet = mexican_hat_wavelet(scale, wavelet_len)

        # Convolve with wavelet
        convolved = convolve(f_regular, wavelet, mode='same')

        # Power = sum of squared coefficients
        power = np.sum(convolved**2) / len(convolved)

        features[f'mhps_{int(scale)}d'] = power
        mhps_values[scale] = power

    # Compute ratios
    if 10 in mhps_values and 100 in mhps_values and mhps_values[100] > 0:
        features['mhps_10_100_ratio'] = mhps_values[10] / mhps_values[100]

    if 30 in mhps_values and 100 in mhps_values and mhps_values[100] > 0:
        features['mhps_30_100_ratio'] = mhps_values[30] / mhps_values[100]

    # Dominant scale (where power is maximum)
    if mhps_values:
        dominant = max(mhps_values, key=mhps_values.get)
        features['mhps_dominant_scale'] = dominant

    return features


# =============================================================================
# 5. ABSOLUTE LUMINOSITY
# =============================================================================

def luminosity_distance_mpc(z: float) -> float:
    """
    Compute luminosity distance in Mpc for a given redshift.

    Uses flat LCDM cosmology with H0=70, Omega_M=0.3, Omega_L=0.7

    Simplified formula valid for z < 2:
    D_L ≈ (c/H0) * z * (1 + z/2 * (1 - 3*Omega_M/4))
    """
    if z <= 0 or np.isnan(z):
        return np.nan

    # More accurate numerical integration for higher z
    if z < 0.1:
        # Low-z approximation: D_L ≈ c*z/H0
        d_l = (c / H0) * z * (1 + z/2)
    else:
        # Simple approximation for moderate z
        # D_L = (c/H0) * z * (1 + 0.5*(1-q0)*z) where q0 ≈ 0.5*Omega_M - Omega_L
        q0 = 0.5 * OMEGA_M - OMEGA_L
        d_l = (c / H0) * z * (1 + 0.5 * (1 - q0) * z)

    return d_l


def compute_luminosity_features(obj_lc: pd.DataFrame,
                                redshift: float,
                                bands: List[str] = ['g', 'r', 'i']) -> Dict[str, float]:
    """
    Convert apparent flux to absolute luminosity.

    Based on PLAsTiCC 3rd place solution:
    luminosity ~ (max_flux - min_flux) * luminosity_distance^2

    Features:
    - Peak absolute luminosity (in relative units)
    - Luminosity amplitude
    - Luminosity decline rate
    """
    features = {
        'luminosity_distance_mpc': np.nan,
        'peak_luminosity': np.nan,
        'luminosity_amplitude': np.nan,
        'mean_luminosity': np.nan,
        'luminosity_decline_rate': np.nan
    }

    # Compute luminosity distance
    d_l = luminosity_distance_mpc(redshift)
    if np.isnan(d_l):
        return features

    features['luminosity_distance_mpc'] = d_l

    # Combine optical bands for better coverage
    optical_data = obj_lc[obj_lc['Filter'].isin(bands)].copy()

    if len(optical_data) < 5:
        return features

    # Sort by time
    optical_data = optical_data.sort_values('Time (MJD)')
    times = optical_data['Time (MJD)'].values
    fluxes = optical_data['Flux'].values

    # Convert to luminosity (relative units, scaled by D_L^2)
    # L = 4 * pi * D_L^2 * F  (we drop the 4*pi constant)
    luminosities = fluxes * (d_l ** 2)

    # Peak luminosity
    features['peak_luminosity'] = np.max(luminosities)

    # Luminosity amplitude (peak - baseline)
    baseline = np.percentile(luminosities, 10)
    features['luminosity_amplitude'] = np.max(luminosities) - baseline

    # Mean luminosity
    features['mean_luminosity'] = np.mean(luminosities)

    # Luminosity decline rate (post-peak)
    peak_idx = np.argmax(luminosities)
    if peak_idx < len(luminosities) - 5:
        post_peak_lum = luminosities[peak_idx:]
        post_peak_times = times[peak_idx:]

        # Decline rate in log space
        if len(post_peak_lum) >= 3 and np.min(post_peak_lum) > 0:
            dt = post_peak_times - post_peak_times[0]
            log_lum = np.log10(post_peak_lum)

            # Linear fit to log(L) vs t
            if np.std(dt) > 0:
                coeffs = np.polyfit(dt, log_lum, 1)
                features['luminosity_decline_rate'] = coeffs[0] * 100  # per 100 days

    return features


# =============================================================================
# COMBINED FEATURE EXTRACTION
# =============================================================================

def extract_research_features_single(obj_lc: pd.DataFrame,
                                    metadata: Optional[Dict] = None) -> Dict[str, float]:
    """
    Extract all research-based features for a single object.

    Args:
        obj_lc: Light curve DataFrame for one object
        metadata: Optional dict with 'Z' (redshift) key

    Returns:
        Dictionary of features
    """
    features = {}

    # 1. Power law fit quality
    features.update(compute_power_law_features(obj_lc))

    # 2. Nuclear position proxy
    features.update(compute_nuclear_position_proxy(obj_lc, metadata))

    # 3. Color at peak
    features.update(compute_color_at_peak(obj_lc))

    # 4. MHPS multi-timescale variability
    features.update(compute_mhps_features(obj_lc))

    # 5. Luminosity features (requires redshift)
    if metadata and 'Z' in metadata and metadata['Z'] > 0:
        features.update(compute_luminosity_features(obj_lc, metadata['Z']))
    else:
        # Add NaN luminosity features
        for key in ['luminosity_distance_mpc', 'peak_luminosity',
                    'luminosity_amplitude', 'mean_luminosity', 'luminosity_decline_rate']:
            features[key] = np.nan

    return features


def extract_research_features(lightcurves: pd.DataFrame,
                             object_ids: List[str],
                             metadata_df: Optional[pd.DataFrame] = None,
                             verbose: bool = True) -> pd.DataFrame:
    """
    Extract research-based features for multiple objects.

    Args:
        lightcurves: DataFrame with all light curves
        object_ids: List of object IDs to process
        metadata_df: Optional DataFrame with object metadata (must have 'object_id' and 'Z')
        verbose: Print progress

    Returns:
        DataFrame with research features for each object
    """
    # Pre-group light curves for efficiency
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    # Create metadata lookup
    if metadata_df is not None:
        meta_lookup = metadata_df.set_index('object_id').to_dict('index')
    else:
        meta_lookup = {}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if verbose and (i + 1) % 500 == 0:
            print(f"    Research Features: {i+1}/{len(object_ids)} objects processed")

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue

        # Get metadata for this object
        obj_meta = meta_lookup.get(obj_id, {})

        features = extract_research_features_single(obj_lc, obj_meta)
        features['object_id'] = obj_id
        all_features.append(features)

    return pd.DataFrame(all_features)


# =============================================================================
# MAIN / TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_all_data

    print("Loading data...")
    data = load_all_data()

    print("\nExtracting research features for first 50 objects...")
    sample_ids = data['train_meta']['object_id'].head(50).tolist()

    research_features = extract_research_features(
        data['train_lc'],
        sample_ids,
        data['train_meta'],
        verbose=True
    )

    print(f"\nExtracted {len(research_features.columns)-1} research features")
    print("\nFeature columns:")
    for col in sorted([c for c in research_features.columns if c != 'object_id']):
        print(f"  - {col}")

    print("\nSample values (first 5 objects):")
    key_features = ['r_powerlaw_alpha', 'optical_mean_deviation_53',
                    'nuclear_position_score', 'g_r_color_at_peak',
                    'mhps_30_100_ratio', 'peak_luminosity']
    existing_features = [f for f in key_features if f in research_features.columns]
    print(research_features[['object_id'] + existing_features].head())
