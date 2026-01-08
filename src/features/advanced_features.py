"""
Advanced feature extraction for MALLORN classification.

Based on research from:
1. 2025 TDE paper (arxiv.org/abs/2509.25902) - GP features, post-peak colors
2. ALeRCE classifier - MHPS, FLEET model, power-law decay
3. PLAsTiCC winners - absolute magnitude, tsfresh-style features

New features added:
- Mexican Hat Power Spectrum (MHPS): Variability at different timescales
- FLEET model parameters: Rise/decline width and asymmetry
- Absolute magnitude: Luminosity corrected for distance
- Pre-peak color evolution: Early color signatures
- GP residuals and derivatives: Smoothness and rate of change
- Autocorrelation features: Periodicity detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
from scipy import signal
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')

LSST_BANDS = ['u', 'g', 'r', 'i', 'z', 'y']

# Band effective wavelengths (Angstroms)
BAND_WAVELENGTHS = {
    'u': 3670, 'g': 4825, 'r': 6222,
    'i': 7545, 'z': 8691, 'y': 9710
}


def compute_absolute_magnitude(flux: float, redshift: float, band: str = 'r') -> float:
    """
    Compute absolute magnitude from flux and redshift.

    Uses simple cosmology: d_L = c*z/H0 for z < 0.3, more accurate for higher z.

    Args:
        flux: Flux in microJy
        redshift: Spectroscopic redshift
        band: Filter band for K-correction approximation

    Returns:
        Absolute magnitude (AB system)
    """
    if flux <= 0 or np.isnan(flux) or np.isnan(redshift) or redshift <= 0:
        return np.nan

    # Convert flux to AB magnitude
    # m_AB = -2.5 * log10(flux_Jy) + 8.90
    # flux is in microJy, so flux_Jy = flux * 1e-6
    m_AB = -2.5 * np.log10(flux * 1e-6) + 8.90

    # Luminosity distance (Mpc) - simplified flat Lambda-CDM
    # For z < 0.1: d_L ~ c*z/H0
    # For higher z: integrate properly
    c = 299792.458  # km/s
    H0 = 70.0  # km/s/Mpc

    if redshift < 0.1:
        d_L = c * redshift / H0
    else:
        # Simple integral approximation for flat Lambda-CDM (Omega_m=0.3)
        from scipy.integrate import quad

        def E(z):
            return np.sqrt(0.3 * (1 + z)**3 + 0.7)

        integral, _ = quad(lambda z: 1/E(z), 0, redshift)
        d_L = (c / H0) * (1 + redshift) * integral

    # Distance modulus
    if d_L > 0:
        mu = 5 * np.log10(d_L) + 25  # d_L in Mpc
    else:
        return np.nan

    # Simple K-correction approximation (depends on SED shape)
    # For blue sources (TDEs): K ~ -2.5 * log10(1+z)
    # For red sources: K can be positive
    K_corr = -2.5 * np.log10(1 + redshift)  # Approximate for flat spectrum

    M_abs = m_AB - mu - K_corr

    return M_abs


def mexican_hat_wavelet(times: np.ndarray, fluxes: np.ndarray,
                        scale: float) -> float:
    """
    Compute Mexican Hat (Ricker) wavelet coefficient at given scale.

    This measures variability amplitude at a specific timescale.

    Args:
        times: Observation times (days)
        fluxes: Flux values
        scale: Wavelet scale (days)

    Returns:
        Wavelet coefficient (variability amplitude at this scale)
    """
    if len(times) < 5:
        return np.nan

    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    fluxes = fluxes[sort_idx]

    # Normalize flux
    mean_flux = np.mean(fluxes)
    if mean_flux == 0:
        return np.nan
    norm_flux = (fluxes - mean_flux) / mean_flux

    # Compute Mexican Hat wavelet coefficients
    # Mexican Hat: (1 - t^2/a^2) * exp(-t^2/(2*a^2)) / (sqrt(3*a) * pi^(1/4))
    total_coeff = 0.0
    n_pairs = 0

    for i, (t1, f1) in enumerate(zip(times, norm_flux)):
        for j, (t2, f2) in enumerate(zip(times, norm_flux)):
            if i >= j:
                continue

            dt = abs(t2 - t1)
            t_norm = dt / scale

            # Mexican Hat kernel
            if t_norm < 5:  # Truncate at 5 sigma
                kernel = (1 - t_norm**2) * np.exp(-t_norm**2 / 2)
                total_coeff += (f2 - f1)**2 * abs(kernel)
                n_pairs += 1

    if n_pairs > 0:
        return np.sqrt(total_coeff / n_pairs)
    return np.nan


def compute_mhps_features(times: np.ndarray, fluxes: np.ndarray) -> Dict[str, float]:
    """
    Compute Mexican Hat Power Spectrum (MHPS) features.

    From ALeRCE: Variability at different characteristic timescales.
    AGN show variability at all scales, transients have characteristic scales.

    Args:
        times: Observation times
        fluxes: Flux values

    Returns:
        Dictionary with MHPS features at different scales
    """
    features = {}

    # Characteristic timescales
    scales = {
        '10': 10,    # Short-term variability
        '30': 30,    # ~1 month
        '100': 100,  # ~3 months
        '365': 365   # ~1 year
    }

    mhps_values = {}
    for name, scale in scales.items():
        mhps = mexican_hat_wavelet(times, fluxes, scale)
        features[f'mhps_{name}'] = mhps
        mhps_values[name] = mhps

    # Ratios between scales (diagnostic for transient type)
    if not np.isnan(mhps_values.get('10', np.nan)) and not np.isnan(mhps_values.get('100', np.nan)):
        if mhps_values['100'] > 0:
            features['mhps_ratio_10_100'] = mhps_values['10'] / mhps_values['100']
        else:
            features['mhps_ratio_10_100'] = np.nan
    else:
        features['mhps_ratio_10_100'] = np.nan

    if not np.isnan(mhps_values.get('30', np.nan)) and not np.isnan(mhps_values.get('365', np.nan)):
        if mhps_values['365'] > 0:
            features['mhps_ratio_30_365'] = mhps_values['30'] / mhps_values['365']
        else:
            features['mhps_ratio_30_365'] = np.nan
    else:
        features['mhps_ratio_30_365'] = np.nan

    return features


def fit_fleet_model(times: np.ndarray, fluxes: np.ndarray) -> Dict[str, float]:
    """
    Fit FLEET-style exponential model to lightcurve.

    Model: f(t) = A * exp(-|t-t0|/W) * (1 + alpha * sign(t-t0))

    Extracts:
    - W: Effective width (characteristic timescale)
    - A: Rise-to-decline modifier (asymmetry)

    From: FLEET algorithm for TDE classification.
    """
    features = {
        'fleet_width': np.nan,
        'fleet_asymmetry': np.nan,
        'fleet_chi2': np.nan
    }

    if len(times) < 5:
        return features

    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    fluxes = fluxes[sort_idx]

    # Find peak
    peak_idx = np.argmax(fluxes)
    peak_time = times[peak_idx]
    peak_flux = fluxes[peak_idx]

    if peak_flux <= 0:
        return features

    # Split into rise and fall
    rise_mask = times < peak_time
    fall_mask = times > peak_time

    rise_times = times[rise_mask] if np.any(rise_mask) else np.array([])
    rise_fluxes = fluxes[rise_mask] if np.any(rise_mask) else np.array([])
    fall_times = times[fall_mask] if np.any(fall_mask) else np.array([])
    fall_fluxes = fluxes[fall_mask] if np.any(fall_mask) else np.array([])

    # Fit exponential to rise (if enough data)
    rise_tau = np.nan
    if len(rise_times) >= 3:
        dt_rise = peak_time - rise_times
        # log(flux/peak) = -dt/tau
        valid = rise_fluxes > 0
        if np.sum(valid) >= 3:
            log_ratio = np.log(rise_fluxes[valid] / peak_flux)
            dt = dt_rise[valid]
            if np.std(dt) > 0:
                # Linear regression: log_ratio = -dt/tau => tau = -dt/log_ratio
                slope = np.polyfit(dt, log_ratio, 1)[0]
                if slope < 0:
                    rise_tau = -1 / slope

    # Fit exponential to fall (if enough data)
    fall_tau = np.nan
    if len(fall_times) >= 3:
        dt_fall = fall_times - peak_time
        valid = fall_fluxes > 0
        if np.sum(valid) >= 3:
            log_ratio = np.log(fall_fluxes[valid] / peak_flux)
            dt = dt_fall[valid]
            if np.std(dt) > 0:
                slope = np.polyfit(dt, log_ratio, 1)[0]
                if slope < 0:
                    fall_tau = -1 / slope

    # Combined width and asymmetry
    if not np.isnan(rise_tau) and not np.isnan(fall_tau):
        features['fleet_width'] = (rise_tau + fall_tau) / 2
        features['fleet_asymmetry'] = fall_tau / rise_tau if rise_tau > 0 else np.nan
    elif not np.isnan(fall_tau):
        features['fleet_width'] = fall_tau
        features['fleet_asymmetry'] = np.nan
    elif not np.isnan(rise_tau):
        features['fleet_width'] = rise_tau
        features['fleet_asymmetry'] = np.nan

    return features


def compute_pre_peak_colors(obj_lc: pd.DataFrame, peak_time: float) -> Dict[str, float]:
    """
    Compute color features in the pre-peak phase.

    TDEs are blue from the start, SNe may start redder.
    """
    features = {
        'pre_peak_g_r_mean': np.nan,
        'pre_peak_r_i_mean': np.nan,
        'pre_peak_g_r_slope': np.nan,
        'pre_peak_r_i_slope': np.nan
    }

    if np.isnan(peak_time):
        return features

    # Get pre-peak data
    for b1, b2 in [('g', 'r'), ('r', 'i')]:
        band1 = obj_lc[(obj_lc['Filter'] == b1) & (obj_lc['Time (MJD)'] < peak_time)]
        band2 = obj_lc[(obj_lc['Filter'] == b2) & (obj_lc['Time (MJD)'] < peak_time)]

        if len(band1) < 2 or len(band2) < 2:
            continue

        colors = []
        times = []

        for _, row1 in band1.iterrows():
            t1, f1 = row1['Time (MJD)'], row1['Flux']

            dt = np.abs(band2['Time (MJD)'].values - t1)
            if len(dt) == 0:
                continue
            min_idx = np.argmin(dt)

            if dt[min_idx] < 5 and f1 > 0:
                f2 = band2.iloc[min_idx]['Flux']
                if f2 > 0:
                    color = -2.5 * np.log10(f1 / f2)
                    colors.append(color)
                    times.append(t1)

        if len(colors) >= 2:
            features[f'pre_peak_{b1}_{b2}_mean'] = np.mean(colors)

            if len(colors) >= 3:
                coeffs = np.polyfit(np.array(times) - times[0], colors, 1)
                features[f'pre_peak_{b1}_{b2}_slope'] = coeffs[0] * 10  # per 10 days

    return features


def compute_autocorrelation_features(times: np.ndarray, fluxes: np.ndarray) -> Dict[str, float]:
    """
    Compute autocorrelation features for periodicity/structure detection.

    AGN may show quasi-periodic behavior, transients don't.
    """
    features = {
        'acf_10d': np.nan,
        'acf_30d': np.nan,
        'acf_ratio': np.nan
    }

    if len(times) < 10:
        return features

    # Interpolate to regular grid for ACF
    t_min, t_max = times.min(), times.max()
    t_range = t_max - t_min

    if t_range < 30:
        return features

    # Use 1-day grid
    t_grid = np.arange(t_min, t_max, 1.0)

    if len(t_grid) < 20:
        return features

    # Simple linear interpolation
    flux_grid = np.interp(t_grid, times, fluxes)

    # Normalize
    flux_grid = (flux_grid - np.mean(flux_grid)) / (np.std(flux_grid) + 1e-10)

    # Compute autocorrelation
    n = len(flux_grid)
    acf = np.correlate(flux_grid, flux_grid, mode='full')[n-1:] / n

    # ACF at specific lags
    if len(acf) > 10:
        features['acf_10d'] = acf[10]

    if len(acf) > 30:
        features['acf_30d'] = acf[30]

    if not np.isnan(features['acf_10d']) and not np.isnan(features['acf_30d']):
        if abs(features['acf_30d']) > 0.01:
            features['acf_ratio'] = features['acf_10d'] / features['acf_30d']

    return features


def compute_early_late_features(obj_lc: pd.DataFrame, bands: List[str] = ['g', 'r', 'i']) -> Dict[str, float]:
    """
    Compute features comparing early vs late behavior.

    TDEs: Smooth transition
    SNe: May have plateau, bump, or cliff
    AGN: Irregular
    """
    features = {}

    all_times = obj_lc['Time (MJD)'].values
    if len(all_times) < 10:
        for band in bands:
            features[f'{band}_early_late_flux_ratio'] = np.nan
            features[f'{band}_early_late_var_ratio'] = np.nan
        return features

    # Define early (first 1/3) and late (last 1/3)
    t_range = all_times.max() - all_times.min()
    t_early_end = all_times.min() + t_range / 3
    t_late_start = all_times.max() - t_range / 3

    for band in bands:
        band_data = obj_lc[obj_lc['Filter'] == band]

        if len(band_data) < 5:
            features[f'{band}_early_late_flux_ratio'] = np.nan
            features[f'{band}_early_late_var_ratio'] = np.nan
            continue

        early = band_data[band_data['Time (MJD)'] < t_early_end]['Flux'].values
        late = band_data[band_data['Time (MJD)'] > t_late_start]['Flux'].values

        if len(early) >= 2 and len(late) >= 2:
            early_mean = np.mean(early)
            late_mean = np.mean(late)

            if early_mean > 0:
                features[f'{band}_early_late_flux_ratio'] = late_mean / early_mean
            else:
                features[f'{band}_early_late_flux_ratio'] = np.nan

            early_var = np.var(early)
            late_var = np.var(late)

            if early_var > 0:
                features[f'{band}_early_late_var_ratio'] = late_var / early_var
            else:
                features[f'{band}_early_late_var_ratio'] = np.nan
        else:
            features[f'{band}_early_late_flux_ratio'] = np.nan
            features[f'{band}_early_late_var_ratio'] = np.nan

    return features


def compute_higher_order_stats(fluxes: np.ndarray) -> Dict[str, float]:
    """
    Compute higher-order statistical moments.

    Skewness and kurtosis can distinguish distribution shapes.
    """
    features = {
        'flux_skewness': np.nan,
        'flux_kurtosis': np.nan,
        'flux_biweight': np.nan
    }

    if len(fluxes) < 5:
        return features

    features['flux_skewness'] = skew(fluxes)
    features['flux_kurtosis'] = kurtosis(fluxes)

    # Biweight midvariance (robust variability measure)
    median = np.median(fluxes)
    mad = np.median(np.abs(fluxes - median))

    if mad > 0:
        u = (fluxes - median) / (9 * mad)
        valid = np.abs(u) < 1

        if np.sum(valid) >= 3:
            num = np.sum((fluxes[valid] - median)**2 * (1 - u[valid]**2)**4)
            denom = np.sum((1 - u[valid]**2) * (1 - 5*u[valid]**2))**2

            if denom > 0:
                features['flux_biweight'] = len(fluxes) * num / denom

    return features


def extract_advanced_features_single(obj_lc: pd.DataFrame, redshift: float) -> Dict[str, float]:
    """
    Extract all advanced features for a single object.

    Args:
        obj_lc: DataFrame with lightcurve data
        redshift: Object redshift

    Returns:
        Dictionary with advanced features
    """
    features = {}

    # Get band-organized data
    band_data = {}
    for band in LSST_BANDS:
        band_lc = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')
        if len(band_lc) >= 3:
            band_data[band] = {
                'times': band_lc['Time (MJD)'].values,
                'fluxes': band_lc['Flux'].values,
                'errors': band_lc['Flux_err'].values
            }

    # Find r-band peak time (reference)
    peak_time = np.nan
    if 'r' in band_data:
        peak_idx = np.argmax(band_data['r']['fluxes'])
        peak_time = band_data['r']['times'][peak_idx]

    # === Absolute Magnitude Features ===
    for band in ['g', 'r', 'i']:
        if band in band_data:
            peak_flux = np.max(band_data[band]['fluxes'])
            features[f'{band}_abs_mag_peak'] = compute_absolute_magnitude(peak_flux, redshift, band)

            mean_flux = np.mean(band_data[band]['fluxes'])
            features[f'{band}_abs_mag_mean'] = compute_absolute_magnitude(mean_flux, redshift, band)
        else:
            features[f'{band}_abs_mag_peak'] = np.nan
            features[f'{band}_abs_mag_mean'] = np.nan

    # === MHPS Features ===
    # Use r-band for MHPS (best sampled usually)
    if 'r' in band_data:
        mhps = compute_mhps_features(band_data['r']['times'], band_data['r']['fluxes'])
        for key, val in mhps.items():
            features[f'r_{key}'] = val
    else:
        for scale in ['10', '30', '100', '365']:
            features[f'r_mhps_{scale}'] = np.nan
        features['r_mhps_ratio_10_100'] = np.nan
        features['r_mhps_ratio_30_365'] = np.nan

    # Also compute MHPS for g-band
    if 'g' in band_data:
        mhps_g = compute_mhps_features(band_data['g']['times'], band_data['g']['fluxes'])
        for key, val in mhps_g.items():
            features[f'g_{key}'] = val
    else:
        for scale in ['10', '30', '100', '365']:
            features[f'g_mhps_{scale}'] = np.nan
        features['g_mhps_ratio_10_100'] = np.nan
        features['g_mhps_ratio_30_365'] = np.nan

    # === FLEET Model Features ===
    for band in ['r', 'g']:
        if band in band_data:
            fleet = fit_fleet_model(band_data[band]['times'], band_data[band]['fluxes'])
            for key, val in fleet.items():
                features[f'{band}_{key}'] = val
        else:
            features[f'{band}_fleet_width'] = np.nan
            features[f'{band}_fleet_asymmetry'] = np.nan
            features[f'{band}_fleet_chi2'] = np.nan

    # === Pre-peak Color Features ===
    pre_peak = compute_pre_peak_colors(obj_lc, peak_time)
    features.update(pre_peak)

    # === Autocorrelation Features ===
    if 'r' in band_data:
        acf = compute_autocorrelation_features(band_data['r']['times'], band_data['r']['fluxes'])
        for key, val in acf.items():
            features[f'r_{key}'] = val
    else:
        features['r_acf_10d'] = np.nan
        features['r_acf_30d'] = np.nan
        features['r_acf_ratio'] = np.nan

    # === Early vs Late Features ===
    early_late = compute_early_late_features(obj_lc)
    features.update(early_late)

    # === Higher-order Statistics ===
    all_fluxes = obj_lc['Flux'].values
    hos = compute_higher_order_stats(all_fluxes)
    features.update(hos)

    # === Band-specific higher-order stats ===
    for band in ['g', 'r']:
        if band in band_data:
            band_hos = compute_higher_order_stats(band_data[band]['fluxes'])
            for key, val in band_hos.items():
                features[f'{band}_{key}'] = val
        else:
            features[f'{band}_flux_skewness'] = np.nan
            features[f'{band}_flux_kurtosis'] = np.nan
            features[f'{band}_flux_biweight'] = np.nan

    # === Peak time lag features (enhanced) ===
    if 'g' in band_data and 'r' in band_data:
        g_peak_time = band_data['g']['times'][np.argmax(band_data['g']['fluxes'])]
        r_peak_time = band_data['r']['times'][np.argmax(band_data['r']['fluxes'])]
        features['peak_lag_g_r'] = g_peak_time - r_peak_time
    else:
        features['peak_lag_g_r'] = np.nan

    if 'r' in band_data and 'i' in band_data:
        r_peak_time = band_data['r']['times'][np.argmax(band_data['r']['fluxes'])]
        i_peak_time = band_data['i']['times'][np.argmax(band_data['i']['fluxes'])]
        features['peak_lag_r_i'] = r_peak_time - i_peak_time
    else:
        features['peak_lag_r_i'] = np.nan

    # === Flux ratios at peak (spectral shape) ===
    if 'g' in band_data and 'r' in band_data:
        g_peak = np.max(band_data['g']['fluxes'])
        r_peak = np.max(band_data['r']['fluxes'])
        if r_peak > 0:
            features['peak_flux_ratio_g_r'] = g_peak / r_peak
        else:
            features['peak_flux_ratio_g_r'] = np.nan
    else:
        features['peak_flux_ratio_g_r'] = np.nan

    if 'r' in band_data and 'i' in band_data:
        r_peak = np.max(band_data['r']['fluxes'])
        i_peak = np.max(band_data['i']['fluxes'])
        if i_peak > 0:
            features['peak_flux_ratio_r_i'] = r_peak / i_peak
        else:
            features['peak_flux_ratio_r_i'] = np.nan
    else:
        features['peak_flux_ratio_r_i'] = np.nan

    return features


def extract_advanced_features(
    lightcurves: pd.DataFrame,
    metadata: pd.DataFrame,
    object_ids: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract advanced features for multiple objects.

    Args:
        lightcurves: DataFrame with lightcurve data
        metadata: DataFrame with object metadata (including Z for redshift)
        object_ids: Optional list of object IDs to process
        verbose: Print progress

    Returns:
        DataFrame with advanced features for each object
    """
    if object_ids is None:
        object_ids = lightcurves['object_id'].unique()

    # Create redshift lookup
    z_lookup = dict(zip(metadata['object_id'], metadata['Z']))

    # Pre-group lightcurves
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if verbose and (i + 1) % 100 == 0:
            print(f"    Advanced: {i+1}/{len(object_ids)} objects processed", flush=True)

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue

        redshift = z_lookup.get(obj_id, np.nan)

        features = extract_advanced_features_single(obj_lc, redshift)
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

    print("\nExtracting advanced features for first 20 objects...")
    sample_ids = data['train_meta']['object_id'].head(20).tolist()

    adv_features = extract_advanced_features(
        data['train_lc'],
        data['train_meta'],
        sample_ids
    )

    print(f"\nExtracted {len(adv_features.columns)-1} advanced features")
    print("\nFeature columns:")
    print([c for c in adv_features.columns if c != 'object_id'])
    print("\nSample values:")
    print(adv_features[['object_id', 'r_abs_mag_peak', 'r_mhps_30', 'r_fleet_width']].head(10))
