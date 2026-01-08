"""
Physics-based feature extraction for MALLORN classification.

Features derived from astrophysical literature and previous competitions:
1. Stetson variability indices (J, K) - correlated multi-band variability
2. Structure function - variability vs timescale
3. Rest-frame corrections - account for redshift time dilation
4. Temperature estimation - blackbody temperature from colors
5. Bazin function parameters - parametric lightcurve fit

References:
- Stetson (1996): Variability indices
- PLAsTiCC competition: GP and Bazin features
- TDE classification papers: Post-peak colors, timescales
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
import warnings

LSST_BANDS = ["u", "g", "r", "i", "z", "y"]

# Effective wavelengths in Angstroms for temperature estimation
BAND_WAVELENGTHS_A = {
    "u": 3670, "g": 4825, "r": 6222,
    "i": 7545, "z": 8691, "y": 9710
}


def compute_stetson_j(times1: np.ndarray, fluxes1: np.ndarray, errors1: np.ndarray,
                      times2: np.ndarray, fluxes2: np.ndarray, errors2: np.ndarray,
                      max_dt: float = 0.5) -> float:
    """
    Compute Stetson J index for correlated variability between two bands.

    J measures whether two bands vary together (positive J) or independently.
    TDEs and SNe show correlated multi-band variability.
    AGN show more stochastic behavior.

    Args:
        times1, fluxes1, errors1: First band data
        times2, fluxes2, errors2: Second band data
        max_dt: Maximum time difference for "simultaneous" observations (days)

    Returns:
        Stetson J index
    """
    if len(times1) < 3 or len(times2) < 3:
        return np.nan

    # Normalize fluxes
    mean1, std1 = np.mean(fluxes1), np.std(fluxes1)
    mean2, std2 = np.mean(fluxes2), np.std(fluxes2)

    if std1 == 0 or std2 == 0:
        return 0.0

    # Find quasi-simultaneous observations
    j_sum = 0.0
    n_pairs = 0

    for i, (t1, f1, e1) in enumerate(zip(times1, fluxes1, errors1)):
        # Find closest observation in band 2
        dt = np.abs(times2 - t1)
        min_idx = np.argmin(dt)

        if dt[min_idx] <= max_dt:
            f2 = fluxes2[min_idx]
            e2 = errors2[min_idx]

            # Weighted residuals
            if e1 > 0 and e2 > 0:
                delta1 = (f1 - mean1) / e1
                delta2 = (f2 - mean2) / e2
                j_sum += np.sign(delta1 * delta2) * np.sqrt(np.abs(delta1 * delta2))
                n_pairs += 1

    if n_pairs == 0:
        return np.nan

    return j_sum / n_pairs


def compute_stetson_k(fluxes: np.ndarray, errors: np.ndarray) -> float:
    """
    Compute Stetson K index (kurtosis-like measure of variability).

    K is sensitive to the shape of the flux distribution.
    K ~ 0.798 for Gaussian noise, higher for true variability.
    """
    if len(fluxes) < 4:
        return np.nan

    mean_flux = np.mean(fluxes)
    n = len(fluxes)

    # Weighted residuals
    valid = errors > 0
    if np.sum(valid) < 4:
        return np.nan

    delta = np.abs(fluxes[valid] - mean_flux) / errors[valid]

    k = np.sum(delta) / np.sqrt(np.sum(delta**2)) / np.sqrt(n)

    return k


def compute_structure_function(times: np.ndarray, fluxes: np.ndarray,
                               tau_bins: List[float] = [1, 5, 10, 30, 100]) -> Dict[str, float]:
    """
    Compute structure function: variance of flux differences vs time lag.

    SF(τ) = <(f(t+τ) - f(t))²>

    Different transient types have characteristic SF shapes.
    Optimized with vectorized operations for speed.

    Args:
        times: Observation times
        fluxes: Flux values
        tau_bins: Time lag bins in days

    Returns:
        Dictionary with SF values at each tau bin
    """
    if len(times) < 5:
        return {f'sf_tau_{tau}': np.nan for tau in tau_bins}

    features = {}

    # Vectorized computation of all pairwise differences
    n = len(times)
    # Use upper triangle indices for efficiency
    i_idx, j_idx = np.triu_indices(n, k=1)
    dt_all = np.abs(times[j_idx] - times[i_idx])
    df_all = (fluxes[j_idx] - fluxes[i_idx])**2

    for tau in tau_bins:
        tau_min = tau * 0.5
        tau_max = tau * 1.5

        # Boolean mask for this tau bin
        mask = (dt_all >= tau_min) & (dt_all <= tau_max)
        sf_values = df_all[mask]

        if len(sf_values) >= 3:
            features[f'sf_tau_{tau}'] = np.sqrt(np.mean(sf_values))
        else:
            features[f'sf_tau_{tau}'] = np.nan

    # SF slope (log-log)
    valid_taus = []
    valid_sfs = []
    for tau in tau_bins:
        sf = features[f'sf_tau_{tau}']
        if not np.isnan(sf) and sf > 0:
            valid_taus.append(np.log10(tau))
            valid_sfs.append(np.log10(sf))

    if len(valid_taus) >= 3:
        coeffs = np.polyfit(valid_taus, valid_sfs, 1)
        features['sf_slope'] = coeffs[0]
    else:
        features['sf_slope'] = np.nan

    return features


def estimate_temperature(g_flux: float, r_flux: float, i_flux: float) -> float:
    """
    Estimate blackbody temperature from optical colors.

    Using Wien's displacement law approximation.
    TDEs: T ~ 20,000-40,000 K (hot)
    SNe at peak: T ~ 10,000-15,000 K, cooling over time
    """
    if any(f <= 0 or np.isnan(f) for f in [g_flux, r_flux, i_flux]):
        return np.nan

    # Use g-r color as temperature proxy
    # Bluer (higher g/r) = hotter
    # Rough calibration: T ~ 7000K / (g-r + 0.5)

    try:
        g_r_color = -2.5 * np.log10(g_flux / r_flux)

        # Simple temperature estimate (approximate)
        if g_r_color < -0.5:
            temp = 50000  # Very hot
        elif g_r_color > 2.0:
            temp = 3000   # Cool
        else:
            temp = 7000 / (g_r_color + 0.6)

        return np.clip(temp, 3000, 100000)
    except:
        return np.nan


def fit_bazin_simple(times: np.ndarray, fluxes: np.ndarray) -> Dict[str, float]:
    """
    Extract Bazin-like parameters without actual fitting.

    The Bazin function: f(t) = A * exp(-(t-t0)/τ_fall) / (1 + exp(-(t-t0)/τ_rise))

    We approximate key parameters from the data directly.
    """
    if len(times) < 5:
        return {
            'bazin_amplitude': np.nan,
            'bazin_t0': np.nan,
            'bazin_rise_approx': np.nan,
            'bazin_fall_approx': np.nan,
            'bazin_plateau': np.nan
        }

    features = {}

    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    fluxes = fluxes[sort_idx]

    # Peak
    peak_idx = np.argmax(fluxes)
    peak_time = times[peak_idx]
    peak_flux = fluxes[peak_idx]

    features['bazin_amplitude'] = peak_flux
    features['bazin_t0'] = peak_time

    # Approximate rise time (time for flux to go from 10% to 90% of peak)
    pre_peak = fluxes[:peak_idx+1]
    if len(pre_peak) >= 2:
        thresh_10 = 0.1 * peak_flux
        thresh_90 = 0.9 * peak_flux

        t_10 = times[0]  # default
        t_90 = peak_time

        for i, (t, f) in enumerate(zip(times[:peak_idx+1], pre_peak)):
            if f >= thresh_10 and t_10 == times[0]:
                t_10 = t
            if f >= thresh_90:
                t_90 = t
                break

        features['bazin_rise_approx'] = t_90 - t_10
    else:
        features['bazin_rise_approx'] = np.nan

    # Approximate fall time (e-folding time after peak)
    post_peak_times = times[peak_idx:]
    post_peak_fluxes = fluxes[peak_idx:]

    if len(post_peak_times) >= 3:
        # Time to fall to 1/e of peak
        target = peak_flux / np.e

        fall_time = np.nan
        for t, f in zip(post_peak_times, post_peak_fluxes):
            if f <= target:
                fall_time = t - peak_time
                break

        if np.isnan(fall_time) and len(post_peak_times) > 1:
            # Extrapolate
            fall_time = (post_peak_times[-1] - peak_time) * peak_flux / (peak_flux - post_peak_fluxes[-1] + 1e-6)

        features['bazin_fall_approx'] = fall_time
    else:
        features['bazin_fall_approx'] = np.nan

    # Plateau detection (is there a flat period?)
    if len(post_peak_fluxes) >= 5:
        mid_idx = len(post_peak_fluxes) // 2
        early_flux = np.mean(post_peak_fluxes[:mid_idx])
        late_flux = np.mean(post_peak_fluxes[mid_idx:])

        if early_flux > 0:
            features['bazin_plateau'] = late_flux / early_flux
        else:
            features['bazin_plateau'] = np.nan
    else:
        features['bazin_plateau'] = np.nan

    return features


def extract_physics_features_single(obj_lc: pd.DataFrame, redshift: float) -> Dict[str, float]:
    """
    Extract physics-based features for a single object.

    Args:
        obj_lc: DataFrame with lightcurve data
        redshift: Object redshift for rest-frame corrections

    Returns:
        Dictionary of physics-based features
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

    # === Stetson Indices ===
    # J index for band pairs
    for b1, b2 in [('g', 'r'), ('r', 'i'), ('g', 'i')]:
        if b1 in band_data and b2 in band_data:
            j = compute_stetson_j(
                band_data[b1]['times'], band_data[b1]['fluxes'], band_data[b1]['errors'],
                band_data[b2]['times'], band_data[b2]['fluxes'], band_data[b2]['errors']
            )
            features[f'stetson_j_{b1}{b2}'] = j
        else:
            features[f'stetson_j_{b1}{b2}'] = np.nan

    # K index per band
    for band in ['g', 'r', 'i']:
        if band in band_data:
            k = compute_stetson_k(band_data[band]['fluxes'], band_data[band]['errors'])
            features[f'stetson_k_{band}'] = k
        else:
            features[f'stetson_k_{band}'] = np.nan

    # === Structure Function ===
    # For r-band (usually best sampled)
    if 'r' in band_data:
        sf = compute_structure_function(band_data['r']['times'], band_data['r']['fluxes'])
        for key, val in sf.items():
            features[f'r_{key}'] = val
    else:
        for tau in [1, 5, 10, 30, 100]:
            features[f'r_sf_tau_{tau}'] = np.nan
        features['r_sf_slope'] = np.nan

    # === Rest-frame corrections ===
    z = redshift if not np.isnan(redshift) else 0

    # Rest-frame timescales
    for band in ['g', 'r', 'i']:
        if band in band_data:
            times = band_data[band]['times']
            fluxes = band_data[band]['fluxes']

            # Observed timescale
            obs_duration = times[-1] - times[0]
            # Rest-frame duration
            rest_duration = obs_duration / (1 + z)
            features[f'{band}_rest_duration'] = rest_duration

            # Rest-frame rise time (if we have peak)
            peak_idx = np.argmax(fluxes)
            if peak_idx > 0:
                obs_rise = times[peak_idx] - times[0]
                features[f'{band}_rest_rise'] = obs_rise / (1 + z)
            else:
                features[f'{band}_rest_rise'] = np.nan

            # Rest-frame fade time
            if peak_idx < len(times) - 1:
                obs_fade = times[-1] - times[peak_idx]
                features[f'{band}_rest_fade'] = obs_fade / (1 + z)
            else:
                features[f'{band}_rest_fade'] = np.nan
        else:
            features[f'{band}_rest_duration'] = np.nan
            features[f'{band}_rest_rise'] = np.nan
            features[f'{band}_rest_fade'] = np.nan

    # === Temperature estimation ===
    # Get fluxes near peak
    if 'g' in band_data and 'r' in band_data and 'i' in band_data:
        # Use peak fluxes
        g_peak = np.max(band_data['g']['fluxes'])
        r_peak = np.max(band_data['r']['fluxes'])
        i_peak = np.max(band_data['i']['fluxes'])

        features['temp_at_peak'] = estimate_temperature(g_peak, r_peak, i_peak)

        # Temperature 50 days after peak (if available)
        r_times = band_data['r']['times']
        r_peak_time = r_times[np.argmax(band_data['r']['fluxes'])]

        # Find fluxes around t_peak + 50
        target_time = r_peak_time + 50
        g_late = np.nan
        r_late = np.nan
        i_late = np.nan

        for band, var in [('g', 'g_late'), ('r', 'r_late'), ('i', 'i_late')]:
            if band in band_data:
                dt = np.abs(band_data[band]['times'] - target_time)
                min_idx = np.argmin(dt)
                if dt[min_idx] < 20:
                    if band == 'g':
                        g_late = band_data[band]['fluxes'][min_idx]
                    elif band == 'r':
                        r_late = band_data[band]['fluxes'][min_idx]
                    else:
                        i_late = band_data[band]['fluxes'][min_idx]

        features['temp_post_50d'] = estimate_temperature(g_late, r_late, i_late)

        # Temperature evolution
        if not np.isnan(features['temp_at_peak']) and not np.isnan(features['temp_post_50d']):
            features['temp_evolution'] = (features['temp_post_50d'] - features['temp_at_peak']) / 50.0
        else:
            features['temp_evolution'] = np.nan
    else:
        features['temp_at_peak'] = np.nan
        features['temp_post_50d'] = np.nan
        features['temp_evolution'] = np.nan

    # === Bazin-like parameters ===
    # From r-band
    if 'r' in band_data:
        bazin = fit_bazin_simple(band_data['r']['times'], band_data['r']['fluxes'])
        for key, val in bazin.items():
            features[f'r_{key}'] = val
    else:
        for key in ['bazin_amplitude', 'bazin_t0', 'bazin_rise_approx', 'bazin_fall_approx', 'bazin_plateau']:
            features[f'r_{key}'] = np.nan

    # === Additional variability features ===
    # Weighted mean magnitude error
    all_fluxes = obj_lc['Flux'].values
    all_errors = obj_lc['Flux_err'].values

    valid = (all_errors > 0) & (all_fluxes > 0)
    if np.sum(valid) > 0:
        snr = all_fluxes[valid] / all_errors[valid]
        features['mean_snr'] = np.mean(snr)
        features['median_snr'] = np.median(snr)

        # Excess variance (intrinsic variability beyond noise)
        mean_flux = np.mean(all_fluxes[valid])
        var_flux = np.var(all_fluxes[valid])
        mean_var_noise = np.mean(all_errors[valid]**2)

        excess_var = (var_flux - mean_var_noise) / mean_flux**2
        features['excess_variance'] = max(0, excess_var)
    else:
        features['mean_snr'] = np.nan
        features['median_snr'] = np.nan
        features['excess_variance'] = np.nan

    return features


def extract_physics_features(
    lightcurves: pd.DataFrame,
    metadata: pd.DataFrame,
    object_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract physics-based features for multiple objects.

    Args:
        lightcurves: DataFrame with lightcurve data
        metadata: DataFrame with object_id and Z (redshift)
        object_ids: Optional list of object IDs

    Returns:
        DataFrame with physics-based features
    """
    if object_ids is None:
        object_ids = lightcurves['object_id'].unique()

    # Create redshift lookup
    z_lookup = dict(zip(metadata['object_id'], metadata['Z']))

    # Pre-group lightcurves by object_id for O(1) lookup (major speedup!)
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if (i + 1) % 200 == 0:
            print(f"    Physics: {i+1}/{len(object_ids)} objects processed")

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue

        redshift = z_lookup.get(obj_id, np.nan)

        features = extract_physics_features_single(obj_lc, redshift)
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

    print("\nExtracting physics features for first 20 objects...")
    sample_ids = data['train_meta']['object_id'].head(20).tolist()
    physics_features = extract_physics_features(
        data['train_lc'],
        data['train_meta'],
        sample_ids
    )

    print(f"\nExtracted {len(physics_features.columns)-1} physics features")
    print("\nFeature columns:")
    print([c for c in physics_features.columns if c != 'object_id'])
    print("\nSample values:")
    print(physics_features[['object_id', 'stetson_j_gr', 'temp_at_peak', 'r_sf_slope']].head())
