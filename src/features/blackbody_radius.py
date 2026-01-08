"""
Blackbody Radius Evolution Features for TDE vs SN Discrimination

This is a CRITICAL discriminating feature based on fundamental physics:

PHYSICS BASIS (Stefan-Boltzmann Law):
    L = 4πR²σT⁴
    Therefore: R_bb = sqrt(L / (4πσT⁴))

KEY DISCRIMINATOR:
    - SUPERNOVAE: R_bb INCREASES initially (expanding ejecta), then decreases slowly
      "Regardless of density profile, photospheric radius ALWAYS increases early on"
      (Piro & Nakar 2013, ApJ Letters)

    - TDEs: R_bb DECREASES from the very beginning or stays constant
      "Photospheric radius decays from the very beginning - essentially IMPOSSIBLE
       for it to be a supernova" (Perley et al. 2019 on AT 2018cow)

    - AGN: Stochastic R_bb evolution, no systematic trend

FEATURES COMPUTED:
1. R_bb at multiple epochs: peak, +10d, +20d, +30d, +50d, +100d
2. dR_bb/dt at early phase (0-30d) and late phase (30-100d)
3. R_bb ratios: R(peak)/R(50d), R(10d)/R(30d), etc.
4. Direction metrics: is R_bb monotonically decreasing?
5. R_bb variance and trend over time
6. T² evolution (since R ∝ 1/T²)
7. Combined L/T⁴ proxy metrics

References:
- Piro & Nakar 2013: Photospheric radius evolution in explosions
- Perley et al. 2019: AT 2018cow as potential TDE (R_bb behavior)
- van Velzen et al. 2020: TDE photospheric properties
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')

# LSST bands
LSST_BANDS = ["u", "g", "r", "i", "z", "y"]

# Effective wavelengths in Angstroms
BAND_WAVELENGTHS = {
    "u": 3670, "g": 4825, "r": 6222,
    "i": 7545, "z": 8691, "y": 9710
}

# Physical constants (CGS)
H_PLANCK = 6.626e-27    # erg·s
C_LIGHT = 2.998e10      # cm/s
K_BOLTZ = 1.381e-16     # erg/K
SIGMA_SB = 5.670e-5     # erg/(cm²·s·K⁴) Stefan-Boltzmann constant


def planck_function(wavelength_A: float, T: float) -> float:
    """Planck function B_λ(T) in CGS units."""
    if T <= 0 or wavelength_A <= 0:
        return 0.0

    lam_cm = wavelength_A * 1e-8
    try:
        x = (H_PLANCK * C_LIGHT) / (lam_cm * K_BOLTZ * T)
        if x > 700:
            return 0.0
        return (2 * H_PLANCK * C_LIGHT**2 / lam_cm**5) / (np.exp(x) - 1)
    except:
        return 0.0


def fit_blackbody_temperature(fluxes: Dict[str, float],
                               bands: List[str] = ['g', 'r', 'i']) -> Tuple[float, float, float]:
    """
    Fit blackbody temperature to multi-band fluxes.

    Returns:
        (temperature_K, amplitude, chi2_reduced)
    """
    # Get valid bands
    valid_bands = [b for b in bands if b in fluxes and
                   fluxes[b] is not None and
                   np.isfinite(fluxes[b]) and
                   fluxes[b] > 0]

    if len(valid_bands) < 2:
        return np.nan, np.nan, np.nan

    wavelengths = np.array([BAND_WAVELENGTHS[b] for b in valid_bands])
    obs_fluxes = np.array([fluxes[b] for b in valid_bands])

    # Normalize
    flux_norm = np.median(obs_fluxes)
    if flux_norm <= 0:
        return np.nan, np.nan, np.nan
    obs_norm = obs_fluxes / flux_norm

    def model(lam, T, A):
        return A * np.array([planck_function(l, T) for l in lam])

    try:
        # Grid search for initial T
        best_chi2 = np.inf
        best_T = 15000

        for T_init in [8000, 12000, 15000, 20000, 30000, 50000]:
            try:
                popt, _ = curve_fit(model, wavelengths, obs_norm,
                                   p0=[T_init, 1e-10],
                                   bounds=([3000, 1e-20], [100000, 1e0]),
                                   maxfev=1000)
                pred = model(wavelengths, *popt)
                chi2 = np.sum((obs_norm - pred)**2)
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_T = popt[0]
                    best_A = popt[1]
            except:
                continue

        if best_chi2 == np.inf:
            return np.nan, np.nan, np.nan

        chi2_red = best_chi2 / max(1, len(valid_bands) - 2)
        return best_T, best_A, chi2_red

    except:
        return np.nan, np.nan, np.nan


def estimate_bolometric_flux(fluxes: Dict[str, float], T: float,
                              bands: List[str] = ['g', 'r', 'i', 'z']) -> float:
    """
    Estimate pseudo-bolometric flux by integrating blackbody with measured normalization.

    L_bol ∝ A * T⁴ where A is the amplitude from BB fit
    """
    if np.isnan(T) or T <= 0:
        return np.nan

    valid_bands = [b for b in bands if b in fluxes and
                   fluxes[b] is not None and
                   np.isfinite(fluxes[b]) and
                   fluxes[b] > 0]

    if len(valid_bands) < 2:
        return np.nan

    # Use mean flux as proxy for amplitude, scale by T^4 to get L
    mean_flux = np.mean([fluxes[b] for b in valid_bands])

    # L_bol ∝ F_obs * (some distance factor we don't know)
    # But R ∝ sqrt(L) / T², so R ∝ sqrt(F) / T²
    # We return F as proxy for L (will be normalized anyway)
    return mean_flux


def compute_blackbody_radius(L_proxy: float, T: float) -> float:
    """
    Compute blackbody radius (in arbitrary units, normalized later).

    R_bb = sqrt(L / (4πσT⁴))
    R_bb ∝ sqrt(L) / T²

    We use L_proxy (flux) as proportional to L.
    """
    if np.isnan(L_proxy) or np.isnan(T) or L_proxy <= 0 or T <= 0:
        return np.nan

    # R ∝ sqrt(L) / T²
    R = np.sqrt(L_proxy) / (T**2)

    # Scale to reasonable numbers (avoid tiny values)
    return R * 1e8


def get_fluxes_at_epoch(obj_lc: pd.DataFrame, peak_time: float,
                        epoch_offset: float, window: float = 5.0) -> Dict[str, float]:
    """
    Get flux in each band at a specific epoch (relative to peak).

    Args:
        obj_lc: Lightcurve DataFrame
        peak_time: Time of peak (MJD)
        epoch_offset: Days after peak (0 = at peak, 50 = 50 days after)
        window: Time window to average observations

    Returns:
        Dict of band -> flux
    """
    target_time = peak_time + epoch_offset
    fluxes = {}

    for band in LSST_BANDS:
        band_lc = obj_lc[obj_lc['Filter'] == band]
        if len(band_lc) == 0:
            fluxes[band] = np.nan
            continue

        times = band_lc['Time (MJD)'].values
        flux_vals = band_lc['Flux'].values

        # Find observations within window
        mask = np.abs(times - target_time) <= window

        if np.sum(mask) == 0:
            # Try interpolation if we have data on both sides
            before = times < target_time
            after = times > target_time

            if np.any(before) and np.any(after):
                # Linear interpolation
                t_before = times[before][-1]
                t_after = times[after][0]
                f_before = flux_vals[before][-1]
                f_after = flux_vals[after][0]

                if t_after - t_before < 30:  # Max 30 day gap
                    weight = (target_time - t_before) / (t_after - t_before)
                    fluxes[band] = f_before + weight * (f_after - f_before)
                else:
                    fluxes[band] = np.nan
            else:
                fluxes[band] = np.nan
        else:
            # Average observations in window
            fluxes[band] = np.mean(flux_vals[mask])

    return fluxes


def find_global_peak(obj_lc: pd.DataFrame, bands: List[str] = ['g', 'r', 'i']) -> float:
    """Find time of global peak across specified bands."""
    peak_times = []
    peak_fluxes = []

    for band in bands:
        band_lc = obj_lc[obj_lc['Filter'] == band]
        if len(band_lc) > 0:
            max_idx = band_lc['Flux'].idxmax()
            peak_times.append(band_lc.loc[max_idx, 'Time (MJD)'])
            peak_fluxes.append(band_lc.loc[max_idx, 'Flux'])

    if len(peak_times) == 0:
        return np.nan

    # Use flux-weighted average peak time
    weights = np.array(peak_fluxes) / np.sum(peak_fluxes)
    return np.average(peak_times, weights=weights)


def extract_radius_features_single(obj_lc: pd.DataFrame) -> Dict[str, float]:
    """
    Extract comprehensive blackbody radius evolution features for a single object.

    This is the CORE function that computes all R_bb-related features.
    """
    features = {}

    # Find peak time
    peak_time = find_global_peak(obj_lc)
    if np.isnan(peak_time):
        # Return NaN features
        for key in ['R_bb_peak', 'R_bb_10d', 'R_bb_20d', 'R_bb_30d', 'R_bb_50d', 'R_bb_100d',
                    'T_peak', 'T_30d', 'T_50d', 'T_100d',
                    'dRdt_early', 'dRdt_late', 'dRdt_overall',
                    'R_ratio_peak_50d', 'R_ratio_peak_100d', 'R_ratio_10d_30d',
                    'R_increasing_early', 'R_monotonic_decrease',
                    'T_variance', 'T_drop_peak_50d', 'T_drop_peak_100d',
                    'R_bb_variance', 'R_bb_range', 'R_bb_trend_slope']:
            features[key] = np.nan
        return features

    # Epochs to compute R_bb (days relative to peak)
    epochs = [0, 10, 20, 30, 50, 100]
    epoch_names = ['peak', '10d', '20d', '30d', '50d', '100d']

    R_values = []
    T_values = []
    L_values = []
    valid_epochs = []

    for epoch, name in zip(epochs, epoch_names):
        # Get fluxes at this epoch
        fluxes = get_fluxes_at_epoch(obj_lc, peak_time, epoch)

        # Fit temperature
        T, A, chi2 = fit_blackbody_temperature(fluxes)
        features[f'T_{name}'] = T
        features[f'T_chi2_{name}'] = chi2

        # Estimate bolometric flux
        L = estimate_bolometric_flux(fluxes, T)

        # Compute R_bb
        R = compute_blackbody_radius(L, T)
        features[f'R_bb_{name}'] = R
        features[f'L_proxy_{name}'] = L

        if not np.isnan(R) and not np.isnan(T):
            R_values.append(R)
            T_values.append(T)
            L_values.append(L)
            valid_epochs.append(epoch)

    # ========================================
    # DERIVED FEATURES - The Key Discriminators
    # ========================================

    if len(valid_epochs) >= 2:
        R_arr = np.array(R_values)
        T_arr = np.array(T_values)
        epoch_arr = np.array(valid_epochs)

        # --- Rate of change dR/dt ---
        # Early phase (0-30 days) - CRITICAL for SN vs TDE
        early_mask = epoch_arr <= 30
        if np.sum(early_mask) >= 2:
            early_epochs = epoch_arr[early_mask]
            early_R = R_arr[early_mask]
            slope, _ = np.polyfit(early_epochs, early_R, 1)
            features['dRdt_early'] = slope

            # Is R increasing early? (SN signature)
            features['R_increasing_early'] = float(slope > 0)
        else:
            features['dRdt_early'] = np.nan
            features['R_increasing_early'] = np.nan

        # Late phase (30-100 days)
        late_mask = epoch_arr >= 30
        if np.sum(late_mask) >= 2:
            late_epochs = epoch_arr[late_mask]
            late_R = R_arr[late_mask]
            slope, _ = np.polyfit(late_epochs, late_R, 1)
            features['dRdt_late'] = slope
        else:
            features['dRdt_late'] = np.nan

        # Overall trend
        slope_overall, _ = np.polyfit(epoch_arr, R_arr, 1)
        features['dRdt_overall'] = slope_overall
        features['R_bb_trend_slope'] = slope_overall

        # --- Monotonic decrease check (TDE signature) ---
        diffs = np.diff(R_arr)
        features['R_monotonic_decrease'] = float(np.all(diffs < 0))
        features['R_frac_decreasing'] = np.sum(diffs < 0) / len(diffs)

        # --- R ratios ---
        R_peak = features.get('R_bb_peak', np.nan)
        R_10d = features.get('R_bb_10d', np.nan)
        R_30d = features.get('R_bb_30d', np.nan)
        R_50d = features.get('R_bb_50d', np.nan)
        R_100d = features.get('R_bb_100d', np.nan)

        if not np.isnan(R_peak) and R_50d and not np.isnan(R_50d) and R_50d > 0:
            features['R_ratio_peak_50d'] = R_peak / R_50d
        else:
            features['R_ratio_peak_50d'] = np.nan

        if not np.isnan(R_peak) and R_100d and not np.isnan(R_100d) and R_100d > 0:
            features['R_ratio_peak_100d'] = R_peak / R_100d
        else:
            features['R_ratio_peak_100d'] = np.nan

        if R_10d and not np.isnan(R_10d) and R_30d and not np.isnan(R_30d) and R_30d > 0:
            features['R_ratio_10d_30d'] = R_10d / R_30d
        else:
            features['R_ratio_10d_30d'] = np.nan

        # --- R statistics ---
        features['R_bb_variance'] = np.var(R_arr)
        features['R_bb_range'] = np.max(R_arr) - np.min(R_arr)
        features['R_bb_mean'] = np.mean(R_arr)
        features['R_bb_std'] = np.std(R_arr)

        # Relative change
        features['R_bb_rel_change'] = (R_arr[-1] - R_arr[0]) / (R_arr[0] + 1e-10)

        # --- Temperature evolution ---
        features['T_variance'] = np.var(T_arr)
        features['T_std'] = np.std(T_arr)
        features['T_range'] = np.max(T_arr) - np.min(T_arr)

        T_peak = features.get('T_peak', np.nan)
        T_50d = features.get('T_50d', np.nan)
        T_100d = features.get('T_100d', np.nan)

        if not np.isnan(T_peak) and not np.isnan(T_50d):
            features['T_drop_peak_50d'] = T_peak - T_50d
            features['T_ratio_peak_50d'] = T_peak / (T_50d + 1)
        else:
            features['T_drop_peak_50d'] = np.nan
            features['T_ratio_peak_50d'] = np.nan

        if not np.isnan(T_peak) and not np.isnan(T_100d):
            features['T_drop_peak_100d'] = T_peak - T_100d
            features['T_ratio_peak_100d'] = T_peak / (T_100d + 1)
        else:
            features['T_drop_peak_100d'] = np.nan
            features['T_ratio_peak_100d'] = np.nan

        # Temperature trend
        T_slope, _ = np.polyfit(epoch_arr, T_arr, 1)
        features['dTdt'] = T_slope

        # --- Combined metrics ---
        # TDE score: low T variance + decreasing R + high T
        T_var_norm = features['T_variance'] / (np.mean(T_arr)**2 + 1)
        features['T_constancy'] = 1.0 / (T_var_norm + 0.01)  # High = constant T = TDE

        # R evolution direction score
        # Positive = increasing = SN-like, Negative = decreasing = TDE-like
        features['R_direction_score'] = features['dRdt_overall'] / (np.mean(R_arr) + 1e-10)

    else:
        # Not enough epochs
        for key in ['dRdt_early', 'dRdt_late', 'dRdt_overall',
                    'R_increasing_early', 'R_monotonic_decrease', 'R_frac_decreasing',
                    'R_ratio_peak_50d', 'R_ratio_peak_100d', 'R_ratio_10d_30d',
                    'R_bb_variance', 'R_bb_range', 'R_bb_mean', 'R_bb_std', 'R_bb_rel_change',
                    'R_bb_trend_slope', 'T_variance', 'T_std', 'T_range',
                    'T_drop_peak_50d', 'T_drop_peak_100d', 'T_ratio_peak_50d', 'T_ratio_peak_100d',
                    'dTdt', 'T_constancy', 'R_direction_score']:
            features[key] = np.nan

    return features


def extract_radius_features(lightcurves: pd.DataFrame,
                            object_ids: List[str]) -> pd.DataFrame:
    """
    Extract blackbody radius features for multiple objects.

    Args:
        lightcurves: DataFrame with columns [object_id, Time (MJD), Flux, Flux_err, Filter]
        object_ids: List of object IDs to process

    Returns:
        DataFrame with radius features for each object
    """
    # Pre-group for efficiency
    grouped = {oid: group for oid, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, oid in enumerate(object_ids):
        if (i + 1) % 500 == 0:
            print(f"    R_bb features: {i+1}/{len(object_ids)} objects", flush=True)

        obj_lc = grouped.get(oid, pd.DataFrame())

        if obj_lc.empty:
            features = {'object_id': oid}
            all_features.append(features)
            continue

        features = extract_radius_features_single(obj_lc)
        features['object_id'] = oid
        all_features.append(features)

    return pd.DataFrame(all_features)


if __name__ == "__main__":
    # Test the module
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_all_data

    print("Loading data...")
    data = load_all_data()

    print("\nTesting R_bb features on first 20 objects...")
    sample_ids = data['train_meta']['object_id'].head(20).tolist()

    rbb_features = extract_radius_features(data['train_lc'], sample_ids)

    print(f"\nExtracted {len(rbb_features.columns)-1} R_bb features")
    print("\nFeature columns:")
    for col in rbb_features.columns:
        if col != 'object_id':
            print(f"  {col}")

    print("\nSample values:")
    print(rbb_features[['object_id', 'R_bb_peak', 'dRdt_early', 'R_increasing_early', 'T_peak']].head(10))
