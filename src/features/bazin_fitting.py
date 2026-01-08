"""
Bazin Function Parametric Lightcurve Fitting for MALLORN.

The Bazin function is a parametric model for transient lightcurves:
    f(t) = A * exp(-(t-t0)/τ_fall) / (1 + exp(-(t-t0)/τ_rise)) + B

Used by PLAsTiCC 1st and 2nd place teams.

Parameters:
- A: Amplitude (peak brightness above baseline)
- t0: Time of peak
- τ_rise: Rise timescale
- τ_fall: Fall/decay timescale
- B: Baseline flux

Why it works:
1. More robust than raw statistics on sparse lightcurves
2. Parameters have physical meaning (rise/fall times)
3. Captures overall shape with just 5 numbers
4. Smooths over observational noise

References:
- Bazin et al. (2009): "SN Ia light curve fitting"
- PLAsTiCC 1st place (kozodoi): Used Bazin for all bands
- PLAsTiCC 2nd place: Bazin + GP features
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, OptimizeWarning
from typing import List, Dict, Optional
import warnings

LSST_BANDS = ["u", "g", "r", "i", "z", "y"]


def bazin_function(t, A, t0, tau_rise, tau_fall, B):
    """
    Bazin function for transient lightcurves.

    f(t) = A * exp(-(t-t0)/τ_fall) / (1 + exp(-(t-t0)/τ_rise)) + B

    Args:
        t: Time array
        A: Amplitude (peak brightness above baseline)
        t0: Time of peak
        tau_rise: Rise timescale (days)
        tau_fall: Fall timescale (days)
        B: Baseline flux

    Returns:
        Flux array
    """
    # Numerator: exponential decay
    numerator = np.exp(-(t - t0) / tau_fall)

    # Denominator: sigmoid rise
    denominator = 1.0 + np.exp(-(t - t0) / tau_rise)

    return A * numerator / denominator + B


def fit_bazin_single_band(times: np.ndarray, fluxes: np.ndarray,
                          flux_errors: np.ndarray) -> Dict[str, float]:
    """
    Fit Bazin function to a single band lightcurve.

    Args:
        times: Observation times (MJD)
        fluxes: Flux measurements
        flux_errors: Flux uncertainties

    Returns:
        Dictionary with fitted parameters and derived features
    """
    if len(times) < 5:
        # Need at least 5 points to fit 5 parameters
        return {
            'bazin_A': np.nan,
            'bazin_t0': np.nan,
            'bazin_tau_rise': np.nan,
            'bazin_tau_fall': np.nan,
            'bazin_B': np.nan,
            'bazin_fit_chi2': np.nan,
            'bazin_rise_fall_ratio': np.nan,
            'bazin_peak_flux': np.nan
        }

    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    fluxes = fluxes[sort_idx]
    flux_errors = flux_errors[sort_idx]

    # Initial parameter guesses
    # Find rough peak
    peak_idx = np.argmax(fluxes)
    t0_guess = times[peak_idx]
    A_guess = fluxes[peak_idx] - np.median(fluxes)
    B_guess = np.median(fluxes)

    # Rise and fall time guesses
    duration = times[-1] - times[0]
    tau_rise_guess = duration * 0.2  # ~20% of duration
    tau_fall_guess = duration * 0.3  # ~30% of duration

    # Parameter bounds
    # A: [0, 3× max flux]
    # t0: [first time, last time]
    # tau_rise: [0.1, duration]
    # tau_fall: [0.1, duration]
    # B: [-max flux, 2× max flux]

    max_flux = np.max(fluxes)
    bounds = (
        [0, times[0], 0.1, 0.1, -max_flux],  # Lower bounds
        [3*max_flux, times[-1], duration, duration, 2*max_flux]  # Upper bounds
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", RuntimeWarning)

            # Fit with uncertainties as weights
            sigma = np.where(flux_errors > 0, flux_errors, 1.0)

            popt, pcov = curve_fit(
                bazin_function,
                times,
                fluxes,
                p0=[A_guess, t0_guess, tau_rise_guess, tau_fall_guess, B_guess],
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=True,
                maxfev=2000
            )

            A, t0, tau_rise, tau_fall, B = popt

            # Clip extreme values to prevent inf
            A = np.clip(A, -1e6, 1e6)
            tau_rise = np.clip(tau_rise, 0.1, 1e4)
            tau_fall = np.clip(tau_fall, 0.1, 1e4)
            B = np.clip(B, -1e6, 1e6)

            # Compute fit quality (reduced chi-squared)
            fitted_fluxes = bazin_function(times, A, t0, tau_rise, tau_fall, B)
            residuals = fluxes - fitted_fluxes
            chi2 = np.sum((residuals / sigma)**2)
            reduced_chi2 = np.clip(chi2 / (len(times) - 5), 0, 1e6)  # Clip chi2

            # Derived features
            rise_fall_ratio = np.clip(tau_rise / (tau_fall + 1e-6), 0, 100)
            peak_flux = np.clip(A + B, -1e6, 1e6)

            return {
                'bazin_A': A,
                'bazin_t0': t0,
                'bazin_tau_rise': tau_rise,
                'bazin_tau_fall': tau_fall,
                'bazin_B': B,
                'bazin_fit_chi2': reduced_chi2,
                'bazin_rise_fall_ratio': rise_fall_ratio,
                'bazin_peak_flux': peak_flux
            }

    except (RuntimeError, ValueError, OptimizeWarning):
        # Fitting failed - return NaN
        return {
            'bazin_A': np.nan,
            'bazin_t0': np.nan,
            'bazin_tau_rise': np.nan,
            'bazin_tau_fall': np.nan,
            'bazin_B': np.nan,
            'bazin_fit_chi2': np.nan,
            'bazin_rise_fall_ratio': np.nan,
            'bazin_peak_flux': np.nan
        }


def extract_bazin_features_single(obj_lc: pd.DataFrame) -> Dict[str, float]:
    """
    Extract Bazin features for a single object across all bands.

    Args:
        obj_lc: Lightcurve DataFrame for one object

    Returns:
        Dictionary with Bazin features for all bands
    """
    features = {}

    for band in LSST_BANDS:
        band_lc = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_lc) < 5:
            # Insufficient data for fitting
            for param in ['bazin_A', 'bazin_t0', 'bazin_tau_rise', 'bazin_tau_fall',
                         'bazin_B', 'bazin_fit_chi2', 'bazin_rise_fall_ratio', 'bazin_peak_flux']:
                features[f'{band}_{param}'] = np.nan
            continue

        times = band_lc['Time (MJD)'].values
        fluxes = band_lc['Flux'].values
        flux_errors = band_lc['Flux_err'].values

        # Fit Bazin function
        band_features = fit_bazin_single_band(times, fluxes, flux_errors)

        # Add band prefix
        for key, val in band_features.items():
            features[f'{band}_{key}'] = val

    # Cross-band features
    # Rise time consistency across bands
    rise_times = []
    fall_times = []
    for band in ['g', 'r', 'i']:  # Focus on well-sampled bands
        rise_key = f'{band}_bazin_tau_rise'
        fall_key = f'{band}_bazin_tau_fall'
        if rise_key in features and not np.isnan(features[rise_key]):
            rise_times.append(features[rise_key])
        if fall_key in features and not np.isnan(features[fall_key]):
            fall_times.append(features[fall_key])

    if len(rise_times) >= 2:
        features['bazin_rise_consistency'] = np.std(rise_times) / np.mean(rise_times)
    else:
        features['bazin_rise_consistency'] = np.nan

    if len(fall_times) >= 2:
        features['bazin_fall_consistency'] = np.std(fall_times) / np.mean(fall_times)
    else:
        features['bazin_fall_consistency'] = np.nan

    # Average fit quality
    chi2_values = []
    for band in LSST_BANDS:
        chi2_key = f'{band}_bazin_fit_chi2'
        if chi2_key in features and not np.isnan(features[chi2_key]):
            chi2_values.append(features[chi2_key])

    if len(chi2_values) > 0:
        features['bazin_avg_fit_chi2'] = np.mean(chi2_values)
        features['bazin_fit_quality_dispersion'] = np.std(chi2_values)
    else:
        features['bazin_avg_fit_chi2'] = np.nan
        features['bazin_fit_quality_dispersion'] = np.nan

    return features


def extract_bazin_features(
    lightcurves: pd.DataFrame,
    object_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract Bazin features for multiple objects.

    Args:
        lightcurves: DataFrame with lightcurve data
        object_ids: Optional list of object IDs

    Returns:
        DataFrame with Bazin features
    """
    if object_ids is None:
        object_ids = lightcurves['object_id'].unique()

    # Pre-group for efficiency
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if (i + 1) % 500 == 0:
            print(f"    Bazin: {i+1}/{len(object_ids)} objects processed")

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue

        features = extract_bazin_features_single(obj_lc)
        features['object_id'] = obj_id
        all_features.append(features)

    return pd.DataFrame(all_features)


if __name__ == "__main__":
    # Test Bazin fitting
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_all_data

    print("Loading data...")
    data = load_all_data()

    print("\nTesting Bazin fitting on first 10 objects...")
    sample_ids = data['train_meta']['object_id'].head(10).tolist()
    bazin_features = extract_bazin_features(data['train_lc'], sample_ids)

    print(f"\nExtracted {len(bazin_features.columns)-1} Bazin features")
    print("\nFeature columns (first 20):")
    print([c for c in bazin_features.columns if c != 'object_id'][:20])

    print("\nSample values for r-band:")
    r_cols = [c for c in bazin_features.columns if c.startswith('r_bazin')]
    print(bazin_features[['object_id'] + r_cols[:5]].head())

    # Check how many successful fits
    successful_fits = bazin_features['r_bazin_A'].notna().sum()
    print(f"\nSuccessful r-band fits: {successful_fits}/{len(bazin_features)} ({100*successful_fits/len(bazin_features):.1f}%)")
