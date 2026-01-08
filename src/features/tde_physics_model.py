"""
TDE Physics-Based Parametric Model for MALLORN (Technique #9)

Custom parametric lightcurve model based on TDE astrophysics.

Key Physics:
1. Fallback rate: Ṁ(t) ∝ t^(-5/3) (Guillochon & Ramirez-Ruiz 2013)
   - From relativistic orbital dynamics of debris stream
   - Distinctive power-law decay (not exponential like SNe)

2. Super-Eddington accretion (Lodato & Rossi 2011):
   - L ∝ Ṁ for sub-Eddington
   - L ∝ Ṁ^(2/3) for super-Eddington (photon trapping)

3. Circularization timescale:
   - Not instantaneous peak (unlike Bazin sigmoid)
   - Can show "stalled rise" or plateau phase

Model Variants:
- tde_hybrid: Exponential + power law (flexible, robust fitting)
- tde_guillochon: Pure power law with exponential cutoff (physics-accurate)
- tde_piecewise: Linear rise + power law decay (simple)

References:
- Guillochon & Ramirez-Ruiz (2013): ApJ 767, 25
- Lodato & Rossi (2011): MNRAS 410, 359
- van Velzen et al. (2020): Space Sci. Rev. 216, 124
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

LSST_BANDS = ["u", "g", "r", "i", "z", "y"]


def tde_hybrid_model(t: np.ndarray, A: float, t0: float, tau_rise: float,
                     tau_fall: float, alpha: float, B: float) -> np.ndarray:
    """
    TDE hybrid model: Sigmoid rise + exponential decay with power law modulation.

    f(t) = A * [1/(1 + exp(-(t-t0)/tau_rise))] *
           [exp(-(t-t0)/tau_fall)] *
           [1 + (t-t0)/tau_fall]^(-alpha) + B

    This combines:
    - Sigmoid rise (captures circularization timescale)
    - Exponential decay (early-time behavior)
    - Power law decay (late-time t^(-5/3) fallback)

    Args:
        t: Time array (MJD)
        A: Peak amplitude above baseline
        t0: Time of peak brightness
        tau_rise: Rise timescale (days, ~20-50 for TDEs)
        tau_fall: Exponential decay timescale (days, ~100-300)
        alpha: Power law index (~1.67 for t^(-5/3) fallback)
        B: Baseline flux

    Returns:
        Model flux at each time
    """
    # Rise term (sigmoid)
    rise_term = 1.0 / (1.0 + np.exp(-(t - t0) / tau_rise))

    # Decay terms
    dt = t - t0
    exp_decay = np.exp(-dt / tau_fall)

    # Power law term (only for t > t0 to avoid negative bases)
    power_law = np.ones_like(t)
    mask_post_peak = dt > 0
    if np.any(mask_post_peak):
        # Use (1 + dt/tau_fall)^(-alpha) to avoid singularity at t=t0
        power_law[mask_post_peak] = (1.0 + dt[mask_post_peak] / tau_fall) ** (-alpha)

    return A * rise_term * exp_decay * power_law + B


def tde_guillochon_model(t: np.ndarray, A: float, t0: float, tau_rise: float,
                         tau_fall: float, B: float) -> np.ndarray:
    """
    Guillochon model: Power law rise + exponential decay.

    f(t) = A * (t/t0)^α * exp(-(t-t0)/tau_fall) + B  for t > 0
           B                                           for t ≤ 0

    This is closer to the theoretical TDE model but with fixed alpha.

    Args:
        t: Time array (MJD)
        A: Peak amplitude
        t0: Time of peak
        tau_rise: Effective rise time (controls power law steepness)
        tau_fall: Decay timescale
        B: Baseline

    Returns:
        Model flux at each time
    """
    # Normalize time
    t_norm = t - (t0 - 3 * tau_rise)  # Shift so peak aligns with t0

    # Power law rise (alpha ~ 0.3-0.5 for TDEs)
    alpha_rise = 0.4  # Fixed based on theory
    rise_term = np.where(t_norm > 0,
                         (t_norm / (3 * tau_rise)) ** alpha_rise,
                         0.0)

    # Exponential decay after peak
    decay_term = np.exp(-(t - t0) / tau_fall)

    # Combine (cap rise at 1.0)
    rise_capped = np.minimum(rise_term, 1.0)

    return A * rise_capped * decay_term + B


def tde_piecewise_model(t: np.ndarray, A: float, t0: float, tau_rise: float,
                        tau_fall: float, alpha: float, B: float) -> np.ndarray:
    """
    Piecewise model: Linear rise + power law decay.

    f(t) = A * min((t-t_start)/tau_rise, 1) * (1 + (t-t0)/tau_fall)^(-alpha) + B

    Simpler than hybrid, easier to fit.

    Args:
        t: Time array
        A: Amplitude
        t0: Peak time
        tau_rise: Rise time (linear)
        tau_fall: Power law timescale
        alpha: Power law index
        B: Baseline

    Returns:
        Model flux
    """
    t_start = t0 - tau_rise

    # Linear rise
    rise_term = np.clip((t - t_start) / tau_rise, 0.0, 1.0)

    # Power law decay
    dt = t - t0
    decay_term = np.ones_like(t)
    mask_post = dt > 0
    if np.any(mask_post):
        decay_term[mask_post] = (1.0 + dt[mask_post] / tau_fall) ** (-alpha)

    return A * rise_term * decay_term + B


def fit_tde_single_band(times: np.ndarray, fluxes: np.ndarray,
                        flux_errors: np.ndarray,
                        model_type: str = 'hybrid') -> Dict[str, float]:
    """
    Fit TDE parametric model to single-band lightcurve.

    Args:
        times: Observation times (MJD)
        fluxes: Flux measurements
        flux_errors: Flux uncertainties
        model_type: 'hybrid', 'guillochon', or 'piecewise'

    Returns:
        Dictionary of fitted parameters and derived features
    """
    if len(times) < 6:  # Need at least 6 points for 6-parameter model
        return {
            'tde_A': np.nan, 'tde_t0': np.nan, 'tde_tau_rise': np.nan,
            'tde_tau_fall': np.nan, 'tde_alpha': np.nan, 'tde_B': np.nan,
            'tde_fit_chi2': np.nan, 'tde_alpha_value': np.nan,
            'tde_peak_flux': np.nan, 'tde_model_type': model_type
        }

    # Select model function
    if model_type == 'hybrid':
        model_func = tde_hybrid_model
        n_params = 6
    elif model_type == 'guillochon':
        model_func = tde_guillochon_model
        n_params = 5
    else:  # piecewise
        model_func = tde_piecewise_model
        n_params = 6

    # Estimate initial parameters from data
    peak_idx = np.argmax(fluxes)
    t_peak = times[peak_idx]
    f_peak = fluxes[peak_idx]
    f_baseline = np.median(fluxes[fluxes < np.percentile(fluxes, 40)])

    A_guess = f_peak - f_baseline
    t0_guess = t_peak

    # Estimate rise time (time from start to peak)
    pre_peak = times < t_peak
    if np.any(pre_peak):
        tau_rise_guess = (t_peak - times[pre_peak][0]) / 2
    else:
        tau_rise_guess = 30.0  # Default
    tau_rise_guess = np.clip(tau_rise_guess, 5.0, 100.0)

    # Estimate decay time (time from peak to 50% intensity)
    post_peak = times > t_peak
    if np.any(post_peak) and np.any(fluxes[post_peak] < f_peak * 0.5):
        t_half = times[post_peak][fluxes[post_peak] < f_peak * 0.5][0]
        tau_fall_guess = (t_half - t_peak) / np.log(2)
    else:
        tau_fall_guess = 100.0
    tau_fall_guess = np.clip(tau_fall_guess, 10.0, 500.0)

    # Initial guess for power law index (TDE theory: ~5/3)
    alpha_guess = 1.67

    B_guess = f_baseline

    # Set up bounds
    if model_type == 'hybrid':
        p0 = [A_guess, t0_guess, tau_rise_guess, tau_fall_guess, alpha_guess, B_guess]
        bounds = (
            [0, times[0] - 50, 1, 10, 0.5, -np.inf],  # Lower bounds
            [np.inf, times[-1] + 50, 200, 1000, 3.0, np.inf]  # Upper bounds
        )
    elif model_type == 'guillochon':
        p0 = [A_guess, t0_guess, tau_rise_guess, tau_fall_guess, B_guess]
        bounds = (
            [0, times[0] - 50, 1, 10, -np.inf],
            [np.inf, times[-1] + 50, 200, 1000, np.inf]
        )
    else:  # piecewise
        p0 = [A_guess, t0_guess, tau_rise_guess, tau_fall_guess, alpha_guess, B_guess]
        bounds = (
            [0, times[0] - 50, 5, 10, 0.5, -np.inf],
            [np.inf, times[-1] + 50, 200, 1000, 3.0, np.inf]
        )

    # Weighted fit (inverse variance weights)
    sigma = np.where(flux_errors > 0, flux_errors, 1.0)

    try:
        popt, pcov = curve_fit(
            model_func, times, fluxes,
            p0=p0, bounds=bounds, sigma=sigma,
            maxfev=3000, method='trf'
        )

        # Extract parameters
        if model_type == 'guillochon':
            A, t0, tau_rise, tau_fall, B = popt
            alpha = 1.67  # Fixed for Guillochon model
        else:
            A, t0, tau_rise, tau_fall, alpha, B = popt

        # Clip extreme values
        A = np.clip(A, -1e6, 1e6)
        t0 = np.clip(t0, times[0] - 100, times[-1] + 100)
        tau_rise = np.clip(tau_rise, 0.1, 300)
        tau_fall = np.clip(tau_fall, 1, 2000)
        alpha = np.clip(alpha, 0.1, 5.0)
        B = np.clip(B, -1e6, 1e6)

        # Compute fit quality
        model_fluxes = model_func(times, *popt)
        residuals = fluxes - model_fluxes
        chi2 = np.sum((residuals / sigma) ** 2)
        reduced_chi2 = np.clip(chi2 / (len(times) - n_params), 0, 1e6)

        # Derived features
        peak_flux = np.clip(A + B, -1e6, 1e6)

        return {
            'tde_A': A,
            'tde_t0': t0,
            'tde_tau_rise': tau_rise,
            'tde_tau_fall': tau_fall,
            'tde_alpha': alpha,
            'tde_B': B,
            'tde_fit_chi2': reduced_chi2,
            'tde_alpha_value': alpha,  # Store actual fitted alpha
            'tde_peak_flux': peak_flux,
            'tde_model_type': model_type
        }

    except Exception as e:
        # Fitting failed
        return {
            'tde_A': np.nan, 'tde_t0': np.nan, 'tde_tau_rise': np.nan,
            'tde_tau_fall': np.nan, 'tde_alpha': np.nan, 'tde_B': np.nan,
            'tde_fit_chi2': np.nan, 'tde_alpha_value': np.nan,
            'tde_peak_flux': np.nan, 'tde_model_type': model_type
        }


def extract_tde_features_single(obj_lc: pd.DataFrame,
                                 model_type: str = 'hybrid') -> Dict[str, float]:
    """
    Extract TDE model features for a single object across all bands.

    Args:
        obj_lc: Lightcurve DataFrame for one object
        model_type: Model variant to use

    Returns:
        Dictionary with TDE features for all bands + cross-band metrics
    """
    features = {}

    # Fit each band
    alpha_values = []
    tau_fall_values = []
    tau_rise_values = []
    fit_chi2_values = []

    for band in LSST_BANDS:
        band_lc = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_lc) < 6:
            # Insufficient data
            for key in ['tde_A', 'tde_t0', 'tde_tau_rise', 'tde_tau_fall',
                       'tde_alpha', 'tde_B', 'tde_fit_chi2', 'tde_alpha_value', 'tde_peak_flux']:
                features[f'{band}_{key}'] = np.nan
            continue

        times = band_lc['Time (MJD)'].values
        fluxes = band_lc['Flux'].values
        flux_errors = band_lc['Flux_err'].values

        # Fit TDE model
        fit_results = fit_tde_single_band(times, fluxes, flux_errors, model_type)

        # Store per-band features
        for key, val in fit_results.items():
            if key != 'tde_model_type':
                features[f'{band}_{key}'] = val

        # Collect for cross-band analysis
        if not np.isnan(fit_results['tde_alpha']):
            alpha_values.append(fit_results['tde_alpha'])
            tau_fall_values.append(fit_results['tde_tau_fall'])
            tau_rise_values.append(fit_results['tde_tau_rise'])
            fit_chi2_values.append(fit_results['tde_fit_chi2'])

    # Cross-band consistency features
    if len(alpha_values) >= 2:
        # TDEs should have consistent alpha ~ 5/3 across bands
        features['tde_alpha_consistency'] = np.std(alpha_values) / np.mean(np.abs(alpha_values))
        features['tde_mean_alpha'] = np.mean(alpha_values)
        features['tde_alpha_deviation'] = np.abs(np.mean(alpha_values) - 1.67)  # Distance from theory
    else:
        features['tde_alpha_consistency'] = np.nan
        features['tde_mean_alpha'] = np.nan
        features['tde_alpha_deviation'] = np.nan

    if len(tau_fall_values) >= 2:
        features['tde_tau_fall_consistency'] = np.std(tau_fall_values) / np.mean(tau_fall_values)
    else:
        features['tde_tau_fall_consistency'] = np.nan

    if len(tau_rise_values) >= 2:
        features['tde_tau_rise_consistency'] = np.std(tau_rise_values) / np.mean(tau_rise_values)
    else:
        features['tde_tau_rise_consistency'] = np.nan

    if len(fit_chi2_values) > 0:
        features['tde_avg_fit_chi2'] = np.mean(fit_chi2_values)
        features['tde_fit_quality_dispersion'] = np.std(fit_chi2_values)
    else:
        features['tde_avg_fit_chi2'] = np.nan
        features['tde_fit_quality_dispersion'] = np.nan

    return features


def extract_tde_features(lightcurves: pd.DataFrame,
                         object_ids: Optional[List[str]] = None,
                         model_type: str = 'hybrid') -> pd.DataFrame:
    """
    Extract TDE model features for multiple objects.

    Args:
        lightcurves: DataFrame with lightcurve data
        object_ids: Optional list of object IDs
        model_type: 'hybrid', 'guillochon', or 'piecewise'

    Returns:
        DataFrame with TDE features
    """
    if object_ids is None:
        object_ids = lightcurves['object_id'].unique()

    # Pre-group for efficiency
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if (i + 1) % 500 == 0:
            print(f"    TDE: {i+1}/{len(object_ids)} objects processed")

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue

        features = extract_tde_features_single(obj_lc, model_type)
        features['object_id'] = obj_id
        all_features.append(features)

    return pd.DataFrame(all_features)


if __name__ == "__main__":
    # Test TDE model
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_all_data

    print("Testing TDE physics model...")
    print("=" * 80)

    # Test model functions
    t_test = np.linspace(0, 200, 100)

    print("\n1. Testing hybrid model...")
    flux_hybrid = tde_hybrid_model(t_test, A=100, t0=50, tau_rise=20, tau_fall=80, alpha=1.67, B=10)
    print(f"   Flux range: [{flux_hybrid.min():.1f}, {flux_hybrid.max():.1f}]")
    print(f"   Peak at t={t_test[np.argmax(flux_hybrid)]:.1f} days")

    print("\n2. Testing Guillochon model...")
    flux_guill = tde_guillochon_model(t_test, A=100, t0=50, tau_rise=20, tau_fall=80, B=10)
    print(f"   Flux range: [{flux_guill.min():.1f}, {flux_guill.max():.1f}]")

    print("\n3. Testing piecewise model...")
    flux_piece = tde_piecewise_model(t_test, A=100, t0=50, tau_rise=30, tau_fall=80, alpha=1.67, B=10)
    print(f"   Flux range: [{flux_piece.min():.1f}, {flux_piece.max():.1f}]")

    print("\n4. Testing on real data (first 10 objects)...")
    data = load_all_data()
    sample_ids = data['train_meta']['object_id'].head(10).tolist()

    tde_features = extract_tde_features(data['train_lc'], sample_ids, model_type='hybrid')

    print(f"\n5. Extracted {len(tde_features.columns)-1} TDE features")
    print("\n   Feature columns (first 15):")
    print([c for c in tde_features.columns if c != 'object_id'][:15])

    print("\n6. Sample values for r-band:")
    r_cols = [c for c in tde_features.columns if c.startswith('r_tde')]
    print(tde_features[['object_id'] + r_cols[:5]].head())

    print("\n7. Feature coverage (non-NaN):")
    for col in r_cols[:5]:
        coverage = tde_features[col].notna().sum() / len(tde_features)
        print(f"   {col}: {100*coverage:.1f}%")

    print("\n" + "=" * 80)
    print("TDE model test complete!")
