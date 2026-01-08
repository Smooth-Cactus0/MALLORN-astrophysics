"""
Gaussian Process features for MALLORN classification.

GP fitting extracts key hyperparameters that characterize lightcurve behavior:
- Length scale: characteristic timescale of variability
- Amplitude: strength of variability
- Interpolated fluxes: cleaner color estimates at specific times

The 2025 TDE paper (arxiv.org/abs/2509.25902) confirms GP hyperparameters
are among the most predictive features for TDE classification.

TDEs: Long length scales (smooth evolution), high amplitude
SNe: Short length scales (fast evolution), amplitude drops quickly
AGN: Intermediate length scales, stochastic patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import warnings

warnings.filterwarnings('ignore')

LSST_BANDS = ["u", "g", "r", "i", "z", "y"]


def fit_gp_single_band(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    normalize: bool = True
) -> Tuple[Optional[GaussianProcessRegressor], Dict[str, float]]:
    """
    Fit a Gaussian Process to a single-band lightcurve.

    Args:
        times: Observation times (MJD)
        fluxes: Flux values
        errors: Flux errors
        normalize: Whether to normalize fluxes before fitting

    Returns:
        Tuple of (fitted GP model, dictionary of extracted features)
    """
    features = {
        'gp_length_scale': np.nan,
        'gp_amplitude': np.nan,
        'gp_noise': np.nan,
        'gp_log_likelihood': np.nan,
    }

    if len(times) < 5:
        return None, features

    # Remove NaN values
    valid = ~(np.isnan(fluxes) | np.isnan(errors) | (errors <= 0))
    if np.sum(valid) < 5:
        return None, features

    times = times[valid]
    fluxes = fluxes[valid]
    errors = errors[valid]

    # Normalize time and flux for numerical stability
    t_min, t_max = times.min(), times.max()
    t_range = t_max - t_min
    if t_range == 0:
        return None, features

    times_norm = (times - t_min) / t_range  # Scale to [0, 1]

    if normalize:
        f_mean = np.mean(fluxes)
        f_std = np.std(fluxes)
        if f_std == 0:
            f_std = 1
        fluxes_norm = (fluxes - f_mean) / f_std
        errors_norm = errors / f_std
    else:
        fluxes_norm = fluxes
        errors_norm = errors
        f_std = 1

    # Define kernel: amplitude * RBF + noise
    # Initial length scale guess: 0.2 of time range (normalized)
    # Bounds allow wide range of timescales
    kernel = (
        ConstantKernel(1.0, (0.01, 100.0)) *  # Amplitude
        RBF(length_scale=0.2, length_scale_bounds=(0.01, 2.0)) +  # Main variability
        WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 10.0))  # Noise
    )

    try:
        # Use alpha for observation noise (squared errors)
        alpha = (errors_norm ** 2).clip(min=1e-10)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=3,
            normalize_y=False,
            random_state=42
        )

        gp.fit(times_norm.reshape(-1, 1), fluxes_norm)

        # Extract hyperparameters from fitted kernel
        # Kernel structure: ConstantKernel * RBF + WhiteKernel
        fitted_kernel = gp.kernel_

        # Parse kernel parameters
        # k1 = ConstantKernel * RBF, k2 = WhiteKernel
        k1 = fitted_kernel.k1  # ConstantKernel * RBF
        k2 = fitted_kernel.k2  # WhiteKernel

        # Get amplitude (from ConstantKernel)
        amplitude = np.sqrt(k1.k1.constant_value)  # k1.k1 is ConstantKernel

        # Get length scale (from RBF) - convert back to days
        length_scale_norm = k1.k2.length_scale  # k1.k2 is RBF
        length_scale_days = length_scale_norm * t_range

        # Get noise level
        noise_level = np.sqrt(k2.noise_level)

        # Log marginal likelihood
        log_likelihood = gp.log_marginal_likelihood_value_

        features = {
            'gp_length_scale': length_scale_days,
            'gp_amplitude': amplitude * f_std,  # Convert back to original scale
            'gp_noise': noise_level * f_std,
            'gp_log_likelihood': log_likelihood,
        }

        return gp, features

    except Exception as e:
        return None, features


def interpolate_at_times(
    gp: GaussianProcessRegressor,
    times_orig: np.ndarray,
    target_times: np.ndarray,
    t_min: float,
    t_range: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate flux at specific times using fitted GP.

    Returns:
        Tuple of (predicted mean, predicted std)
    """
    if gp is None:
        return np.full(len(target_times), np.nan), np.full(len(target_times), np.nan)

    # Normalize target times the same way
    target_norm = (target_times - t_min) / t_range

    # Clip to observed range
    target_norm = np.clip(target_norm, 0, 1)

    try:
        mean, std = gp.predict(target_norm.reshape(-1, 1), return_std=True)
        return mean, std
    except:
        return np.full(len(target_times), np.nan), np.full(len(target_times), np.nan)


def extract_gp_features_single(obj_lc: pd.DataFrame) -> Dict[str, float]:
    """
    Extract GP-based features for a single object.

    Features include:
    - Per-band GP hyperparameters (length scale, amplitude)
    - Cross-band length scale ratios
    - GP-interpolated colors at key times
    """
    features = {}

    band_gps = {}
    band_params = {}

    # Fit GP to each band
    for band in ['g', 'r', 'i', 'z']:  # Focus on main bands
        band_lc = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_lc) >= 5:
            times = band_lc['Time (MJD)'].values
            fluxes = band_lc['Flux'].values
            errors = band_lc['Flux_err'].values

            gp, gp_features = fit_gp_single_band(times, fluxes, errors)

            band_gps[band] = {
                'gp': gp,
                't_min': times.min(),
                't_range': times.max() - times.min(),
                'f_mean': np.mean(fluxes),
                'f_std': np.std(fluxes) if np.std(fluxes) > 0 else 1
            }
            band_params[band] = gp_features

            # Store per-band features
            for key, val in gp_features.items():
                features[f'{band}_{key}'] = val
        else:
            for key in ['gp_length_scale', 'gp_amplitude', 'gp_noise', 'gp_log_likelihood']:
                features[f'{band}_{key}'] = np.nan

    # Cross-band length scale ratios (diagnostic for transient type)
    for b1, b2 in [('g', 'r'), ('r', 'i')]:
        if b1 in band_params and b2 in band_params:
            ls1 = band_params[b1].get('gp_length_scale', np.nan)
            ls2 = band_params[b2].get('gp_length_scale', np.nan)
            if not np.isnan(ls1) and not np.isnan(ls2) and ls2 > 0:
                features[f'gp_ls_ratio_{b1}{b2}'] = ls1 / ls2
            else:
                features[f'gp_ls_ratio_{b1}{b2}'] = np.nan
        else:
            features[f'gp_ls_ratio_{b1}{b2}'] = np.nan

    # Mean length scale across bands (overall timescale)
    valid_ls = [band_params[b].get('gp_length_scale', np.nan)
                for b in ['g', 'r', 'i'] if b in band_params]
    valid_ls = [ls for ls in valid_ls if not np.isnan(ls)]

    if valid_ls:
        features['gp_mean_length_scale'] = np.mean(valid_ls)
        features['gp_std_length_scale'] = np.std(valid_ls) if len(valid_ls) > 1 else 0
    else:
        features['gp_mean_length_scale'] = np.nan
        features['gp_std_length_scale'] = np.nan

    # Mean amplitude across bands
    valid_amp = [band_params[b].get('gp_amplitude', np.nan)
                 for b in ['g', 'r', 'i'] if b in band_params]
    valid_amp = [a for a in valid_amp if not np.isnan(a)]

    if valid_amp:
        features['gp_mean_amplitude'] = np.mean(valid_amp)
    else:
        features['gp_mean_amplitude'] = np.nan

    return features


def extract_gp_features(
    lightcurves: pd.DataFrame,
    metadata: pd.DataFrame,
    object_ids: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract GP-based features for multiple objects.

    Args:
        lightcurves: DataFrame with lightcurve data
        metadata: DataFrame with object metadata
        object_ids: Optional list of object IDs to process
        verbose: Whether to print progress

    Returns:
        DataFrame with GP features for each object
    """
    if object_ids is None:
        object_ids = lightcurves['object_id'].unique()

    # Pre-group by object_id for faster lookup
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if verbose and (i + 1) % 100 == 0:
            print(f"    GP features: {i+1}/{len(object_ids)} objects processed")

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue

        features = extract_gp_features_single(obj_lc)
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

    print("\nExtracting GP features for first 20 objects...")
    sample_ids = data['train_meta']['object_id'].head(20).tolist()

    gp_features = extract_gp_features(
        data['train_lc'],
        data['train_meta'],
        sample_ids
    )

    print(f"\nExtracted {len(gp_features.columns)-1} GP features")
    print("\nFeature columns:")
    print([c for c in gp_features.columns if c != 'object_id'])
    print("\nSample values:")
    print(gp_features[['object_id', 'g_gp_length_scale', 'r_gp_length_scale', 'gp_mean_length_scale']].head(10))
