"""
Multi-band Gaussian Process feature extraction for MALLORN classification.

Following the 2025 TDE paper (arxiv.org/abs/2509.25902):
- Uses george package with 2D Matérn-3/2 kernel
- Fits all bands simultaneously (time × wavelength)
- Extracts: amplitude, time length scale, wavelength length scale
- Also provides GP-interpolated fluxes for cleaner color estimation

Key advantage over per-band GP:
- Models correlations BETWEEN bands (achromatic vs chromatic variability)
- Single coherent model for the entire multi-band lightcurve
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
import george
from george import kernels
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# LSST band wavelengths in Angstroms
BAND_WAVELENGTHS = {
    'u': 3670, 'g': 4825, 'r': 6222,
    'i': 7545, 'z': 8691, 'y': 9710
}

LSST_BANDS = ['u', 'g', 'r', 'i', 'z', 'y']


def prepare_multiband_data(obj_lc: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare lightcurve data for multi-band GP fitting.

    Args:
        obj_lc: DataFrame with Time (MJD), Flux, Flux_err, Filter columns

    Returns:
        Tuple of (X, y, yerr) where X is (N, 2) array with [time, wavelength]
    """
    times = []
    wavelengths = []
    fluxes = []
    errors = []

    for _, row in obj_lc.iterrows():
        band = row['Filter']
        if band not in BAND_WAVELENGTHS:
            continue

        flux = row['Flux']
        flux_err = row['Flux_err']

        # Skip invalid data
        if np.isnan(flux) or np.isnan(flux_err) or flux_err <= 0:
            continue

        times.append(row['Time (MJD)'])
        wavelengths.append(BAND_WAVELENGTHS[band])
        fluxes.append(flux)
        errors.append(flux_err)

    if len(times) < 10:
        return None, None, None

    times = np.array(times)
    wavelengths = np.array(wavelengths)
    fluxes = np.array(fluxes)
    errors = np.array(errors)

    # Normalize time to start at 0
    times = times - times.min()

    # Normalize flux for numerical stability
    flux_scale = np.median(np.abs(fluxes[fluxes != 0]))
    if flux_scale == 0:
        flux_scale = 1.0
    fluxes = fluxes / flux_scale
    errors = errors / flux_scale

    # Stack into 2D input
    X = np.column_stack([times, wavelengths])

    return X, fluxes, errors, flux_scale


def fit_multiband_gp(
    X: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    init_time_scale: float = 100.0,
    init_wave_scale: float = 6000.0
) -> Tuple[Optional[george.GP], Dict[str, float]]:
    """
    Fit a 2D Gaussian Process to multi-band lightcurve data.

    Uses Matérn-3/2 kernel following the 2025 TDE paper.

    Args:
        X: (N, 2) array with [time, wavelength]
        y: Normalized flux values
        yerr: Flux errors
        init_time_scale: Initial time length scale (days)
        init_wave_scale: Initial wavelength length scale (Angstroms)

    Returns:
        Tuple of (fitted GP, extracted features dict)
    """
    features = {
        'gp2d_amplitude': np.nan,
        'gp2d_time_scale': np.nan,
        'gp2d_wave_scale': np.nan,
        'gp2d_log_likelihood': np.nan,
        'gp2d_time_wave_ratio': np.nan,
    }

    if X is None or len(X) < 10:
        return None, features

    try:
        # Initial amplitude estimate from data variance
        init_amplitude = np.var(y)

        # 2D Matérn-3/2 kernel (separable in time and wavelength)
        # K(t1,w1; t2,w2) = amp * Matern32(|t1-t2|/l_t) * Matern32(|w1-w2|/l_w)
        kernel = init_amplitude * kernels.Matern32Kernel(
            metric=[init_time_scale**2, init_wave_scale**2],
            ndim=2
        )

        # Create GP with white noise
        gp = george.GP(kernel, mean=np.mean(y), fit_mean=True)

        # Compute the likelihood
        gp.compute(X, yerr)

        # Optimize hyperparameters
        def neg_log_likelihood(p):
            gp.set_parameter_vector(p)
            try:
                ll = gp.log_likelihood(y, quiet=True)
                return -ll if np.isfinite(ll) else 1e25
            except:
                return 1e25

        def grad_neg_log_likelihood(p):
            gp.set_parameter_vector(p)
            try:
                return -gp.grad_log_likelihood(y, quiet=True)
            except:
                return np.zeros_like(p)

        # Run optimization
        p0 = gp.get_parameter_vector()
        result = minimize(
            neg_log_likelihood,
            p0,
            jac=grad_neg_log_likelihood,
            method='L-BFGS-B',
            options={'maxiter': 100}
        )

        # Update GP with optimized parameters
        gp.set_parameter_vector(result.x)

        # Extract features from optimized kernel
        # george stores log of parameters
        params = gp.get_parameter_vector()

        # Kernel structure: log_amplitude, log_metric[0], log_metric[1], mean
        log_amp = params[0]
        log_metric_time = params[1]
        log_metric_wave = params[2]

        amplitude = np.exp(log_amp)
        time_scale = np.sqrt(np.exp(log_metric_time))
        wave_scale = np.sqrt(np.exp(log_metric_wave))

        features = {
            'gp2d_amplitude': amplitude,
            'gp2d_time_scale': time_scale,
            'gp2d_wave_scale': wave_scale,
            'gp2d_log_likelihood': -result.fun,
            'gp2d_time_wave_ratio': time_scale / (wave_scale / 1000),  # Normalize wavelength to thousands
        }

        return gp, features

    except Exception as e:
        return None, features


def interpolate_multiband(
    gp: george.GP,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    peak_time: float,
    flux_scale: float,
    epochs: List[float] = [0, 20, 50, 100]
) -> Dict[str, float]:
    """
    Interpolate flux at specific epochs and bands using the fitted GP.

    This provides CLEANER color estimates than raw observations.

    Args:
        gp: Fitted GP model
        X_obs: Original observation coordinates
        y_obs: Original flux values (normalized)
        peak_time: Time of peak flux
        flux_scale: Scale factor used for normalization
        epochs: Days relative to peak to interpolate

    Returns:
        Dictionary with interpolated fluxes and colors
    """
    features = {}

    if gp is None:
        # Return NaN features
        for epoch in epochs:
            for band in ['g', 'r', 'i']:
                features[f'gp_flux_{band}_{epoch}d'] = np.nan
            features[f'gp_gr_color_{epoch}d'] = np.nan
            features[f'gp_ri_color_{epoch}d'] = np.nan
        return features

    try:
        # Create prediction grid
        pred_times = [peak_time + e for e in epochs]
        pred_bands = ['g', 'r', 'i']

        for epoch, pred_time in zip(epochs, pred_times):
            epoch_fluxes = {}

            for band in pred_bands:
                wave = BAND_WAVELENGTHS[band]
                X_pred = np.array([[pred_time, wave]])

                # Predict
                mu, var = gp.predict(y_obs, X_pred, return_var=True)
                flux = mu[0] * flux_scale  # Denormalize
                epoch_fluxes[band] = flux
                features[f'gp_flux_{band}_{epoch}d'] = flux

            # Compute colors (magnitude difference)
            g_flux = epoch_fluxes.get('g', np.nan)
            r_flux = epoch_fluxes.get('r', np.nan)
            i_flux = epoch_fluxes.get('i', np.nan)

            if g_flux > 0 and r_flux > 0:
                features[f'gp_gr_color_{epoch}d'] = -2.5 * np.log10(g_flux / r_flux)
            else:
                features[f'gp_gr_color_{epoch}d'] = np.nan

            if r_flux > 0 and i_flux > 0:
                features[f'gp_ri_color_{epoch}d'] = -2.5 * np.log10(r_flux / i_flux)
            else:
                features[f'gp_ri_color_{epoch}d'] = np.nan

        # Color evolution (slope)
        gr_0 = features.get('gp_gr_color_0d', np.nan)
        gr_50 = features.get('gp_gr_color_50d', np.nan)
        gr_100 = features.get('gp_gr_color_100d', np.nan)

        if not np.isnan(gr_0) and not np.isnan(gr_50):
            features['gp_gr_slope_50d'] = (gr_50 - gr_0) / 50.0
        else:
            features['gp_gr_slope_50d'] = np.nan

        if not np.isnan(gr_0) and not np.isnan(gr_100):
            features['gp_gr_slope_100d'] = (gr_100 - gr_0) / 100.0
        else:
            features['gp_gr_slope_100d'] = np.nan

    except Exception as e:
        # Return NaN features on error
        for epoch in epochs:
            for band in ['g', 'r', 'i']:
                features[f'gp_flux_{band}_{epoch}d'] = np.nan
            features[f'gp_gr_color_{epoch}d'] = np.nan
            features[f'gp_ri_color_{epoch}d'] = np.nan
        features['gp_gr_slope_50d'] = np.nan
        features['gp_gr_slope_100d'] = np.nan

    return features


def extract_multiband_gp_features_single(obj_lc: pd.DataFrame) -> Dict[str, float]:
    """
    Extract multi-band GP features for a single object.

    Args:
        obj_lc: DataFrame with lightcurve data

    Returns:
        Dictionary with GP hyperparameters and interpolated features
    """
    features = {}

    # Prepare data
    result = prepare_multiband_data(obj_lc)
    if result[0] is None:
        # Return empty features
        features = {
            'gp2d_amplitude': np.nan,
            'gp2d_time_scale': np.nan,
            'gp2d_wave_scale': np.nan,
            'gp2d_log_likelihood': np.nan,
            'gp2d_time_wave_ratio': np.nan,
        }
        for epoch in [0, 20, 50, 100]:
            for band in ['g', 'r', 'i']:
                features[f'gp_flux_{band}_{epoch}d'] = np.nan
            features[f'gp_gr_color_{epoch}d'] = np.nan
            features[f'gp_ri_color_{epoch}d'] = np.nan
        features['gp_gr_slope_50d'] = np.nan
        features['gp_gr_slope_100d'] = np.nan
        return features

    X, y, yerr, flux_scale = result

    # Fit GP
    gp, gp_features = fit_multiband_gp(X, y, yerr)
    features.update(gp_features)

    # Find peak time (from r-band if available, else max flux)
    r_mask = obj_lc['Filter'] == 'r'
    if r_mask.sum() > 0:
        r_lc = obj_lc[r_mask]
        peak_idx = r_lc['Flux'].idxmax()
        peak_time = r_lc.loc[peak_idx, 'Time (MJD)'] - obj_lc['Time (MJD)'].min()
    else:
        peak_idx = obj_lc['Flux'].idxmax()
        peak_time = obj_lc.loc[peak_idx, 'Time (MJD)'] - obj_lc['Time (MJD)'].min()

    # Interpolate at specific epochs
    interp_features = interpolate_multiband(gp, X, y, peak_time, flux_scale)
    features.update(interp_features)

    return features


def extract_multiband_gp_features(
    lightcurves: pd.DataFrame,
    metadata: pd.DataFrame,
    object_ids: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract multi-band GP features for multiple objects.

    Args:
        lightcurves: DataFrame with lightcurve data
        metadata: DataFrame with object metadata
        object_ids: Optional list of object IDs to process
        verbose: Whether to print progress

    Returns:
        DataFrame with multi-band GP features for each object
    """
    if object_ids is None:
        object_ids = lightcurves['object_id'].unique()

    # Pre-group by object_id
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if verbose and (i + 1) % 100 == 0:
            print(f"    Multi-band GP: {i+1}/{len(object_ids)} objects processed", flush=True)

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue

        features = extract_multiband_gp_features_single(obj_lc)
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

    print("\nExtracting multi-band GP features for first 20 objects...")
    sample_ids = data['train_meta']['object_id'].head(20).tolist()

    gp_features = extract_multiband_gp_features(
        data['train_lc'],
        data['train_meta'],
        sample_ids
    )

    print(f"\nExtracted {len(gp_features.columns)-1} multi-band GP features")
    print("\nFeature columns:")
    print([c for c in gp_features.columns if c != 'object_id'])
    print("\nSample values:")
    print(gp_features[['object_id', 'gp2d_time_scale', 'gp2d_wave_scale', 'gp_gr_color_0d', 'gp_gr_color_50d']].head(10))
