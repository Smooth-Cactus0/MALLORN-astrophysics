"""
DTW (Dynamic Time Warping) Features for TDE Classification

Extracts shape-based features using DTW distance to templates.
Key insight: TDEs have characteristic shapes that DTW can match
regardless of time dilation (redshift) or observation cadence.

Features extracted:
1. DTW distance to TDE template (per band)
2. DTW distance to non-TDE template (per band)
3. Distance ratio (TDE vs non-TDE affinity)
4. Warping path characteristics (how much time stretching needed)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

LSST_BANDS = ['g', 'r', 'i']  # Focus on bands with best coverage


def normalize_lightcurve(times: np.ndarray, fluxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize lightcurve for shape comparison.

    - Time: normalize to [0, 1] range (removes duration differences)
    - Flux: normalize to [0, 1] range (removes brightness differences)

    This isolates the SHAPE, which is what we care about.
    """
    if len(times) < 2 or len(fluxes) < 2:
        return np.array([0, 1]), np.array([0, 0])

    # Time normalization
    t_min, t_max = times.min(), times.max()
    if t_max - t_min > 0:
        t_norm = (times - t_min) / (t_max - t_min)
    else:
        t_norm = np.zeros_like(times)

    # Flux normalization (min-max)
    f_min, f_max = fluxes.min(), fluxes.max()
    if f_max - f_min > 0:
        f_norm = (fluxes - f_min) / (f_max - f_min)
    else:
        f_norm = np.zeros_like(fluxes)

    return t_norm, f_norm


def resample_lightcurve(times: np.ndarray, fluxes: np.ndarray, n_points: int = 50) -> np.ndarray:
    """
    Resample lightcurve to fixed number of points for DTW comparison.

    Uses linear interpolation on normalized time grid.
    Returns only flux values (time is implicit in index).
    """
    if len(times) < 2:
        return np.zeros(n_points)

    t_norm, f_norm = normalize_lightcurve(times, fluxes)

    # Sort by time
    sort_idx = np.argsort(t_norm)
    t_sorted = t_norm[sort_idx]
    f_sorted = f_norm[sort_idx]

    # Remove duplicates (take mean)
    unique_t, unique_idx = np.unique(t_sorted, return_index=True)
    if len(unique_t) < 2:
        return np.zeros(n_points)

    unique_f = f_sorted[unique_idx]

    # Interpolate to regular grid
    try:
        interp_func = interp1d(unique_t, unique_f, kind='linear',
                               bounds_error=False, fill_value=(unique_f[0], unique_f[-1]))
        t_grid = np.linspace(0, 1, n_points)
        f_resampled = interp_func(t_grid)
        return f_resampled
    except Exception:
        return np.zeros(n_points)


def create_templates(lightcurves: pd.DataFrame,
                     train_meta: pd.DataFrame,
                     bands: List[str] = LSST_BANDS,
                     n_points: int = 50) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Create average TDE and non-TDE templates for each band.

    Returns:
        {
            'tde': {'g': array, 'r': array, 'i': array},
            'non_tde': {'g': array, 'r': array, 'i': array}
        }
    """
    print("    Creating DTW templates...")

    # Get TDE and non-TDE object IDs
    tde_ids = set(train_meta[train_meta['target'] == 1]['object_id'])
    non_tde_ids = set(train_meta[train_meta['target'] == 0]['object_id'])

    # Pre-group lightcurves
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    templates = {'tde': {}, 'non_tde': {}}

    for band in bands:
        tde_curves = []
        non_tde_curves = []

        for obj_id, obj_lc in grouped.items():
            band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

            if len(band_data) < 5:
                continue

            times = band_data['Time (MJD)'].values
            fluxes = band_data['Flux'].values

            resampled = resample_lightcurve(times, fluxes, n_points)

            if obj_id in tde_ids:
                tde_curves.append(resampled)
            elif obj_id in non_tde_ids:
                non_tde_curves.append(resampled)

        # Average templates
        if tde_curves:
            templates['tde'][band] = np.median(tde_curves, axis=0)
        else:
            templates['tde'][band] = np.zeros(n_points)

        if non_tde_curves:
            templates['non_tde'][band] = np.median(non_tde_curves, axis=0)
        else:
            templates['non_tde'][band] = np.zeros(n_points)

        print(f"      {band}-band: {len(tde_curves)} TDE, {len(non_tde_curves)} non-TDE curves")

    return templates


def compute_dtw_distance(curve1: np.ndarray, curve2: np.ndarray) -> Tuple[float, float]:
    """
    Compute DTW distance and warping amount between two curves.

    Returns:
        (dtw_distance, warping_amount)

    warping_amount: measures how much time stretching was needed
                    0 = perfect alignment, higher = more warping
    """
    if len(curve1) < 2 or len(curve2) < 2:
        return np.nan, np.nan

    try:
        distance, path = fastdtw(curve1.reshape(-1, 1), curve2.reshape(-1, 1), dist=euclidean)

        # Compute warping amount
        # Perfect alignment: path[i] = (i, i)
        # Warping: deviation from diagonal
        path = np.array(path)
        diagonal = np.arange(len(path))

        # Average deviation from diagonal (normalized)
        warping = np.mean(np.abs(path[:, 0] - path[:, 1])) / len(curve1)

        return distance, warping

    except Exception:
        return np.nan, np.nan


def extract_dtw_features_single(obj_lc: pd.DataFrame,
                                 templates: Dict[str, Dict[str, np.ndarray]],
                                 bands: List[str] = LSST_BANDS,
                                 n_points: int = 50) -> Dict[str, float]:
    """
    Extract DTW features for a single object.
    """
    features = {}

    dtw_tde_total = 0
    dtw_non_tde_total = 0
    n_bands = 0

    for band in bands:
        band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_data) < 5:
            features[f'{band}_dtw_tde'] = np.nan
            features[f'{band}_dtw_non_tde'] = np.nan
            features[f'{band}_dtw_ratio'] = np.nan
            features[f'{band}_dtw_warp_tde'] = np.nan
            features[f'{band}_dtw_warp_non_tde'] = np.nan
            continue

        times = band_data['Time (MJD)'].values
        fluxes = band_data['Flux'].values

        # Resample object lightcurve
        obj_curve = resample_lightcurve(times, fluxes, n_points)

        # DTW to TDE template
        dtw_tde, warp_tde = compute_dtw_distance(obj_curve, templates['tde'][band])

        # DTW to non-TDE template
        dtw_non_tde, warp_non_tde = compute_dtw_distance(obj_curve, templates['non_tde'][band])

        features[f'{band}_dtw_tde'] = dtw_tde
        features[f'{band}_dtw_non_tde'] = dtw_non_tde

        # Ratio: lower = more TDE-like
        if dtw_non_tde > 0 and not np.isnan(dtw_tde) and not np.isnan(dtw_non_tde):
            features[f'{band}_dtw_ratio'] = dtw_tde / dtw_non_tde
            dtw_tde_total += dtw_tde
            dtw_non_tde_total += dtw_non_tde
            n_bands += 1
        else:
            features[f'{band}_dtw_ratio'] = np.nan

        # Warping amounts
        features[f'{band}_dtw_warp_tde'] = warp_tde
        features[f'{band}_dtw_warp_non_tde'] = warp_non_tde

        # Warping difference (TDEs should warp less to TDE template)
        if not np.isnan(warp_tde) and not np.isnan(warp_non_tde):
            features[f'{band}_warp_diff'] = warp_tde - warp_non_tde
        else:
            features[f'{band}_warp_diff'] = np.nan

    # Aggregate features across bands
    if n_bands > 0:
        features['dtw_tde_mean'] = dtw_tde_total / n_bands
        features['dtw_non_tde_mean'] = dtw_non_tde_total / n_bands
        features['dtw_ratio_mean'] = dtw_tde_total / dtw_non_tde_total
    else:
        features['dtw_tde_mean'] = np.nan
        features['dtw_non_tde_mean'] = np.nan
        features['dtw_ratio_mean'] = np.nan

    return features


def extract_dtw_features(lightcurves: pd.DataFrame,
                         object_ids: List[str],
                         templates: Dict[str, Dict[str, np.ndarray]],
                         bands: List[str] = LSST_BANDS) -> pd.DataFrame:
    """
    Extract DTW features for multiple objects.

    Args:
        lightcurves: DataFrame with lightcurve data
        object_ids: List of object IDs to process
        templates: Pre-computed templates from create_templates()
        bands: List of bands to use

    Returns:
        DataFrame with DTW features
    """
    # Pre-group for efficiency
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if (i + 1) % 500 == 0:
            print(f"    DTW Features: {i+1}/{len(object_ids)} objects processed")

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue

        features = extract_dtw_features_single(obj_lc, templates, bands)
        features['object_id'] = obj_id
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

    print("\nCreating templates from training data...")
    templates = create_templates(
        data['train_lc'],
        data['train_meta']
    )

    print("\nExtracting DTW features for first 50 objects...")
    sample_ids = data['train_meta']['object_id'].head(50).tolist()
    dtw_features = extract_dtw_features(
        data['train_lc'],
        sample_ids,
        templates
    )

    print(f"\nExtracted {len(dtw_features.columns)-1} DTW features")
    print("\nFeature columns:")
    print([c for c in dtw_features.columns if c != 'object_id'])
    print("\nSample values:")
    print(dtw_features[['object_id', 'g_dtw_tde', 'g_dtw_ratio', 'dtw_ratio_mean']].head(10))
