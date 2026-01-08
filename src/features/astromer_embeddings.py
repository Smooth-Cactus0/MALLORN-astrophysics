"""
ASTROMER Embedding Feature Extraction for MALLORN classification.

ASTROMER is a transformer pre-trained on millions of MACHO light curves.
We use it to generate learned representations that capture temporal patterns
beyond what hand-crafted features can express.

Strategy:
- Extract embeddings for each band separately (ASTROMER is single-band)
- Use 'macho' pre-trained weights
- Pool embeddings (mean, max) to create fixed-size feature vectors
- Feed embeddings to GBM alongside engineered features

Reference: https://github.com/astromer-science/python-library
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings('ignore')

LSST_BANDS = ["g", "r", "i", "z"]  # Focus on main bands


def load_astromer_model(pretrained: str = 'macho'):
    """
    Load pre-trained ASTROMER encoder.

    Args:
        pretrained: Which pre-trained weights to use ('macho' or 'atlas')

    Returns:
        Loaded ASTROMER SingleBandEncoder model
    """
    try:
        from ASTROMER.models import SingleBandEncoder
        model = SingleBandEncoder()
        model = model.from_pretraining(pretrained)
        return model
    except Exception as e:
        print(f"Warning: Could not load ASTROMER model: {e}")
        return None


def prepare_lightcurve_for_astromer(
    times: np.ndarray,
    fluxes: np.ndarray,
    errors: np.ndarray,
    max_length: int = 200
) -> Optional[np.ndarray]:
    """
    Prepare a single-band lightcurve for ASTROMER input.

    ASTROMER expects: Lx3 array with [time, magnitude, magnitude_error]

    Args:
        times: Observation times (MJD)
        fluxes: Flux values
        errors: Flux errors
        max_length: Maximum sequence length

    Returns:
        Array of shape (L, 3) or None if invalid
    """
    if len(times) < 5:
        return None

    # Remove invalid points
    valid = (fluxes > 0) & (errors > 0) & ~np.isnan(fluxes) & ~np.isnan(errors)
    if np.sum(valid) < 5:
        return None

    times = times[valid]
    fluxes = fluxes[valid]
    errors = errors[valid]

    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    fluxes = fluxes[sort_idx]
    errors = errors[sort_idx]

    # Convert flux to magnitude
    # mag = -2.5 * log10(flux) + zeropoint
    # We use a relative magnitude (no absolute calibration)
    with np.errstate(divide='ignore', invalid='ignore'):
        mags = -2.5 * np.log10(fluxes)
        # Magnitude error: dm = 2.5 / ln(10) * (df/f)
        mag_errors = 2.5 / np.log(10) * (errors / fluxes)

    # Handle any remaining infinities
    valid = np.isfinite(mags) & np.isfinite(mag_errors)
    if np.sum(valid) < 5:
        return None

    times = times[valid]
    mags = mags[valid]
    mag_errors = mag_errors[valid]

    # Normalize time to start at 0
    times = times - times.min()

    # Truncate if too long
    if len(times) > max_length:
        times = times[:max_length]
        mags = mags[:max_length]
        mag_errors = mag_errors[:max_length]

    # Create ASTROMER input format
    lc_array = np.column_stack([times, mags, mag_errors])

    return lc_array


def extract_astromer_embedding(
    model,
    lightcurve: np.ndarray
) -> Optional[np.ndarray]:
    """
    Extract embedding from ASTROMER for a single lightcurve.

    Args:
        model: Loaded ASTROMER model
        lightcurve: Prepared lightcurve array (L, 3)

    Returns:
        Embedding vector or None
    """
    if model is None or lightcurve is None:
        return None

    try:
        # ASTROMER expects list of lightcurves
        lc_list = [lightcurve]

        # Get embeddings using the encoder
        # The model.encode() method returns embeddings
        embeddings = model.encode(lc_list)

        if embeddings is not None and len(embeddings) > 0:
            return embeddings[0]  # Return first (and only) embedding
        return None
    except Exception as e:
        return None


def pool_embedding(embedding: np.ndarray) -> Dict[str, float]:
    """
    Pool a sequence embedding into fixed-size features.

    Args:
        embedding: Embedding array, possibly 2D (seq_len, hidden_dim)

    Returns:
        Dictionary with pooled features
    """
    if embedding is None:
        return {}

    # Handle different shapes
    if embedding.ndim == 1:
        vec = embedding
    elif embedding.ndim == 2:
        # Pool across sequence dimension
        vec_mean = np.mean(embedding, axis=0)
        vec_max = np.max(embedding, axis=0)
        vec = np.concatenate([vec_mean, vec_max])
    else:
        return {}

    # Create features from embedding dimensions
    features = {}

    # Use first N dimensions as features (embeddings can be large)
    n_dims = min(32, len(vec))  # Limit to 32 dimensions per band
    for i in range(n_dims):
        features[f'emb_{i}'] = float(vec[i])

    # Add summary statistics
    features['emb_mean'] = float(np.mean(vec))
    features['emb_std'] = float(np.std(vec))
    features['emb_max'] = float(np.max(vec))
    features['emb_min'] = float(np.min(vec))

    return features


def extract_astromer_features_single(
    obj_lc: pd.DataFrame,
    model
) -> Dict[str, float]:
    """
    Extract ASTROMER-based features for a single object.

    Args:
        obj_lc: DataFrame with lightcurve data
        model: Loaded ASTROMER model

    Returns:
        Dictionary of ASTROMER features
    """
    features = {}

    if model is None:
        # Return empty features if model not available
        for band in LSST_BANDS:
            for i in range(36):  # 32 dims + 4 stats
                features[f'{band}_astromer_{i}'] = np.nan
        return features

    for band in LSST_BANDS:
        band_lc = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_lc) >= 5:
            times = band_lc['Time (MJD)'].values
            fluxes = band_lc['Flux'].values
            errors = band_lc['Flux_err'].values

            # Prepare lightcurve
            lc_array = prepare_lightcurve_for_astromer(times, fluxes, errors)

            if lc_array is not None:
                # Get embedding
                embedding = extract_astromer_embedding(model, lc_array)

                # Pool to features
                emb_features = pool_embedding(embedding)

                # Add band prefix
                for key, val in emb_features.items():
                    features[f'{band}_astromer_{key}'] = val
            else:
                # Fill with NaN
                for i in range(32):
                    features[f'{band}_astromer_emb_{i}'] = np.nan
                for stat in ['mean', 'std', 'max', 'min']:
                    features[f'{band}_astromer_emb_{stat}'] = np.nan
        else:
            # Fill with NaN
            for i in range(32):
                features[f'{band}_astromer_emb_{i}'] = np.nan
            for stat in ['mean', 'std', 'max', 'min']:
                features[f'{band}_astromer_emb_{stat}'] = np.nan

    # Cross-band embedding similarities
    # (useful for detecting achromatic vs chromatic variability)
    for b1, b2 in [('g', 'r'), ('r', 'i')]:
        means = []
        for b in [b1, b2]:
            m = features.get(f'{b}_astromer_emb_mean', np.nan)
            if not np.isnan(m):
                means.append(m)

        if len(means) == 2:
            features[f'astromer_{b1}{b2}_mean_ratio'] = means[0] / (means[1] + 1e-6)
        else:
            features[f'astromer_{b1}{b2}_mean_ratio'] = np.nan

    return features


def extract_astromer_features(
    lightcurves: pd.DataFrame,
    metadata: pd.DataFrame,
    object_ids: Optional[List[str]] = None,
    pretrained: str = 'macho',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract ASTROMER-based features for multiple objects.

    Args:
        lightcurves: DataFrame with lightcurve data
        metadata: DataFrame with object metadata
        object_ids: Optional list of object IDs to process
        pretrained: Which pre-trained weights to use
        verbose: Whether to print progress

    Returns:
        DataFrame with ASTROMER features for each object
    """
    if object_ids is None:
        object_ids = lightcurves['object_id'].unique()

    # Load model once
    if verbose:
        print(f"    Loading ASTROMER model ({pretrained})...")
    model = load_astromer_model(pretrained)

    if model is None:
        print("    Warning: ASTROMER model could not be loaded, returning empty features")
        return pd.DataFrame({'object_id': object_ids})

    # Pre-group by object_id
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if verbose and (i + 1) % 50 == 0:
            print(f"    ASTROMER: {i+1}/{len(object_ids)} objects processed")

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue

        features = extract_astromer_features_single(obj_lc, model)
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

    print("\nExtracting ASTROMER features for first 10 objects...")
    sample_ids = data['train_meta']['object_id'].head(10).tolist()

    astromer_features = extract_astromer_features(
        data['train_lc'],
        data['train_meta'],
        sample_ids
    )

    print(f"\nExtracted {len(astromer_features.columns)-1} ASTROMER features")
    print("\nFeature columns (first 20):")
    cols = [c for c in astromer_features.columns if c != 'object_id']
    print(cols[:20])
    print(f"... and {len(cols)-20} more" if len(cols) > 20 else "")
