"""
Gaussian Process-Based Data Augmentation

Based on PLAsTiCC 1st place (Kyle Boone):
- Time shifting: Shift lightcurves by ±20 days
- Random observation removal: Drop 10-30% of observations randomly
- S/N degradation: Degrade low-z objects to simulate high-z
- Expand 3,043 → 12,000+ training samples

Physics basis:
- Time shifting: Observation epoch shouldn't matter (translation invariance)
- Observation removal: Models must work with sparse cadence
- S/N degradation: Should classify across all redshifts

Implementation:
- Use existing GP fits from multiband_gp_cache.pkl
- Generate augmented lightcurves by perturbing observations
- Extract features from augmented data
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import pickle
from pathlib import Path


def time_shift_lightcurve(lc: pd.DataFrame, shift_days: float) -> pd.DataFrame:
    """
    Shift lightcurve in time.

    Args:
        lc: Lightcurve DataFrame (must have Time (MJD))
        shift_days: Days to shift (positive = shift forward)

    Returns:
        Shifted lightcurve
    """
    lc_shifted = lc.copy()
    lc_shifted['Time (MJD)'] = lc_shifted['Time (MJD)'] + shift_days
    return lc_shifted


def remove_random_observations(lc: pd.DataFrame, drop_fraction: float,
                               random_state: int = None) -> pd.DataFrame:
    """
    Randomly remove observations from lightcurve.

    Args:
        lc: Lightcurve DataFrame
        drop_fraction: Fraction of observations to remove (0-1)
        random_state: Random seed for reproducibility

    Returns:
        Lightcurve with observations removed
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_drop = int(len(lc) * drop_fraction)
    keep_indices = np.random.choice(len(lc), len(lc) - n_drop, replace=False)

    lc_sparse = lc.iloc[keep_indices].copy()
    return lc_sparse


def degrade_signal_to_noise(lc: pd.DataFrame, noise_factor: float,
                            random_state: int = None) -> pd.DataFrame:
    """
    Degrade signal-to-noise ratio by adding noise.

    Simulates higher redshift (fainter, noisier observations).

    Args:
        lc: Lightcurve DataFrame (must have Flux, Flux_err)
        noise_factor: Factor to multiply flux error (>1 = more noise)
        random_state: Random seed for reproducibility

    Returns:
        Lightcurve with degraded S/N
    """
    if random_state is not None:
        np.random.seed(random_state)

    lc_degraded = lc.copy()

    # Add noise to flux
    additional_noise = np.random.normal(0, lc['Flux_err'] * (noise_factor - 1), len(lc))
    lc_degraded['Flux'] = lc['Flux'] + additional_noise

    # Increase flux error
    lc_degraded['Flux_err'] = lc['Flux_err'] * noise_factor

    return lc_degraded


def augment_single_object(obj_lc: pd.DataFrame, obj_id: str,
                          augmentation_type: str,
                          aug_params: Dict,
                          random_state: int = None) -> Tuple[pd.DataFrame, str]:
    """
    Apply single augmentation to object.

    Args:
        obj_lc: Lightcurve for one object
        obj_id: Object ID
        augmentation_type: Type of augmentation ('time_shift', 'sparse', 'noise')
        aug_params: Parameters for augmentation
        random_state: Random seed

    Returns:
        (augmented_lc, augmented_obj_id)
    """
    if augmentation_type == 'time_shift':
        shift_days = aug_params['shift_days']
        aug_lc = time_shift_lightcurve(obj_lc, shift_days)
        aug_id = f"{obj_id}_shift{int(shift_days):+d}"

    elif augmentation_type == 'sparse':
        drop_fraction = aug_params['drop_fraction']
        aug_lc = remove_random_observations(obj_lc, drop_fraction, random_state)
        aug_id = f"{obj_id}_sparse{int(drop_fraction*100)}"

    elif augmentation_type == 'noise':
        noise_factor = aug_params['noise_factor']
        aug_lc = degrade_signal_to_noise(obj_lc, noise_factor, random_state)
        aug_id = f"{obj_id}_noise{int(noise_factor*10)}"

    elif augmentation_type == 'combined':
        # Apply multiple augmentations
        aug_lc = obj_lc.copy()

        if 'shift_days' in aug_params and aug_params['shift_days'] != 0:
            aug_lc = time_shift_lightcurve(aug_lc, aug_params['shift_days'])

        if 'drop_fraction' in aug_params and aug_params['drop_fraction'] > 0:
            aug_lc = remove_random_observations(aug_lc, aug_params['drop_fraction'], random_state)

        if 'noise_factor' in aug_params and aug_params['noise_factor'] > 1:
            aug_lc = degrade_signal_to_noise(aug_lc, aug_params['noise_factor'], random_state)

        aug_id = f"{obj_id}_aug{random_state}"

    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")

    # Update object_id
    aug_lc = aug_lc.copy()
    aug_lc['object_id'] = aug_id

    return aug_lc, aug_id


def create_augmented_dataset(lightcurves: pd.DataFrame,
                             metadata: pd.DataFrame,
                             n_augmentations_per_object: int = 3,
                             random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create augmented training dataset following Boone's strategy.

    Args:
        lightcurves: Original lightcurve DataFrame
        metadata: Original metadata DataFrame (must have object_id, target)
        n_augmentations_per_object: Number of augmented copies per object
        random_seed: Random seed for reproducibility

    Returns:
        (augmented_lightcurves, augmented_metadata)
    """
    np.random.seed(random_seed)

    all_aug_lcs = []
    all_aug_meta = []

    # Pre-group lightcurves
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    object_ids = metadata['object_id'].tolist()
    targets = metadata['target'].values

    print(f"   Creating {n_augmentations_per_object} augmented copies per object...", flush=True)

    for i, obj_id in enumerate(object_ids):
        if (i + 1) % 500 == 0:
            print(f"    Augmentation: {i+1}/{len(object_ids)} objects processed", flush=True)

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue

        target = targets[i]

        # Create n augmented copies
        for aug_idx in range(n_augmentations_per_object):
            random_state = random_seed + i * n_augmentations_per_object + aug_idx

            # Randomly choose augmentation strategy
            strategy = np.random.choice(['time_shift', 'sparse', 'noise', 'combined'])

            if strategy == 'time_shift':
                shift_days = np.random.uniform(-20, 20)
                aug_params = {'shift_days': shift_days}

            elif strategy == 'sparse':
                drop_fraction = np.random.uniform(0.1, 0.3)
                aug_params = {'drop_fraction': drop_fraction}

            elif strategy == 'noise':
                noise_factor = np.random.uniform(1.2, 2.0)
                aug_params = {'noise_factor': noise_factor}

            elif strategy == 'combined':
                aug_params = {
                    'shift_days': np.random.uniform(-10, 10),
                    'drop_fraction': np.random.uniform(0.1, 0.2),
                    'noise_factor': np.random.uniform(1.1, 1.5)
                }

            # Apply augmentation
            aug_lc, aug_id = augment_single_object(obj_lc, obj_id, strategy,
                                                   aug_params, random_state)

            all_aug_lcs.append(aug_lc)

            # Create metadata entry
            aug_meta = {
                'object_id': aug_id,
                'target': target,
                'original_id': obj_id,
                'augmentation_type': strategy
            }
            all_aug_meta.append(aug_meta)

    # Combine augmented data
    augmented_lcs = pd.concat(all_aug_lcs, ignore_index=True)
    augmented_meta = pd.DataFrame(all_aug_meta)

    # Combine with original data
    combined_lcs = pd.concat([lightcurves, augmented_lcs], ignore_index=True)

    original_meta = metadata[['object_id', 'target']].copy()
    original_meta['original_id'] = original_meta['object_id']
    original_meta['augmentation_type'] = 'original'

    combined_meta = pd.concat([original_meta, augmented_meta], ignore_index=True)

    print(f"   Original dataset: {len(metadata)} objects", flush=True)
    print(f"   Augmented dataset: {len(combined_meta)} objects", flush=True)
    print(f"   Expansion factor: {len(combined_meta) / len(metadata):.1f}x", flush=True)

    return combined_lcs, combined_meta


if __name__ == "__main__":
    # Test augmentation
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_all_data

    print("Loading data...")
    data = load_all_data()

    # Test on small sample
    print("\nTesting augmentation on first 10 objects...")
    sample_meta = data['train_meta'].head(10)
    sample_ids = sample_meta['object_id'].tolist()
    sample_lc = data['train_lc'][data['train_lc']['object_id'].isin(sample_ids)]

    aug_lc, aug_meta = create_augmented_dataset(
        sample_lc, sample_meta,
        n_augmentations_per_object=2,
        random_seed=42
    )

    print(f"\nOriginal: {len(sample_meta)} objects")
    print(f"Augmented: {len(aug_meta)} objects")
    print(f"Expansion: {len(aug_meta) / len(sample_meta):.1f}x")

    print("\nAugmentation types:")
    print(aug_meta['augmentation_type'].value_counts())

    print("\nSample augmented object IDs:")
    print(aug_meta['object_id'].head(15).tolist())
