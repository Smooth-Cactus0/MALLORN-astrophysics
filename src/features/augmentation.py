"""
Data Augmentation for TDE Lightcurves

Physics-justified augmentations that create realistic synthetic TDE lightcurves:
1. Flux scaling - simulates different distances/luminosities
2. Time stretching - simulates different redshifts
3. Noise injection - simulates measurement uncertainty
4. Observation dropout - simulates different cadences
5. Time shifting - simulates different discovery epochs
6. Mixup - interpolates between two TDEs

All augmentations preserve the fundamental TDE physics (shape, color evolution).
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LightcurveAugmenter:
    """
    Augmentation pipeline for astronomical lightcurves.

    Designed specifically for TDE augmentation to address class imbalance.
    Each augmentation is physics-motivated and preserves TDE characteristics.
    """

    def __init__(
        self,
        flux_scale_range: Tuple[float, float] = (0.5, 2.0),
        time_stretch_range: Tuple[float, float] = (0.8, 1.2),
        noise_scale_range: Tuple[float, float] = (0.5, 1.5),
        dropout_range: Tuple[float, float] = (0.1, 0.3),
        random_state: int = 42
    ):
        """
        Args:
            flux_scale_range: Min/max flux scaling factors
            time_stretch_range: Min/max time stretch factors
            noise_scale_range: Min/max noise injection scaling
            dropout_range: Min/max fraction of observations to drop
            random_state: Random seed for reproducibility
        """
        self.flux_scale_range = flux_scale_range
        self.time_stretch_range = time_stretch_range
        self.noise_scale_range = noise_scale_range
        self.dropout_range = dropout_range
        self.rng = np.random.RandomState(random_state)

    def flux_scaling(self, lc: pd.DataFrame, scale: float) -> pd.DataFrame:
        """
        Scale flux values by a constant factor.

        Physics: Simulates observing the same TDE at different distances
        or with different intrinsic luminosities.

        Preserves: Lightcurve shape, color ratios, temporal evolution
        """
        aug = lc.copy()
        aug['Flux'] = aug['Flux'] * scale
        aug['Flux_err'] = aug['Flux_err'] * scale
        return aug

    def time_stretching(self, lc: pd.DataFrame, stretch: float) -> pd.DataFrame:
        """
        Stretch/compress time axis.

        Physics: Simulates cosmological time dilation at different redshifts.
        A TDE at z=0.5 appears 1.5x slower than at z=0.

        Preserves: Lightcurve shape (just stretched), color evolution pattern
        """
        aug = lc.copy()
        t_min = aug['Time (MJD)'].min()
        aug['Time (MJD)'] = t_min + (aug['Time (MJD)'] - t_min) * stretch
        return aug

    def noise_injection(self, lc: pd.DataFrame, scale: float) -> pd.DataFrame:
        """
        Add Gaussian noise proportional to flux errors.

        Physics: Simulates different observing conditions (seeing, sky brightness).

        Preserves: Overall lightcurve shape, but adds realistic scatter
        """
        aug = lc.copy()
        noise = self.rng.normal(0, aug['Flux_err'].values * scale)
        aug['Flux'] = aug['Flux'] + noise
        return aug

    def observation_dropout(self, lc: pd.DataFrame, dropout_frac: float) -> pd.DataFrame:
        """
        Randomly remove a fraction of observations.

        Physics: Simulates different survey cadences, weather losses,
        or telescope time allocation.

        Preserves: Underlying lightcurve (just sparser sampling)
        """
        n_keep = max(5, int(len(lc) * (1 - dropout_frac)))  # Keep at least 5 points
        keep_idx = self.rng.choice(len(lc), size=n_keep, replace=False)
        keep_idx = np.sort(keep_idx)
        return lc.iloc[keep_idx].reset_index(drop=True)

    def time_shift(self, lc: pd.DataFrame, shift_days: float) -> pd.DataFrame:
        """
        Shift all times by a constant offset.

        Physics: Simulates different discovery epochs (the TDE happened
        at a different time but we're still observing the same event).

        Preserves: Everything (this is mainly for data diversity)
        """
        aug = lc.copy()
        aug['Time (MJD)'] = aug['Time (MJD)'] + shift_days
        return aug

    def band_specific_noise(self, lc: pd.DataFrame) -> pd.DataFrame:
        """
        Add band-specific noise patterns.

        Physics: Different bands have different noise characteristics
        (u-band is noisier than r-band due to lower throughput).
        """
        aug = lc.copy()
        band_noise_scale = {'u': 1.5, 'g': 1.0, 'r': 0.8, 'i': 0.9, 'z': 1.1, 'y': 1.3}

        for band, scale in band_noise_scale.items():
            mask = aug['Filter'] == band
            if mask.any():
                noise = self.rng.normal(0, aug.loc[mask, 'Flux_err'].values * scale * 0.3)
                aug.loc[mask, 'Flux'] = aug.loc[mask, 'Flux'] + noise

        return aug

    def augment_single(self, lc: pd.DataFrame, n_augmentations: int = 10) -> List[pd.DataFrame]:
        """
        Generate multiple augmented versions of a single lightcurve.

        Args:
            lc: Original lightcurve DataFrame
            n_augmentations: Number of augmented versions to create

        Returns:
            List of augmented DataFrames
        """
        augmented = []

        for i in range(n_augmentations):
            aug = lc.copy()

            # Apply random combination of augmentations

            # 1. Flux scaling (always apply)
            scale = self.rng.uniform(*self.flux_scale_range)
            aug = self.flux_scaling(aug, scale)

            # 2. Time stretching (80% chance)
            if self.rng.random() < 0.8:
                stretch = self.rng.uniform(*self.time_stretch_range)
                aug = self.time_stretching(aug, stretch)

            # 3. Noise injection (70% chance)
            if self.rng.random() < 0.7:
                noise_scale = self.rng.uniform(*self.noise_scale_range)
                aug = self.noise_injection(aug, noise_scale)

            # 4. Observation dropout (50% chance)
            if self.rng.random() < 0.5:
                dropout = self.rng.uniform(*self.dropout_range)
                aug = self.observation_dropout(aug, dropout)

            # 5. Time shift (30% chance)
            if self.rng.random() < 0.3:
                shift = self.rng.uniform(-100, 100)
                aug = self.time_shift(aug, shift)

            # 6. Band-specific noise (40% chance)
            if self.rng.random() < 0.4:
                aug = self.band_specific_noise(aug)

            augmented.append(aug)

        return augmented


def mixup_lightcurves(
    lc1: pd.DataFrame,
    lc2: pd.DataFrame,
    alpha: float = 0.5,
    rng: np.random.RandomState = None
) -> pd.DataFrame:
    """
    Create a mixup of two lightcurves.

    Interpolates flux values between two TDEs at matching time/band points.
    This creates synthetic TDEs that are "between" two real ones.

    Args:
        lc1, lc2: Two lightcurve DataFrames
        alpha: Mixing coefficient (0.5 = equal mix)
        rng: Random state

    Returns:
        Mixed lightcurve DataFrame
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Use lc1 as base structure
    mixed = lc1.copy()

    # For each observation in lc1, find closest in lc2 and interpolate
    for band in lc1['Filter'].unique():
        mask1 = mixed['Filter'] == band
        lc2_band = lc2[lc2['Filter'] == band]

        if len(lc2_band) == 0:
            continue

        for idx in mixed[mask1].index:
            t1 = mixed.loc[idx, 'Time (MJD)']

            # Find closest time in lc2
            dt = np.abs(lc2_band['Time (MJD)'].values - t1)
            closest_idx = dt.argmin()

            if dt[closest_idx] < 30:  # Only mix if within 30 days
                f1 = mixed.loc[idx, 'Flux']
                f2 = lc2_band.iloc[closest_idx]['Flux']

                # Interpolate
                mixed.loc[idx, 'Flux'] = alpha * f1 + (1 - alpha) * f2

    return mixed


def augment_tde_dataset(
    lightcurves: pd.DataFrame,
    metadata: pd.DataFrame,
    augmentations_per_tde: int = 10,
    include_mixup: bool = True,
    mixup_per_tde: int = 2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Augment all TDE lightcurves in the dataset.

    Args:
        lightcurves: Full lightcurve DataFrame
        metadata: Metadata with target labels
        augmentations_per_tde: Number of augmented versions per TDE
        include_mixup: Whether to include mixup augmentations
        mixup_per_tde: Number of mixup versions per TDE
        random_state: Random seed

    Returns:
        (augmented_lightcurves, augmented_metadata)
    """
    rng = np.random.RandomState(random_state)
    augmenter = LightcurveAugmenter(random_state=random_state)

    # Get TDE object IDs
    tde_ids = metadata[metadata['target'] == 1]['object_id'].tolist()
    print(f"  Found {len(tde_ids)} TDE objects to augment")

    # Group lightcurves by object
    grouped = {obj_id: group.copy() for obj_id, group in lightcurves.groupby('object_id')}

    augmented_lcs = []
    augmented_meta = []

    # Generate augmentations
    for i, obj_id in enumerate(tde_ids):
        if (i + 1) % 20 == 0:
            print(f"    Augmenting TDE {i+1}/{len(tde_ids)}")

        original_lc = grouped.get(obj_id)
        if original_lc is None or len(original_lc) < 5:
            continue

        original_meta = metadata[metadata['object_id'] == obj_id].iloc[0]

        # Standard augmentations
        aug_lcs = augmenter.augment_single(original_lc, n_augmentations=augmentations_per_tde)

        for j, aug_lc in enumerate(aug_lcs):
            new_id = f"{obj_id}_aug{j}"
            aug_lc['object_id'] = new_id
            augmented_lcs.append(aug_lc)

            # Create metadata entry
            new_meta = original_meta.copy()
            new_meta['object_id'] = new_id
            augmented_meta.append(new_meta)

        # Mixup augmentations
        if include_mixup and len(tde_ids) > 1:
            for k in range(mixup_per_tde):
                # Pick random other TDE
                other_id = rng.choice([x for x in tde_ids if x != obj_id])
                other_lc = grouped.get(other_id)

                if other_lc is not None and len(other_lc) >= 5:
                    alpha = rng.uniform(0.3, 0.7)
                    mixed_lc = mixup_lightcurves(original_lc, other_lc, alpha, rng)

                    new_id = f"{obj_id}_mix{k}"
                    mixed_lc['object_id'] = new_id
                    augmented_lcs.append(mixed_lc)

                    new_meta = original_meta.copy()
                    new_meta['object_id'] = new_id
                    augmented_meta.append(new_meta)

    # Combine into DataFrames
    if augmented_lcs:
        aug_lc_df = pd.concat(augmented_lcs, ignore_index=True)
        aug_meta_df = pd.DataFrame(augmented_meta)

        print(f"  Generated {len(aug_meta_df)} augmented TDE samples")
        print(f"  Total lightcurve observations: {len(aug_lc_df)}")

        return aug_lc_df, aug_meta_df
    else:
        return pd.DataFrame(), pd.DataFrame()


def augment_all_samples(
    lightcurves: pd.DataFrame,
    metadata: pd.DataFrame,
    augmentations_per_sample: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Augment ALL lightcurves in the dataset (both TDE and non-TDE).

    This preserves the original class distribution while increasing
    the total dataset size, giving the model more examples to learn from.

    Args:
        lightcurves: Full lightcurve DataFrame
        metadata: Metadata with target labels
        augmentations_per_sample: Number of augmented versions per sample
        random_state: Random seed

    Returns:
        (augmented_lightcurves, augmented_metadata)
    """
    augmenter = LightcurveAugmenter(random_state=random_state)

    all_ids = metadata['object_id'].tolist()
    n_tde = (metadata['target'] == 1).sum()
    n_non_tde = (metadata['target'] == 0).sum()

    print(f"  Augmenting ALL {len(all_ids)} objects ({n_tde} TDE, {n_non_tde} non-TDE)")
    print(f"  Target: {augmentations_per_sample}x augmentation -> ~{len(all_ids) * augmentations_per_sample} new samples")

    # Group lightcurves by object
    grouped = {obj_id: group.copy() for obj_id, group in lightcurves.groupby('object_id')}

    augmented_lcs = []
    augmented_meta = []

    # Generate augmentations for all samples
    for i, obj_id in enumerate(all_ids):
        if (i + 1) % 500 == 0:
            print(f"    Augmenting sample {i+1}/{len(all_ids)}")

        original_lc = grouped.get(obj_id)
        if original_lc is None or len(original_lc) < 5:
            continue

        original_meta = metadata[metadata['object_id'] == obj_id].iloc[0]

        # Generate augmented versions
        aug_lcs = augmenter.augment_single(original_lc, n_augmentations=augmentations_per_sample)

        for j, aug_lc in enumerate(aug_lcs):
            new_id = f"{obj_id}_aug{j}"
            aug_lc['object_id'] = new_id
            augmented_lcs.append(aug_lc)

            # Create metadata entry (preserves original target label)
            new_meta = original_meta.copy()
            new_meta['object_id'] = new_id
            augmented_meta.append(new_meta)

    # Combine into DataFrames
    if augmented_lcs:
        aug_lc_df = pd.concat(augmented_lcs, ignore_index=True)
        aug_meta_df = pd.DataFrame(augmented_meta)

        # Verify class distribution is preserved
        aug_tde = (aug_meta_df['target'] == 1).sum()
        aug_non_tde = (aug_meta_df['target'] == 0).sum()
        aug_tde_pct = aug_tde / len(aug_meta_df) * 100
        orig_tde_pct = n_tde / len(all_ids) * 100

        print(f"  Generated {len(aug_meta_df)} augmented samples")
        print(f"  Class distribution: {aug_tde_pct:.1f}% TDE (original: {orig_tde_pct:.1f}%)")
        print(f"  Total lightcurve observations: {len(aug_lc_df)}")

        return aug_lc_df, aug_meta_df
    else:
        return pd.DataFrame(), pd.DataFrame()


if __name__ == "__main__":
    # Test the augmentation module
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_all_data

    print("Loading data...")
    data = load_all_data()

    print("\nTesting augmentation pipeline...")

    # Test on small sample
    sample_meta = data['train_meta'][data['train_meta']['target'] == 1].head(5)
    sample_ids = sample_meta['object_id'].tolist()
    sample_lc = data['train_lc'][data['train_lc']['object_id'].isin(sample_ids)]

    aug_lc, aug_meta = augment_tde_dataset(
        sample_lc,
        sample_meta,
        augmentations_per_tde=5,
        include_mixup=True,
        mixup_per_tde=2
    )

    print(f"\nOriginal TDEs: {len(sample_meta)}")
    print(f"Augmented TDEs: {len(aug_meta)}")
    print(f"Augmentation factor: {len(aug_meta) / len(sample_meta):.1f}x")
