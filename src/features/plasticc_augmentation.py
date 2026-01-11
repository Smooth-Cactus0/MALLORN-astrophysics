"""
PLAsTiCC-Style Data Augmentation for MALLORN

Based on winning techniques from PLAsTiCC Kaggle competition:

1. Kyle Boone (1st place - Avocado):
   - Gaussian Process interpolation for smooth lightcurves
   - Redshift augmentation: generate high-z versions from low-z objects
     - Time dilation: t_obs = t_rest * (1+z)
     - Flux scaling: F_obs = F_rest / (d_L(z))^2
   - Match TEST distribution, not just balance classes

2. Major Tom (3rd place):
   - Per-band skew: multiply each band by random (1+delta)
   - Golden feature: (max-min) * luminosity_distance^2

Key insight: The gap between train/test distributions was the main challenge.
Augment to make training data LOOK LIKE test data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')

# Cosmology constants (flat LambdaCDM)
H0 = 70.0  # km/s/Mpc
OMEGA_M = 0.3
OMEGA_L = 0.7
C_KMS = 299792.458  # km/s


def luminosity_distance(z: float) -> float:
    """
    Calculate luminosity distance for a given redshift.
    Uses simple numerical integration for flat LambdaCDM.

    Returns distance in Mpc.
    """
    if z <= 0:
        return 0.0

    # Comoving distance via numerical integration
    n_steps = 100
    z_arr = np.linspace(0, z, n_steps)
    dz = z / n_steps

    # E(z) = sqrt(Omega_m(1+z)^3 + Omega_L)
    E_z = np.sqrt(OMEGA_M * (1 + z_arr)**3 + OMEGA_L)

    # Comoving distance: d_C = c/H0 * integral(dz/E(z))
    d_C = (C_KMS / H0) * np.sum(1.0 / E_z) * dz

    # Luminosity distance: d_L = (1+z) * d_C
    d_L = (1 + z) * d_C

    return d_L


def redshift_augment_lightcurve(
    lc: pd.DataFrame,
    z_orig: float,
    z_new: float,
    rng: np.random.RandomState = None
) -> pd.DataFrame:
    """
    Augment a lightcurve by simulating observation at a different redshift.

    Physics:
    1. Time dilation: t_new = t_orig * (1+z_new) / (1+z_orig)
    2. Flux scaling: F_new = F_orig * (d_L(z_orig) / d_L(z_new))^2
    3. Wavelength shift: affects which rest-frame wavelengths we observe
       (simplified - we keep same observed bands)

    Args:
        lc: Original lightcurve DataFrame
        z_orig: Original redshift
        z_new: Target redshift
        rng: Random state

    Returns:
        Augmented lightcurve at new redshift
    """
    if rng is None:
        rng = np.random.RandomState(42)

    aug = lc.copy()

    # Time dilation factor
    time_factor = (1 + z_new) / (1 + z_orig)

    # Luminosity distance ratio (flux scales as 1/d_L^2)
    d_L_orig = luminosity_distance(z_orig) + 1e-10  # Avoid div by zero
    d_L_new = luminosity_distance(z_new) + 1e-10
    flux_factor = (d_L_orig / d_L_new) ** 2

    # Apply time dilation (stretch around first observation)
    t_min = aug['Time (MJD)'].min()
    aug['Time (MJD)'] = t_min + (aug['Time (MJD)'] - t_min) * time_factor

    # Apply flux scaling
    aug['Flux'] = aug['Flux'] * flux_factor
    aug['Flux_err'] = aug['Flux_err'] * flux_factor

    # Add noise proportional to reduced S/N at higher z
    if z_new > z_orig:
        # Objects at higher z have lower S/N
        sn_reduction = np.sqrt(flux_factor)  # S/N scales with sqrt(flux)
        if sn_reduction < 1:
            extra_noise = aug['Flux_err'] * (1/sn_reduction - 1)
            noise = rng.normal(0, extra_noise.values)
            aug['Flux'] = aug['Flux'] + noise

    return aug


def per_band_skew(
    lc: pd.DataFrame,
    skew_range: Tuple[float, float] = (-0.2, 0.2),
    rng: np.random.RandomState = None
) -> pd.DataFrame:
    """
    Apply random multiplicative skew to each band (3rd place technique).

    This simulates calibration uncertainties and atmospheric variations.

    Args:
        lc: Lightcurve DataFrame
        skew_range: Range for delta in (1+delta) multiplier
        rng: Random state

    Returns:
        Skewed lightcurve
    """
    if rng is None:
        rng = np.random.RandomState(42)

    aug = lc.copy()

    for band in aug['Filter'].unique():
        mask = aug['Filter'] == band
        delta = rng.uniform(*skew_range)
        aug.loc[mask, 'Flux'] = aug.loc[mask, 'Flux'] * (1 + delta)
        aug.loc[mask, 'Flux_err'] = aug.loc[mask, 'Flux_err'] * (1 + delta)

    return aug


def quality_degradation(
    lc: pd.DataFrame,
    drop_frac: float = 0.2,
    noise_boost: float = 1.5,
    rng: np.random.RandomState = None
) -> pd.DataFrame:
    """
    Degrade lightcurve quality to match fainter/more distant objects.

    This helps match training to test distribution.

    Args:
        lc: Lightcurve DataFrame
        drop_frac: Fraction of observations to drop
        noise_boost: Factor to multiply flux errors by
        rng: Random state

    Returns:
        Degraded lightcurve
    """
    if rng is None:
        rng = np.random.RandomState(42)

    aug = lc.copy()

    # Drop observations
    n_keep = max(5, int(len(aug) * (1 - drop_frac)))
    keep_idx = rng.choice(len(aug), size=n_keep, replace=False)
    aug = aug.iloc[np.sort(keep_idx)].reset_index(drop=True)

    # Boost noise
    aug['Flux_err'] = aug['Flux_err'] * noise_boost
    noise = rng.normal(0, aug['Flux_err'].values * (noise_boost - 1) / noise_boost)
    aug['Flux'] = aug['Flux'] + noise

    return aug


class PLAsTiCCAugmenter:
    """
    PLAsTiCC-style augmentation pipeline.

    Key principles:
    1. Augment to match TEST distribution (not just balance classes)
    2. Use physics-based redshift augmentation
    3. Preserve class-specific lightcurve shapes
    """

    def __init__(
        self,
        z_augment_range: Tuple[float, float] = (0.1, 1.5),
        skew_range: Tuple[float, float] = (-0.2, 0.2),
        quality_degradation_prob: float = 0.5,
        random_state: int = 42
    ):
        """
        Args:
            z_augment_range: Range of redshifts to augment to
            skew_range: Range for per-band skew
            quality_degradation_prob: Probability of applying quality degradation
            random_state: Random seed
        """
        self.z_range = z_augment_range
        self.skew_range = skew_range
        self.degrade_prob = quality_degradation_prob
        self.rng = np.random.RandomState(random_state)

    def augment_single(
        self,
        lc: pd.DataFrame,
        z_orig: float,
        n_augmentations: int = 10,
        target_z_distribution: Optional[np.ndarray] = None
    ) -> List[Tuple[pd.DataFrame, float]]:
        """
        Generate multiple augmented versions of a single lightcurve.

        Args:
            lc: Original lightcurve
            z_orig: Original redshift
            n_augmentations: Number of augmented versions
            target_z_distribution: If provided, sample z from this distribution

        Returns:
            List of (augmented_lc, new_z) tuples
        """
        augmented = []

        for i in range(n_augmentations):
            # Sample new redshift
            if target_z_distribution is not None:
                z_new = self.rng.choice(target_z_distribution)
            else:
                z_new = self.rng.uniform(*self.z_range)

            # Apply redshift augmentation
            aug = redshift_augment_lightcurve(lc, z_orig, z_new, self.rng)

            # Apply per-band skew (always)
            aug = per_band_skew(aug, self.skew_range, self.rng)

            # Apply quality degradation (probabilistic)
            if self.rng.random() < self.degrade_prob:
                drop_frac = self.rng.uniform(0.1, 0.3)
                noise_boost = self.rng.uniform(1.2, 2.0)
                aug = quality_degradation(aug, drop_frac, noise_boost, self.rng)

            augmented.append((aug, z_new))

        return augmented


def analyze_distribution_shift(
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame
) -> Dict:
    """
    Analyze the distribution shift between train and test sets.

    This helps us understand what augmentations are needed.
    """
    results = {}

    # Redshift distribution
    train_z = train_meta['Z'].dropna()
    test_z = test_meta['Z'].dropna()

    results['train_z_mean'] = train_z.mean()
    results['train_z_std'] = train_z.std()
    results['train_z_median'] = train_z.median()

    results['test_z_mean'] = test_z.mean()
    results['test_z_std'] = test_z.std()
    results['test_z_median'] = test_z.median()

    results['z_shift'] = results['test_z_mean'] - results['train_z_mean']

    # EBV (extinction) distribution
    train_ebv = train_meta['EBV'].dropna()
    test_ebv = test_meta['EBV'].dropna()

    results['train_ebv_mean'] = train_ebv.mean()
    results['test_ebv_mean'] = test_ebv.mean()

    return results


def augment_for_test_distribution(
    train_lc: pd.DataFrame,
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    augmentations_per_sample: int = 5,
    augment_all: bool = False,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Augment training data to match test distribution.

    This is the KEY insight from PLAsTiCC: don't just balance classes,
    make training data LOOK LIKE test data.

    Args:
        train_lc: Training lightcurves
        train_meta: Training metadata
        test_meta: Test metadata (for distribution matching)
        augmentations_per_sample: Number of augmented versions per sample
        augment_all: If True, augment all samples. If False, only TDEs.
        random_state: Random seed

    Returns:
        (augmented_lc, augmented_meta)
    """
    rng = np.random.RandomState(random_state)

    # Analyze distribution shift
    print("   Analyzing train/test distribution shift...", flush=True)
    shift = analyze_distribution_shift(train_meta, test_meta)
    print(f"      Train z: {shift['train_z_mean']:.3f} +/- {shift['train_z_std']:.3f}", flush=True)
    print(f"      Test z:  {shift['test_z_mean']:.3f} +/- {shift['test_z_std']:.3f}", flush=True)
    print(f"      Shift: {shift['z_shift']:+.3f}", flush=True)

    # Get test redshift distribution for sampling
    test_z = test_meta['Z'].dropna().values

    # Select samples to augment
    if augment_all:
        ids_to_augment = train_meta['object_id'].tolist()
        print(f"   Augmenting ALL {len(ids_to_augment)} samples", flush=True)
    else:
        ids_to_augment = train_meta[train_meta['target'] == 1]['object_id'].tolist()
        print(f"   Augmenting {len(ids_to_augment)} TDE samples", flush=True)

    # Group lightcurves
    grouped = {oid: group.copy() for oid, group in train_lc.groupby('object_id')}

    # Initialize augmenter
    augmenter = PLAsTiCCAugmenter(
        z_augment_range=(shift['test_z_mean'] - shift['test_z_std'],
                         shift['test_z_mean'] + shift['test_z_std']),
        random_state=random_state
    )

    augmented_lcs = []
    augmented_meta = []

    for i, oid in enumerate(ids_to_augment):
        if (i + 1) % 100 == 0:
            print(f"      Augmenting {i+1}/{len(ids_to_augment)}...", flush=True)

        lc = grouped.get(oid)
        if lc is None or len(lc) < 5:
            continue

        meta_row = train_meta[train_meta['object_id'] == oid].iloc[0]
        z_orig = meta_row['Z'] if pd.notna(meta_row['Z']) else 0.5

        # Generate augmentations targeting test distribution
        aug_results = augmenter.augment_single(
            lc, z_orig,
            n_augmentations=augmentations_per_sample,
            target_z_distribution=test_z
        )

        for j, (aug_lc, z_new) in enumerate(aug_results):
            new_id = f"{oid}_zaug{j}"
            aug_lc = aug_lc.copy()
            aug_lc['object_id'] = new_id
            augmented_lcs.append(aug_lc)

            new_meta = meta_row.copy()
            new_meta['object_id'] = new_id
            new_meta['Z'] = z_new  # Update redshift
            augmented_meta.append(new_meta)

    if augmented_lcs:
        aug_lc_df = pd.concat(augmented_lcs, ignore_index=True)
        aug_meta_df = pd.DataFrame(augmented_meta)

        print(f"   Generated {len(aug_meta_df)} augmented samples", flush=True)

        # Verify z distribution matching
        aug_z = aug_meta_df['Z'].dropna()
        print(f"   Augmented z distribution: {aug_z.mean():.3f} +/- {aug_z.std():.3f}", flush=True)

        return aug_lc_df, aug_meta_df

    return pd.DataFrame(), pd.DataFrame()


if __name__ == "__main__":
    # Test the module
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_all_data

    print("Loading data...")
    data = load_all_data()

    print("\nAnalyzing distribution shift...")
    shift = analyze_distribution_shift(data['train_meta'], data['test_meta'])
    for k, v in shift.items():
        print(f"   {k}: {v:.4f}")

    print("\nTesting PLAsTiCC-style augmentation...")

    # Test on small sample
    sample_meta = data['train_meta'][data['train_meta']['target'] == 1].head(5)
    sample_ids = sample_meta['object_id'].tolist()
    sample_lc = data['train_lc'][data['train_lc']['object_id'].isin(sample_ids)]

    aug_lc, aug_meta = augment_for_test_distribution(
        sample_lc,
        sample_meta,
        data['test_meta'],
        augmentations_per_sample=3
    )

    print(f"\nOriginal samples: {len(sample_meta)}")
    print(f"Augmented samples: {len(aug_meta)}")
