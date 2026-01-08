"""
PyTorch Dataset for MALLORN Lightcurve Classification

Handles variable-length multi-band lightcurves with:
- Padding to maximum sequence length
- Masking for valid timesteps
- Band encoding (learnable embeddings)
- Time delta features (time since last observation)
- Optional metadata integration (redshift, EBV)
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


# LSST band mapping for embedding
BAND_TO_IDX = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'y': 5}
N_BANDS = 6


class LightcurveDataset(Dataset):
    """
    PyTorch Dataset for astronomical lightcurves.

    Each sample contains:
    - Sequence of observations: [time, flux, flux_err, band_idx, delta_t]
    - Mask indicating valid timesteps
    - Metadata: [redshift, EBV]
    - Label (if training)

    Observations are sorted by time and padded to max_length.
    """

    def __init__(
        self,
        lightcurves: pd.DataFrame,
        metadata: pd.DataFrame,
        object_ids: List[str],
        labels: Optional[Dict[str, int]] = None,
        max_length: int = 500,
        normalize_flux: bool = True,
        include_metadata: bool = True
    ):
        """
        Args:
            lightcurves: DataFrame with columns [object_id, Time (MJD), Flux, Flux_err, Filter]
            metadata: DataFrame with columns [object_id, Z, EBV]
            object_ids: List of object IDs to include
            labels: Optional dict mapping object_id -> label (0 or 1)
            max_length: Maximum sequence length (longer sequences truncated)
            normalize_flux: Whether to normalize flux per object
            include_metadata: Whether to include redshift/EBV features
        """
        self.object_ids = object_ids
        self.labels = labels
        self.max_length = max_length
        self.normalize_flux = normalize_flux
        self.include_metadata = include_metadata

        # Pre-process data for efficiency
        self.sequences = {}
        self.metadata_features = {}

        # Group lightcurves by object
        grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

        # Create metadata lookup
        meta_dict = metadata.set_index('object_id').to_dict('index')

        print(f"    Preprocessing {len(object_ids)} lightcurves...")

        for i, obj_id in enumerate(object_ids):
            if (i + 1) % 1000 == 0:
                print(f"      {i+1}/{len(object_ids)} processed")

            obj_lc = grouped.get(obj_id, pd.DataFrame())

            if obj_lc.empty:
                # Empty sequence
                self.sequences[obj_id] = {
                    'times': np.array([0.0]),
                    'fluxes': np.array([0.0]),
                    'flux_errs': np.array([1.0]),
                    'bands': np.array([1]),  # g-band default
                    'length': 1
                }
            else:
                # Sort by time
                obj_lc = obj_lc.sort_values('Time (MJD)')

                times = obj_lc['Time (MJD)'].values.astype(np.float32)
                fluxes = obj_lc['Flux'].values.astype(np.float32)
                flux_errs = obj_lc['Flux_err'].values.astype(np.float32)
                bands = obj_lc['Filter'].map(BAND_TO_IDX).values.astype(np.int64)

                # Normalize time to start at 0
                times = times - times.min()

                # Handle NaN/inf values
                fluxes = np.nan_to_num(fluxes, nan=0.0, posinf=0.0, neginf=0.0)
                flux_errs = np.nan_to_num(flux_errs, nan=1.0, posinf=1.0, neginf=1.0)
                flux_errs = np.clip(flux_errs, 0.01, None)  # Ensure positive errors

                # Normalize flux if requested
                if self.normalize_flux and fluxes.std() > 1e-6:
                    flux_mean = fluxes.mean()
                    flux_std = fluxes.std() + 1e-6  # Avoid division by zero
                    fluxes = (fluxes - flux_mean) / flux_std
                    flux_errs = flux_errs / flux_std

                # Truncate if too long
                if len(times) > max_length:
                    times = times[:max_length]
                    fluxes = fluxes[:max_length]
                    flux_errs = flux_errs[:max_length]
                    bands = bands[:max_length]

                self.sequences[obj_id] = {
                    'times': times,
                    'fluxes': fluxes,
                    'flux_errs': flux_errs,
                    'bands': bands,
                    'length': len(times)
                }

            # Metadata (handle NaN values)
            meta = meta_dict.get(obj_id, {'Z': 0.0, 'EBV': 0.0})
            z = meta.get('Z', 0)
            ebv = meta.get('EBV', 0)
            # Convert to float, handling NaN and None
            z = 0.0 if (z is None or (isinstance(z, float) and np.isnan(z))) else float(z)
            ebv = 0.0 if (ebv is None or (isinstance(ebv, float) and np.isnan(ebv))) else float(ebv)
            self.metadata_features[obj_id] = np.array([z, ebv], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.object_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        obj_id = self.object_ids[idx]
        seq = self.sequences[obj_id]

        length = seq['length']

        # Create padded arrays
        times = np.zeros(self.max_length, dtype=np.float32)
        fluxes = np.zeros(self.max_length, dtype=np.float32)
        flux_errs = np.ones(self.max_length, dtype=np.float32)  # Default err=1
        bands = np.zeros(self.max_length, dtype=np.int64)
        delta_t = np.zeros(self.max_length, dtype=np.float32)
        mask = np.zeros(self.max_length, dtype=np.float32)

        # Fill with actual data
        times[:length] = seq['times']
        fluxes[:length] = seq['fluxes']
        flux_errs[:length] = seq['flux_errs']
        bands[:length] = seq['bands']
        mask[:length] = 1.0

        # Compute delta_t (time since previous observation)
        if length > 1:
            delta_t[1:length] = np.diff(seq['times'])
            # Normalize delta_t (typical gaps are 1-30 days)
            delta_t = delta_t / 30.0

        # Stack features: [time, flux, flux_err, delta_t]
        # Band is separate for embedding
        features = np.stack([times, fluxes, flux_errs, delta_t], axis=1)

        result = {
            'features': torch.from_numpy(features),           # (max_length, 4)
            'bands': torch.from_numpy(bands),                  # (max_length,)
            'mask': torch.from_numpy(mask),                    # (max_length,)
            'length': torch.tensor(length, dtype=torch.long),
            'object_id': obj_id
        }

        if self.include_metadata:
            result['metadata'] = torch.from_numpy(self.metadata_features[obj_id])

        if self.labels is not None:
            result['label'] = torch.tensor(self.labels.get(obj_id, 0), dtype=torch.float32)

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    Stacks all tensors and handles variable-length sequences.
    """
    result = {
        'features': torch.stack([b['features'] for b in batch]),
        'bands': torch.stack([b['bands'] for b in batch]),
        'mask': torch.stack([b['mask'] for b in batch]),
        'length': torch.stack([b['length'] for b in batch]),
        'object_ids': [b['object_id'] for b in batch]
    }

    if 'metadata' in batch[0]:
        result['metadata'] = torch.stack([b['metadata'] for b in batch])

    if 'label' in batch[0]:
        result['label'] = torch.stack([b['label'] for b in batch])

    return result


if __name__ == "__main__":
    # Test the dataset
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_all_data

    print("Loading data...")
    data = load_all_data()

    # Create labels dict
    labels = dict(zip(
        data['train_meta']['object_id'],
        data['train_meta']['target']
    ))

    print("\nCreating dataset...")
    sample_ids = data['train_meta']['object_id'].head(100).tolist()
    dataset = LightcurveDataset(
        lightcurves=data['train_lc'],
        metadata=data['train_meta'],
        object_ids=sample_ids,
        labels=labels,
        max_length=300
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test single sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Features shape: {sample['features'].shape}")
    print(f"Bands shape: {sample['bands'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Length: {sample['length']}")
    print(f"Label: {sample['label']}")

    # Test DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch features shape: {batch['features'].shape}")
    print(f"Batch labels shape: {batch['label'].shape}")
