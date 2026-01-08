"""
Extract ASTROMER embeddings for all MALLORN objects.
Run with Python 3.11 which has ASTROMER + TensorFlow 2.15

This creates embeddings that can be loaded by any Python version.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Setup paths
base_path = Path(__file__).parent.parent
os.chdir(base_path)

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 60)
print("ASTROMER Embedding Extraction")
print("=" * 60)

# Check for existing embeddings
train_emb_path = base_path / 'data/processed/train_astromer_embeddings.csv'
test_emb_path = base_path / 'data/processed/test_astromer_embeddings.csv'

if train_emb_path.exists() and test_emb_path.exists():
    print("Embeddings already exist!")
    train_emb = pd.read_csv(train_emb_path)
    test_emb = pd.read_csv(test_emb_path)
    print(f"Train: {train_emb.shape}, Test: {test_emb.shape}")
    sys.exit(0)

# Load metadata
print("\n[1/4] Loading metadata...")
train_log = pd.read_csv(base_path / 'data/raw/train_log.csv')
test_log = pd.read_csv(base_path / 'data/raw/test_log.csv')
train_ids = train_log['object_id'].tolist()
test_ids = test_log['object_id'].tolist()
print(f"   Train: {len(train_ids)}, Test: {len(test_ids)}")

# Load light curves
print("\n[2/4] Loading light curves...")
train_lcs = []
for i in range(1, 21):
    path = base_path / f'data/raw/split_{i:02d}/train_full_lightcurves.csv'
    if path.exists():
        train_lcs.append(pd.read_csv(path))
train_lc = pd.concat(train_lcs, ignore_index=True)

test_lcs = []
for i in range(1, 21):
    path = base_path / f'data/raw/split_{i:02d}/test_full_lightcurves.csv'
    if path.exists():
        test_lcs.append(pd.read_csv(path))
test_lc = pd.concat(test_lcs, ignore_index=True)
print(f"   Train LC: {len(train_lc)}, Test LC: {len(test_lc)}")

# Import TensorFlow and ASTROMER
print("\n[3/4] Loading ASTROMER...")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from ASTROMER.models import SingleBandEncoder

encoder = SingleBandEncoder()
encoder = encoder.from_pretraining('macho')
encoder_layer = encoder.model.get_layer('encoder')
print("   ASTROMER loaded!")

# Extract embeddings function
def extract_embeddings(lightcurves_df, object_ids, desc=""):
    """Extract ASTROMER embeddings for multiple objects."""
    max_obs = 100
    bands = ['g', 'r', 'i', 'z']  # Key bands
    n_components = 32  # Keep top 32 of 256 embedding dimensions

    all_results = []
    total = len(object_ids)

    for idx, obj_id in enumerate(object_ids):
        if (idx + 1) % 200 == 0:
            print(f"   {desc} Progress: {idx+1}/{total} ({100*(idx+1)/total:.1f}%)", flush=True)

        obj_lc = lightcurves_df[lightcurves_df['object_id'] == obj_id]
        result = {'object_id': obj_id}

        for band in bands:
            band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

            # Default values for missing bands
            if len(band_data) < 5:
                for suffix in ['mean', 'max', 'std']:
                    for i in range(n_components):
                        result[f'{band}_emb_{suffix}_{i}'] = 0.0
                continue

            # Prepare data
            times = band_data['Time (MJD)'].values.copy()
            fluxes = band_data['Flux'].values.copy()

            times = times - times.min()
            flux_mean, flux_std = fluxes.mean(), fluxes.std()
            if flux_std > 0:
                fluxes = (fluxes - flux_mean) / flux_std

            n_obs = min(len(times), max_obs)
            times = times[:n_obs]
            fluxes = fluxes[:n_obs]

            # Pad
            times_padded = np.zeros(max_obs, dtype=np.float32)
            fluxes_padded = np.zeros(max_obs, dtype=np.float32)
            mask = np.ones(max_obs, dtype=np.float32)

            times_padded[:n_obs] = times
            fluxes_padded[:n_obs] = fluxes
            mask[:n_obs] = 0

            # Get embeddings
            input_data = {
                'input': tf.constant(fluxes_padded.reshape(1, max_obs, 1)),
                'times': tf.constant(times_padded.reshape(1, max_obs, 1)),
                'mask_in': tf.constant(mask.reshape(1, max_obs, 1))
            }

            emb = encoder_layer(input_data).numpy()[0, :n_obs, :]  # (n_obs, 256)

            # Aggregate embeddings (take first n_components of each aggregation)
            emb_mean = emb.mean(axis=0)[:n_components]
            emb_max = emb.max(axis=0)[:n_components]
            emb_std = emb.std(axis=0)[:n_components]

            for i in range(n_components):
                result[f'{band}_emb_mean_{i}'] = float(emb_mean[i])
                result[f'{band}_emb_max_{i}'] = float(emb_max[i])
                result[f'{band}_emb_std_{i}'] = float(emb_std[i])

        all_results.append(result)

    return pd.DataFrame(all_results)

# Extract training embeddings
print("\n[4/4] Extracting embeddings...")
print("   Training set...")
train_embeddings = extract_embeddings(train_lc, train_ids, "Train")
train_embeddings.to_csv(train_emb_path, index=False)
print(f"   Saved: {train_emb_path}")

print("   Test set...")
test_embeddings = extract_embeddings(test_lc, test_ids, "Test")
test_embeddings.to_csv(test_emb_path, index=False)
print(f"   Saved: {test_emb_path}")

print(f"\nFinal shapes: Train={train_embeddings.shape}, Test={test_embeddings.shape}")
print("Done!")
