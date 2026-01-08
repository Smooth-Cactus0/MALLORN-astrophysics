"""Test ASTROMER with manually downloaded weights."""
import os
import sys

# Change to project directory (where weights/ folder is)
project_dir = r"C:\Users\alexy\Documents\Claude_projects\Kaggle competition\MALLORN astrophysics"
os.chdir(project_dir)
print(f"Working directory: {os.getcwd()}")

# Verify weights exist
weights_path = os.path.join(project_dir, "weights", "macho")
print(f"Weights path exists: {os.path.isdir(weights_path)}")
print(f"Weights contents: {os.listdir(weights_path)}")

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"\nPython: {sys.version}")

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")

try:
    print("\nImporting ASTROMER...")
    from ASTROMER.models import SingleBandEncoder
    print("ASTROMER imported!")

    print("\nLoading pre-trained MACHO weights...")
    encoder = SingleBandEncoder()
    encoder = encoder.from_pretraining('macho')
    print("Pre-trained model loaded successfully!")

    # Test with dummy data
    import numpy as np
    print("\nTesting embedding extraction...")

    # Create dummy light curve data matching ASTROMER format
    # ASTROMER expects: input (flux), times, mask_in, length
    n_samples = 2
    max_obs = 100

    dummy_data = {
        'input': np.random.randn(n_samples, max_obs, 1).astype(np.float32),
        'times': np.linspace(0, 500, max_obs).reshape(1, max_obs, 1).repeat(n_samples, axis=0).astype(np.float32),
        'mask_in': np.zeros((n_samples, max_obs, 1), dtype=np.float32),  # 0 = valid, 1 = masked
        'length': np.array([[max_obs], [max_obs]], dtype=np.int32)
    }

    # Convert to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices({
        'input': dummy_data['input'],
        'times': dummy_data['times'],
        'mask_in': dummy_data['mask_in'],
        'id': ['obj1', 'obj2']
    }).batch(2)

    # Get embeddings
    embeddings = encoder.encode(dataset, concatenate=False)
    print(f"Number of embedding batches: {len(embeddings)}")
    print(f"First embedding shape: {embeddings[0].shape}")

    print("\n" + "="*50)
    print("ASTROMER is working!")
    print("="*50)

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
