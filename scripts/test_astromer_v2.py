"""Test ASTROMER with TensorFlow 2.15."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
print(f"Python: {sys.version}")

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")

try:
    print("\nImporting ASTROMER...")
    from ASTROMER.models import SingleBandEncoder
    print("ASTROMER imported!")

    print("\nLoading pre-trained MACHO weights...")
    encoder = SingleBandEncoder()
    encoder.from_pretraining('macho')
    print("Pre-trained model loaded successfully!")

    # Test with dummy data
    import numpy as np
    print("\nTesting embedding extraction...")

    # Create dummy light curve data
    dummy_data = {
        'input': np.random.randn(2, 100, 1).astype(np.float32),
        'times': np.linspace(0, 500, 100).reshape(1, 100, 1).repeat(2, axis=0).astype(np.float32),
        'mask_in': np.ones((2, 100, 1), dtype=np.float32),
        'length': np.array([[100], [100]], dtype=np.int32)
    }

    # Get embeddings
    embeddings = encoder.encode(dummy_data)
    print(f"Embedding shape: {embeddings.shape}")
    print("\nASTROMER is working!")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
