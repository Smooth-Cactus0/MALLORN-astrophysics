"""Test ASTROMER import and model loading."""
import sys
print(f"Python version: {sys.version}")

try:
    print("Importing ASTROMER...")
    from ASTROMER.models import SingleBandEncoder
    print("ASTROMER imported successfully!")

    # Try to load pre-trained weights
    print("\nLoading pre-trained MACHO weights...")
    encoder = SingleBandEncoder()
    encoder.from_pretraining('macho')
    print("Pre-trained model loaded successfully!")

    # Check model architecture
    print(f"\nModel summary available: {hasattr(encoder, 'encoder')}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
