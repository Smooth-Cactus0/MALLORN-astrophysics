"""Test ORACLE import and model loading."""
import sys
print(f"Python version: {sys.version}")

try:
    from oracle.pretrained.ELAsTiCC import ORACLE1_ELAsTiCC_lite
    print("ORACLE module imported successfully!")

    # Try to load the model
    print("Loading ORACLE1_ELAsTiCC_lite model...")
    model = ORACLE1_ELAsTiCC_lite()
    print("Model loaded successfully!")
    print(f"Model type: {type(model)}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
