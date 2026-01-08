"""
Setup ASTROMER with compatible TensorFlow

ASTROMER requires TensorFlow < 2.16 due to Keras API changes.
This script sets up a working ASTROMER environment.
"""

import subprocess
import sys

# We'll use Python 3.9 (Anaconda) with TensorFlow 2.10
PYTHON = r"C:\Users\alexy\anaconda3\python.exe"

def run_cmd(cmd, description):
    print(f"\n{description}...", flush=True)
    print(f"  Command: {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}", flush=True)
        return False
    print(f"  SUCCESS", flush=True)
    return True

print("=" * 60)
print("Setting up ASTROMER Environment")
print("=" * 60)

# Step 1: Install compatible TensorFlow
run_cmd(f'"{PYTHON}" -m pip install tensorflow==2.10.0 --quiet',
        "Installing TensorFlow 2.10")

# Step 2: Install ASTROMER
run_cmd(f'"{PYTHON}" -m pip install ASTROMER==0.1.7 --quiet',
        "Installing ASTROMER 0.1.7")

# Step 3: Test import
print("\nTesting ASTROMER import...")
test_code = '''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from ASTROMER.models import SingleBandEncoder
print("ASTROMER imported successfully!")
encoder = SingleBandEncoder()
encoder.from_pretraining('macho')
print("Pre-trained MACHO weights loaded!")
print(f"Encoder ready for embedding extraction")
'''

result = subprocess.run(
    [PYTHON, "-c", test_code],
    capture_output=True, text=True
)

if result.returncode == 0:
    print(result.stdout)
    print("\n✓ ASTROMER is ready for use!")
else:
    print(f"Error: {result.stderr}")
    print("\n✗ ASTROMER setup failed")
