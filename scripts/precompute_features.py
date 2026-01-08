"""
Pre-compute GP and enhanced color features.

This script pre-computes expensive features and caches them.
"""

import sys
import pickle
from pathlib import Path

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

print("=" * 60, flush=True)
print("Pre-computing GP and Enhanced Color Features", flush=True)
print("=" * 60, flush=True)

base_path = Path(__file__).parent.parent

# 1. Load data
print("\n1. Loading data...", flush=True)
from utils.data_loader import load_all_data
data = load_all_data()

train_lc = data['train_lc']
test_lc = data['test_lc']
train_meta = data['train_meta']
test_meta = data['test_meta']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()

print(f"   Train: {len(train_ids)} objects", flush=True)
print(f"   Test: {len(test_ids)} objects", flush=True)

# 2. Enhanced colors (faster)
print("\n2. Computing enhanced color features...", flush=True)
from features.colors import extract_color_features

color_cache_path = base_path / 'data/processed/enhanced_colors_cache.pkl'
if color_cache_path.exists():
    print("   Already cached!", flush=True)
else:
    print("   Computing train colors...", flush=True)
    train_colors = extract_color_features(train_lc, train_ids)
    print(f"   Train colors: {len(train_colors.columns)-1} features", flush=True)

    print("   Computing test colors...", flush=True)
    test_colors = extract_color_features(test_lc, test_ids)
    print(f"   Test colors: {len(test_colors.columns)-1} features", flush=True)

    with open(color_cache_path, 'wb') as f:
        pickle.dump({'train': train_colors, 'test': test_colors}, f)
    print(f"   Saved to {color_cache_path}", flush=True)

# 3. GP features (slower)
print("\n3. Computing GP length scale features...", flush=True)
from features.gaussian_process import extract_gp_features

gp_cache_path = base_path / 'data/processed/gp_features_cache.pkl'
if gp_cache_path.exists():
    print("   Already cached!", flush=True)
else:
    print("   Computing train GP features (this takes ~5-10 minutes)...", flush=True)
    train_gp = extract_gp_features(train_lc, train_meta, train_ids, verbose=True)
    print(f"   Train GP: {len(train_gp.columns)-1} features", flush=True)

    print("   Computing test GP features (this takes ~15-20 minutes)...", flush=True)
    test_gp = extract_gp_features(test_lc, test_meta, test_ids, verbose=True)
    print(f"   Test GP: {len(test_gp.columns)-1} features", flush=True)

    with open(gp_cache_path, 'wb') as f:
        pickle.dump({'train': train_gp, 'test': test_gp}, f)
    print(f"   Saved to {gp_cache_path}", flush=True)

print("\n" + "=" * 60, flush=True)
print("DONE! Features are now cached.", flush=True)
print("=" * 60, flush=True)
