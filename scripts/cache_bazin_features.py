"""
Precompute and cache Bazin parametric features for reuse.
"""
import sys
import pickle
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("Caching Bazin Features for v34a...")
print("=" * 80)

# Load data
from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_lc = data['train_lc']
test_lc = data['test_lc']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()

# Extract Bazin features
from features.bazin_fitting import extract_bazin_features

print("\nExtracting Bazin features for training set...")
train_bazin = extract_bazin_features(train_lc, train_ids)
print(f"   Extracted {len(train_bazin.columns)-1} features for {len(train_bazin)} objects")

print("\nExtracting Bazin features for test set...")
test_bazin = extract_bazin_features(test_lc, test_ids)
print(f"   Extracted {len(test_bazin.columns)-1} features for {len(test_bazin)} objects")

# Save cache
cache = {
    'train': train_bazin,
    'test': test_bazin
}

cache_path = base_path / 'data/processed/bazin_features_cache.pkl'
pd.to_pickle(cache, cache_path)

print(f"\nBazin features cached to: {cache_path}")
print(f"Cache size: {cache_path.stat().st_size / 1024:.1f} KB")
print("=" * 80)
