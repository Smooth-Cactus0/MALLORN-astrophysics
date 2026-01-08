"""
Prepare PLAsTiCC Data for MALLORN Training

Converts PLAsTiCC format to MALLORN format and creates combined dataset.

PLAsTiCC has 495 TDEs vs MALLORN's 148 -> 643 total TDEs (4.3x increase!)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

def main():
    print("=" * 60)
    print("Preparing PLAsTiCC Data for MALLORN")
    print("=" * 60)

    base_path = Path(__file__).parent.parent

    # 1. Load PLAsTiCC data
    print("\n1. Loading PLAsTiCC dataset...")
    plasticc_path = base_path / 'data/external/plasticc/dataset.csv'
    plasticc = pd.read_csv(plasticc_path)

    # Passband mapping: PLAsTiCC uses 0-5, MALLORN uses ugrizy
    band_map = {0: 'u', 1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'y'}

    print(f"   Total PLAsTiCC objects: {plasticc['object_id'].nunique()}")

    # 2. Extract TDEs (class 15) and non-TDEs
    print("\n2. Extracting TDEs and non-TDEs...")

    # Get object-level info
    object_info = plasticc.groupby('object_id').agg({
        'target': 'first',
        'hostgal_specz': 'first',
        'hostgal_photoz': 'first',
        'mwebv': 'first'
    }).reset_index()

    tde_ids = object_info[object_info['target'] == 15]['object_id'].values
    print(f"   TDEs: {len(tde_ids)}")

    # For non-TDEs, sample a balanced set (AGN + SNe)
    # AGN (88), SNIa (90), SNII (42), SNIbc (62)
    non_tde_classes = [88, 90, 42, 62]
    non_tde_ids = object_info[object_info['target'].isin(non_tde_classes)]['object_id'].values
    print(f"   Non-TDEs (AGN+SNe): {len(non_tde_ids)}")

    # 3. Convert to MALLORN format
    print("\n3. Converting to MALLORN format...")

    # Get lightcurves for selected objects
    selected_ids = np.concatenate([tde_ids, non_tde_ids])
    plasticc_selected = plasticc[plasticc['object_id'].isin(selected_ids)].copy()

    # Create lightcurve DataFrame in MALLORN format
    lc_mallorn = pd.DataFrame({
        'object_id': 'PLAsTiCC_' + plasticc_selected['object_id'].astype(str),
        'Time (MJD)': plasticc_selected['mjd'],
        'Flux': plasticc_selected['flux'],
        'Flux_err': plasticc_selected['flux_err'],
        'Filter': plasticc_selected['passband'].map(band_map)
    })

    # Create metadata DataFrame in MALLORN format
    meta_info = object_info[object_info['object_id'].isin(selected_ids)].copy()
    meta_mallorn = pd.DataFrame({
        'object_id': 'PLAsTiCC_' + meta_info['object_id'].astype(str),
        'Z': meta_info['hostgal_specz'].fillna(meta_info['hostgal_photoz']),
        'EBV': meta_info['mwebv'],
        'target': (meta_info['target'] == 15).astype(int),  # 1 = TDE, 0 = non-TDE
        'source': 'PLAsTiCC'
    })

    print(f"   Converted lightcurves: {len(lc_mallorn):,} observations")
    print(f"   Converted metadata: {len(meta_mallorn)} objects")
    print(f"   TDEs: {(meta_mallorn['target'] == 1).sum()}")
    print(f"   Non-TDEs: {(meta_mallorn['target'] == 0).sum()}")

    # 4. Load MALLORN data
    print("\n4. Loading MALLORN training data...")
    import sys
    sys.path.insert(0, str(base_path / 'src'))
    from utils.data_loader import load_all_data

    mallorn = load_all_data()
    mallorn_lc = mallorn['train_lc'].copy()
    mallorn_meta = mallorn['train_meta'].copy()
    mallorn_meta['source'] = 'MALLORN'

    print(f"   MALLORN lightcurves: {len(mallorn_lc):,} observations")
    print(f"   MALLORN objects: {len(mallorn_meta)}")
    print(f"   MALLORN TDEs: {(mallorn_meta['target'] == 1).sum()}")

    # 5. Combine datasets
    print("\n5. Combining datasets...")

    combined_lc = pd.concat([mallorn_lc, lc_mallorn], ignore_index=True)
    combined_meta = pd.concat([mallorn_meta[['object_id', 'Z', 'EBV', 'target', 'source']],
                               meta_mallorn], ignore_index=True)

    total_tdes = (combined_meta['target'] == 1).sum()
    total_non_tdes = (combined_meta['target'] == 0).sum()

    print(f"\n   Combined dataset:")
    print(f"   - Total objects: {len(combined_meta)}")
    print(f"   - TDEs: {total_tdes} (MALLORN: 148, PLAsTiCC: {total_tdes - 148})")
    print(f"   - Non-TDEs: {total_non_tdes}")
    print(f"   - TDE percentage: {total_tdes / len(combined_meta) * 100:.1f}%")
    print(f"   - Total lightcurve observations: {len(combined_lc):,}")

    # 6. Save combined dataset
    print("\n6. Saving combined dataset...")

    output_path = base_path / 'data/processed/combined_plasticc_mallorn.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump({
            'combined_lc': combined_lc,
            'combined_meta': combined_meta,
            'mallorn_meta': mallorn_meta,  # Keep original MALLORN meta for validation
            'mallorn_lc': mallorn_lc
        }, f)

    print(f"   Saved to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"\nReady for training:")
    print(f"  Before: 148 TDEs (MALLORN only)")
    print(f"  After:  {total_tdes} TDEs (4.3x increase!)")
    print(f"  Non-TDEs: {total_non_tdes} (diverse: AGN + SNe)")
    print("=" * 60)


if __name__ == "__main__":
    main()
