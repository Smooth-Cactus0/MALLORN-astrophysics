"""
Data loading utilities for MALLORN Astronomical Classification Challenge.

This module provides functions to load and prepare the competition data:
- train_log.csv / test_log.csv: Object metadata with redshift, extinction, labels
- split_XX/train_full_lightcurves.csv: Time-series flux measurements
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


def get_data_path() -> Path:
    """Get the path to the raw data directory."""
    return Path(__file__).parent.parent.parent / "data" / "raw"


def load_metadata(data_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test metadata (log files).

    Returns:
        Tuple of (train_log, test_log) DataFrames
    """
    if data_path is None:
        data_path = get_data_path()

    train_log = pd.read_csv(data_path / "train_log.csv")
    test_log = pd.read_csv(data_path / "test_log.csv")

    return train_log, test_log


def load_lightcurves(split: str = "train", data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load all lightcurve data from all splits.

    Args:
        split: "train" or "test"
        data_path: Optional path to raw data directory

    Returns:
        Concatenated DataFrame with all lightcurve observations
    """
    if data_path is None:
        data_path = get_data_path()

    filename = f"{split}_full_lightcurves.csv"
    all_lightcurves = []

    for i in range(1, 21):
        split_path = data_path / f"split_{i:02d}" / filename
        if split_path.exists():
            df = pd.read_csv(split_path)
            all_lightcurves.append(df)

    if not all_lightcurves:
        raise FileNotFoundError(f"No {split} lightcurve files found")

    return pd.concat(all_lightcurves, ignore_index=True)


def load_all_data(data_path: Optional[Path] = None) -> dict:
    """
    Load all competition data.

    Returns:
        Dictionary with keys:
        - train_meta: Training metadata
        - test_meta: Test metadata
        - train_lc: Training lightcurves
        - test_lc: Test lightcurves
    """
    if data_path is None:
        data_path = get_data_path()

    train_meta, test_meta = load_metadata(data_path)
    train_lc = load_lightcurves("train", data_path)
    test_lc = load_lightcurves("test", data_path)

    return {
        "train_meta": train_meta,
        "test_meta": test_meta,
        "train_lc": train_lc,
        "test_lc": test_lc
    }


def get_object_lightcurve(lightcurves: pd.DataFrame, object_id: str) -> pd.DataFrame:
    """
    Get lightcurve data for a specific object.

    Args:
        lightcurves: Full lightcurves DataFrame
        object_id: The object ID to filter

    Returns:
        DataFrame with lightcurve for the specified object
    """
    return lightcurves[lightcurves["object_id"] == object_id].copy()


def get_class_distribution(train_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Get the distribution of classes in training data.

    Returns:
        DataFrame with SpecType counts and target distribution
    """
    spec_dist = train_meta["SpecType"].value_counts().reset_index()
    spec_dist.columns = ["SpecType", "count"]

    target_dist = train_meta["target"].value_counts().reset_index()
    target_dist.columns = ["target", "count"]

    return spec_dist, target_dist


# Constants
LSST_BANDS = ["u", "g", "r", "i", "z", "y"]
BAND_WAVELENGTHS = {
    "u": 367.0,  # nm
    "g": 482.5,
    "r": 622.2,
    "i": 754.5,
    "z": 869.1,
    "y": 971.0
}


if __name__ == "__main__":
    # Quick test
    data = load_all_data()
    print(f"Train objects: {len(data['train_meta'])}")
    print(f"Test objects: {len(data['test_meta'])}")
    print(f"Train lightcurve observations: {len(data['train_lc'])}")
    print(f"Test lightcurve observations: {len(data['test_lc'])}")
    print(f"\nClass distribution:")
    spec_dist, target_dist = get_class_distribution(data['train_meta'])
    print(spec_dist)
    print(f"\nTarget distribution (0=non-TDE, 1=TDE):")
    print(target_dist)
