"""
Categorical Feature Engineering for CatBoost

CatBoost's key advantage: handles categorical features with native ordered target encoding.
This is more powerful than one-hot encoding and captures complex interactions.

Strategy:
1. Bin continuous features into physics-meaningful categories
2. Create ordinal encodings (low/medium/high) for key discriminative features
3. Let CatBoost learn optimal splits on these categories

Key insight: Astrophysical features often have natural breakpoints
(e.g., redshift ranges, color regimes, timescale classifications)
"""

import numpy as np
import pandas as pd
from typing import List


def create_redshift_categories(z: np.ndarray) -> np.ndarray:
    """
    Bin redshift into physically meaningful categories.

    Different redshift ranges have different observational properties:
    - z < 0.1: Nearby (bright, well-sampled)
    - 0.1 <= z < 0.3: Intermediate
    - 0.3 <= z < 0.6: Moderate distance
    - z >= 0.6: Distant (faint, sparse sampling)
    """
    categories = np.zeros(len(z), dtype=int)
    categories[(z >= 0.1) & (z < 0.3)] = 1
    categories[(z >= 0.3) & (z < 0.6)] = 2
    categories[z >= 0.6] = 3
    return categories


def create_color_categories(colors: np.ndarray, band_name: str) -> np.ndarray:
    """
    Bin colors into Blue/Normal/Red categories.

    Colors indicate:
    - Blue: Hot blackbody (TDEs stay blue)
    - Normal: Cooling transient
    - Red: Cool or dusty source
    """
    categories = np.zeros(len(colors), dtype=int)

    # Use quantiles for robust binning
    valid = np.isfinite(colors)
    if np.sum(valid) > 10:
        q25, q75 = np.nanpercentile(colors, [25, 75])
        categories[colors < q25] = 0  # Blue
        categories[(colors >= q25) & (colors < q75)] = 1  # Normal
        categories[colors >= q75] = 2  # Red

    return categories


def create_timescale_categories(times: np.ndarray, feature_name: str) -> np.ndarray:
    """
    Bin timescales (rise/fall/duration) into Fast/Medium/Slow.

    Timescales discriminate transient types:
    - Fast: < 20 days (fast SNe, some TDEs)
    - Medium: 20-100 days (typical SNe, TDEs)
    - Slow: > 100 days (slow SNe, AGN)
    """
    categories = np.zeros(len(times), dtype=int)

    valid = np.isfinite(times)
    if np.sum(valid) > 10:
        # Use physics-motivated thresholds
        categories[(times >= 20) & (times < 100)] = 1  # Medium
        categories[times >= 100] = 2  # Slow

    return categories


def create_brightness_categories(fluxes: np.ndarray) -> np.ndarray:
    """
    Bin peak brightness into Faint/Medium/Bright.

    Brightness correlates with distance and intrinsic luminosity.
    """
    categories = np.zeros(len(fluxes), dtype=int)

    valid = np.isfinite(fluxes)
    if np.sum(valid) > 10:
        q33, q67 = np.nanpercentile(fluxes, [33, 67])
        categories[fluxes < q33] = 0  # Faint
        categories[(fluxes >= q33) & (fluxes < q67)] = 1  # Medium
        categories[fluxes >= q67] = 2  # Bright

    return categories


def create_asymmetry_categories(asymmetry: np.ndarray) -> np.ndarray:
    """
    Bin asymmetry (rise_time / fall_time) into categories.

    Asymmetry discriminates transient physics:
    - Symmetric (0.5-2): Similar rise/fall (some SNe)
    - Moderate (2-10): Typical SNe, TDEs
    - Highly asymmetric (>10): Fast rise, slow fall (TDEs)
    """
    categories = np.zeros(len(asymmetry), dtype=int)

    valid = np.isfinite(asymmetry) & (asymmetry > 0)
    if np.sum(valid) > 10:
        categories[(asymmetry >= 2) & (asymmetry < 10)] = 1  # Moderate
        categories[asymmetry >= 10] = 2  # Highly asymmetric

    return categories


def create_fit_quality_categories(chi2: np.ndarray) -> np.ndarray:
    """
    Bin fit quality (chi-squared) into Poor/Fair/Good.

    Fit quality indicates:
    - Good fit: Model captures lightcurve well
    - Poor fit: Complex/unusual behavior (AGN, peculiar transients)
    """
    categories = np.zeros(len(chi2), dtype=int)

    valid = np.isfinite(chi2) & (chi2 > 0)
    if np.sum(valid) > 10:
        # Lower chi2 = better fit
        q33, q67 = np.nanpercentile(chi2[valid], [33, 67])
        categories[chi2 <= q33] = 2  # Good fit
        categories[(chi2 > q33) & (chi2 <= q67)] = 1  # Fair
        # chi2 > q67: Poor fit (category 0)

    return categories


def create_variability_categories(variability: np.ndarray) -> np.ndarray:
    """
    Bin variability metrics into Low/Medium/High.

    Variability discriminates:
    - Low: Steady sources
    - Medium: Transients
    - High: AGN, unusual variables
    """
    categories = np.zeros(len(variability), dtype=int)

    valid = np.isfinite(variability)
    if np.sum(valid) > 10:
        q33, q67 = np.nanpercentile(variability, [33, 67])
        categories[variability < q33] = 0  # Low
        categories[(variability >= q33) & (variability < q67)] = 1  # Medium
        categories[variability >= q67] = 2  # High

    return categories


def add_categorical_features(features_df: pd.DataFrame) -> tuple:
    """
    Add categorical features to existing feature DataFrame.

    Args:
        features_df: DataFrame with continuous features (must include object_id)

    Returns:
        Tuple of (enhanced_df, categorical_feature_indices)
    """
    df = features_df.copy()
    categorical_cols = []

    # 1. Redshift categories
    if 'Z' in df.columns:
        df['Z_category'] = create_redshift_categories(df['Z'].values)
        categorical_cols.append('Z_category')

    # 2. Color categories (for key colors)
    color_features = ['gp_gr_color_50d', 'gp_ri_color_50d', 'gp_gr_color_20d', 'gp_ri_color_20d']
    for color_feat in color_features:
        if color_feat in df.columns:
            cat_name = f'{color_feat}_cat'
            df[cat_name] = create_color_categories(df[color_feat].values, color_feat)
            categorical_cols.append(cat_name)

    # 3. Timescale categories
    timescale_features = []
    for band in ['u', 'g', 'r', 'i', 'z', 'y', 'all']:
        for feat in ['rise_time', 'fall_time', 'duration_50', 'duration_25', 'duration_75']:
            feat_name = f'{band}_{feat}'
            if feat_name in df.columns:
                timescale_features.append(feat_name)

    for timescale_feat in timescale_features:
        if timescale_feat in df.columns:
            cat_name = f'{timescale_feat}_cat'
            df[cat_name] = create_timescale_categories(df[timescale_feat].values, timescale_feat)
            categorical_cols.append(cat_name)

    # 4. Brightness categories (peak fluxes)
    brightness_features = []
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        for feat in ['peak_flux', 'mean_flux']:
            feat_name = f'{band}_{feat}'
            if feat_name in df.columns:
                brightness_features.append(feat_name)

    for bright_feat in brightness_features:
        if bright_feat in df.columns:
            cat_name = f'{bright_feat}_cat'
            df[cat_name] = create_brightness_categories(df[bright_feat].values)
            categorical_cols.append(cat_name)

    # 5. Asymmetry categories
    asymmetry_features = [f'{band}_asymmetry' for band in ['u', 'g', 'r', 'i', 'z', 'y', 'all']]
    for asym_feat in asymmetry_features:
        if asym_feat in df.columns:
            cat_name = f'{asym_feat}_cat'
            df[cat_name] = create_asymmetry_categories(df[asym_feat].values)
            categorical_cols.append(cat_name)

    # 6. Fit quality categories
    fit_quality_features = []
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        for feat in ['bazin_fit_chi2', 'gp_fit_chi2']:
            feat_name = f'{band}_{feat}'
            if feat_name in df.columns:
                fit_quality_features.append(feat_name)

    # Add cross-band fit quality
    if 'bazin_avg_fit_chi2' in df.columns:
        fit_quality_features.append('bazin_avg_fit_chi2')
    if 'gp2d_log_likelihood' in df.columns:
        fit_quality_features.append('gp2d_log_likelihood')

    for fit_feat in fit_quality_features:
        if fit_feat in df.columns:
            cat_name = f'{fit_feat}_cat'
            df[cat_name] = create_fit_quality_categories(df[fit_feat].values)
            categorical_cols.append(cat_name)

    # 7. Variability categories
    variability_features = []
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        for feat in ['std_flux', 'mad_flux', 'skew', 'kurtosis']:
            feat_name = f'{band}_{feat}'
            if feat_name in df.columns:
                variability_features.append(feat_name)

    for var_feat in variability_features:
        if var_feat in df.columns:
            cat_name = f'{var_feat}_cat'
            df[cat_name] = create_variability_categories(df[var_feat].values)
            categorical_cols.append(cat_name)

    # Get categorical feature indices (excluding object_id)
    all_cols = [c for c in df.columns if c != 'object_id']
    cat_indices = [all_cols.index(cat_col) for cat_col in categorical_cols if cat_col in all_cols]

    print(f"   Added {len(categorical_cols)} categorical features", flush=True)
    print(f"   Categorical feature indices: {len(cat_indices)}", flush=True)

    return df, cat_indices


if __name__ == "__main__":
    # Test categorical feature creation
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_all_data

    print("Loading data...")
    data = load_all_data()

    # Load v4 features
    import pickle
    base_path = Path(__file__).parent.parent.parent
    cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
    train_features = cached['train_features']

    print(f"\nOriginal features: {len(train_features.columns)-1}")

    # Add categorical features
    enhanced_features, cat_indices = add_categorical_features(train_features)

    print(f"Enhanced features: {len(enhanced_features.columns)-1}")
    print(f"Categorical features: {len(cat_indices)}")

    # Show sample categorical features
    cat_cols = [c for c in enhanced_features.columns if '_cat' in c]
    print(f"\nSample categorical features (first 10):")
    for col in cat_cols[:10]:
        print(f"  {col}: {enhanced_features[col].nunique()} unique values")
