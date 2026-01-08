"""
Feature interactions module for MALLORN.

Creates physics-motivated interactions that capture non-linear relationships
tree models might miss.

Key physics interactions:
1. Color × Redshift: Rest-frame color evolution depends on redshift
2. Temperature × Time: Cooling rate is non-linear in temperature
3. Amplitude × Duration: Peak brightness vs timescale relationship
4. GP features × Redshift: Time-domain features scale with redshift
5. Asymmetry × Color: Rise/fade related to spectral evolution

Reference:
- van Velzen+ 2020: TDE properties correlate in multi-dimensional space
- PLAsTiCC winning solutions: Interaction features crucial for performance
"""

import numpy as np
import pandas as pd
from typing import List, Dict

def create_physics_interactions(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """
    Create physics-motivated feature interactions.

    Args:
        df: DataFrame with original features
        feature_names: List of feature column names

    Returns:
        DataFrame with interaction features added
    """
    interactions = df.copy()

    # Helper to safely get feature if it exists
    def get_feat(name):
        if name in feature_names:
            return df[name].values
        return None

    # ====================
    # 1. COLOR × REDSHIFT
    # ====================
    # Rest-frame colors depend critically on redshift
    Z = get_feat('Z')
    if Z is not None:
        for color_feat in ['g_r_at_peak', 'g_r_post_20d', 'g_r_post_50d', 'r_i_at_peak']:
            color = get_feat(color_feat)
            if color is not None:
                # Color evolution rate depends on redshift
                interactions[f'{color_feat}_x_Z'] = color * Z
                interactions[f'{color_feat}_div_Z'] = color / (Z + 0.1)

        # GP color features × redshift
        for gp_color in ['gp_gr_color_20d', 'gp_gr_color_50d', 'gp_ri_color_20d']:
            color = get_feat(gp_color)
            if color is not None:
                interactions[f'{gp_color}_x_Z'] = color * Z

    # ====================
    # 2. TEMPERATURE × TIME
    # ====================
    # Cooling rate is non-linear: dT/dt ∝ T^4 (Stefan-Boltzmann)
    temp_peak = get_feat('temp_at_peak')
    temp_post = get_feat('temp_post_50d')

    if temp_peak is not None and temp_post is not None:
        # Temperature ratio (cooling factor)
        interactions['temp_cooling_ratio'] = temp_post / (temp_peak + 100)

        # Temperature drop rate
        interactions['temp_drop_rate'] = (temp_peak - temp_post) / 50.0

        # Non-linear cooling (T^4 relation)
        interactions['temp_peak_4th'] = np.power(np.clip(temp_peak, 0, 100000), 0.25)

    # ====================
    # 3. AMPLITUDE × DURATION
    # ====================
    # Brighter transients often have different timescales
    for band in ['g', 'r', 'i']:
        peak = get_feat(f'{band}_peak_flux')
        duration = get_feat(f'{band}_duration_50')

        if peak is not None and duration is not None:
            # Flux-weighted duration
            interactions[f'{band}_flux_duration'] = peak * duration

            # Specific luminosity (flux per day)
            interactions[f'{band}_flux_per_day'] = peak / (duration + 1)

    # ====================
    # 4. GP FEATURES × AMPLITUDE
    # ====================
    # GP length scales interact with amplitude
    gp_time = get_feat('gp2d_time_scale')
    gp_wave = get_feat('gp2d_wave_scale')

    for band in ['g', 'r', 'i']:
        amplitude = get_feat(f'{band}_amplitude')

        if gp_time is not None and amplitude is not None:
            interactions[f'{band}_gp_amp_time'] = amplitude * gp_time

        if gp_wave is not None and amplitude is not None:
            interactions[f'{band}_gp_amp_wave'] = amplitude * gp_wave

    # ====================
    # 5. ASYMMETRY × COLOR
    # ====================
    # Rise/fade behavior correlates with spectral evolution
    for band in ['g', 'r']:
        rise = get_feat(f'{band}_rise_time')
        fade = get_feat(f'{band}_fade_time_50')
        color = get_feat(f'{band}_r_at_peak') if band == 'g' else get_feat('r_i_at_peak')

        if rise is not None and fade is not None:
            asymmetry = rise / (fade + 1)

            if color is not None:
                interactions[f'{band}_asym_x_color'] = asymmetry * color

    # ====================
    # 6. COLOR EVOLUTION INTERACTIONS
    # ====================
    # Color slopes × color values
    g_r_slope_50 = get_feat('g_r_slope_50d')
    g_r_slope_100 = get_feat('g_r_slope_100d')
    g_r_peak = get_feat('g_r_at_peak')

    if g_r_slope_50 is not None and g_r_peak is not None:
        # Initial color affects evolution rate
        interactions['gr_peak_x_slope50'] = g_r_peak * g_r_slope_50

    if g_r_slope_100 is not None and g_r_peak is not None:
        interactions['gr_peak_x_slope100'] = g_r_peak * g_r_slope_100

    # Color change acceleration
    if g_r_slope_50 is not None and g_r_slope_100 is not None:
        interactions['gr_color_accel'] = g_r_slope_100 - g_r_slope_50

    # ====================
    # 7. SKEWNESS × VARIABILITY
    # ====================
    # Shape (skewness) interacts with variability metrics
    for band in ['g', 'r', 'i']:
        skew = get_feat(f'{band}_skew')
        std = get_feat(f'{band}_std')

        if skew is not None and std is not None:
            interactions[f'{band}_skew_x_std'] = skew * std

    # ====================
    # 8. MULTI-BAND CORRELATIONS
    # ====================
    # Cross-band flux ratios × time features
    u_g_ratio = get_feat('u_g_peak_flux_ratio')
    g_r_ratio = get_feat('g_r_peak_flux_ratio')

    if u_g_ratio is not None and g_r_ratio is not None:
        # Combined UV-optical-red color space
        interactions['ug_x_gr_ratio'] = u_g_ratio * g_r_ratio

    # ====================
    # 9. POLYNOMIAL FEATURES (SELECTIVE)
    # ====================
    # Square key discriminative features
    for feat in ['r_skew', 'g_skew', 'flux_p25']:
        val = get_feat(feat)
        if val is not None:
            interactions[f'{feat}_squared'] = val ** 2

    # ====================
    # 10. RATIO FEATURES
    # ====================
    # Create informative ratios

    # Rise/fade ratios across bands
    g_rise = get_feat('g_rise_time')
    r_rise = get_feat('r_rise_time')
    if g_rise is not None and r_rise is not None:
        interactions['rise_ratio_g_r'] = g_rise / (r_rise + 1)

    g_fade = get_feat('g_fade_time_50')
    r_fade = get_feat('r_fade_time_50')
    if g_fade is not None and r_fade is not None:
        interactions['fade_ratio_g_r'] = g_fade / (r_fade + 1)

    # GP time/wave ratio
    if gp_time is not None and gp_wave is not None:
        interactions['gp_time_wave_ratio'] = gp_time / (gp_wave + 1e-6)

    return interactions


def select_top_interactions(
    X_train: pd.DataFrame,
    y: np.ndarray,
    original_feature_names: List[str],
    top_k: int = 30
) -> List[str]:
    """
    Select top K interaction features using simple univariate correlation.

    Args:
        X_train: Training features with interactions
        y: Target labels
        original_feature_names: Original feature names (to identify interactions)
        top_k: Number of top interactions to select

    Returns:
        List of top interaction feature names
    """
    from scipy.stats import pointbiserialr

    # Identify interaction columns
    interaction_cols = [c for c in X_train.columns if c not in original_feature_names]

    # Compute correlations
    correlations = []
    for col in interaction_cols:
        vals = X_train[col].values
        # Handle NaN
        valid = ~np.isnan(vals)
        if np.sum(valid) > 100:  # Need enough samples
            corr, pval = pointbiserialr(y[valid], vals[valid])
            correlations.append({
                'feature': col,
                'abs_corr': abs(corr),
                'pval': pval
            })

    # Sort by correlation
    correlations_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)

    # Select top K with p-value < 0.05
    significant = correlations_df[correlations_df['pval'] < 0.05]
    top_interactions = significant.head(top_k)['feature'].tolist()

    print(f"   Created {len(interaction_cols)} interactions, selected {len(top_interactions)} top features")

    return top_interactions


if __name__ == "__main__":
    # Test interaction creation
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Create dummy data
    np.random.seed(42)
    df = pd.DataFrame({
        'Z': np.random.uniform(0.1, 0.5, 100),
        'g_r_at_peak': np.random.uniform(-0.5, 1.0, 100),
        'temp_at_peak': np.random.uniform(10000, 40000, 100),
        'temp_post_50d': np.random.uniform(8000, 30000, 100),
        'r_peak_flux': np.random.uniform(10, 100, 100),
        'r_duration_50': np.random.uniform(20, 100, 100),
        'r_skew': np.random.uniform(-2, 2, 100),
        'g_skew': np.random.uniform(-2, 2, 100)
    })

    feature_names = df.columns.tolist()
    interactions = create_physics_interactions(df, feature_names)

    print(f"Original features: {len(feature_names)}")
    print(f"With interactions: {len(interactions.columns)}")
    print(f"\nSample interaction features:")
    new_features = [c for c in interactions.columns if c not in feature_names]
    print(new_features[:10])
