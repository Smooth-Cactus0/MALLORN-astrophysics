"""
High Signal-to-Noise Physics Features for MALLORN

Based on competition lead guidance: focus on physics-based features with high SNR.

New features not yet implemented:
1. Structure Function (SF) - KEY for AGN rejection (damped random walk signature)
2. Color-Magnitude Relation - "bluer when brighter" patterns
3. Flux Consistency Metrics - smoothness and stability indicators
4. Decline Consistency - how smooth is the decline across bands?
5. TDE t^-5/3 Deviation - explicit test against TDE power law

Physics rationale:
- TDEs: smooth power-law decline, stable colors, achromatic
- SNe: fast cooling, reddening, band-dependent evolution
- AGN: stochastic variability, damped random walk, "bluer when brighter"

References:
- Kelly+ 2009: AGN damped random walk
- MacLeod+ 2010: Structure function for AGN
- van Velzen+ 2020: TDE optical properties
- Gezari 2021: TDE review
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
import warnings

LSST_BANDS = ["u", "g", "r", "i", "z", "y"]


def compute_structure_function(obj_lc: pd.DataFrame,
                               bands: List[str] = ['r', 'g'],
                               tau_bins: List[float] = [1, 5, 10, 20, 50, 100]) -> Dict[str, float]:
    """
    Compute Structure Function SF(tau) at multiple timescales.

    SF(tau) = sqrt( mean( (m(t+tau) - m(t))^2 ) )

    This is a KEY discriminator for AGN:
    - AGN: SF follows damped random walk, increases with tau then flattens
    - TDE: SF dominated by systematic decline, not stochastic
    - SN: SF shows characteristic timescales of evolution

    Args:
        obj_lc: Lightcurve DataFrame
        bands: Bands to compute SF for
        tau_bins: Time lags in days

    Returns:
        SF values at each tau, SF slope, SF amplitude
    """
    features = {}

    for band in bands:
        band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_data) < 10:
            for tau in tau_bins:
                features[f'{band}_sf_tau_{tau}'] = np.nan
            features[f'{band}_sf_slope'] = np.nan
            features[f'{band}_sf_amplitude'] = np.nan
            features[f'{band}_sf_drw_tau'] = np.nan
            continue

        times = band_data['Time (MJD)'].values
        fluxes = band_data['Flux'].values

        # Convert to magnitudes for SF calculation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mags = -2.5 * np.log10(np.maximum(fluxes, 1e-10))

        sf_values = []
        valid_taus = []

        for tau in tau_bins:
            # Find pairs with time difference close to tau
            diffs_sq = []
            for i in range(len(times)):
                for j in range(i+1, len(times)):
                    dt = times[j] - times[i]
                    if tau * 0.5 < dt < tau * 1.5:  # Within 50% of target tau
                        dm = mags[j] - mags[i]
                        if np.isfinite(dm):
                            diffs_sq.append(dm**2)

            if len(diffs_sq) >= 3:
                sf = np.sqrt(np.mean(diffs_sq))
                features[f'{band}_sf_tau_{tau}'] = sf
                sf_values.append(sf)
                valid_taus.append(tau)
            else:
                features[f'{band}_sf_tau_{tau}'] = np.nan

        # SF slope (in log-log space)
        if len(sf_values) >= 3:
            log_tau = np.log10(valid_taus)
            log_sf = np.log10(np.maximum(sf_values, 1e-10))

            coeffs = np.polyfit(log_tau, log_sf, 1)
            features[f'{band}_sf_slope'] = coeffs[0]  # Power-law index
            features[f'{band}_sf_amplitude'] = 10**coeffs[1]  # Normalization

            # Estimate DRW timescale (where SF flattens)
            # For DRW: SF(tau) = sigma * sqrt(1 - exp(-tau/tau_drw))
            # Approximate: tau_drw ~ tau where SF starts to flatten
            if len(sf_values) >= 4:
                sf_arr = np.array(sf_values)
                tau_arr = np.array(valid_taus)
                # Find where derivative drops below threshold
                dsf = np.diff(sf_arr) / np.diff(tau_arr)
                if len(dsf) > 0 and np.any(dsf < 0.01):
                    flat_idx = np.argmax(dsf < 0.01)
                    features[f'{band}_sf_drw_tau'] = tau_arr[flat_idx]
                else:
                    features[f'{band}_sf_drw_tau'] = np.nan
            else:
                features[f'{band}_sf_drw_tau'] = np.nan
        else:
            features[f'{band}_sf_slope'] = np.nan
            features[f'{band}_sf_amplitude'] = np.nan
            features[f'{band}_sf_drw_tau'] = np.nan

    return features


def compute_color_magnitude_relation(obj_lc: pd.DataFrame) -> Dict[str, float]:
    """
    Compute the color-magnitude relation.

    "Bluer when brighter" (BWB) is a key AGN signature.
    TDEs and SNe show different color-magnitude patterns.

    Returns:
        Correlation, slope of color vs magnitude
    """
    features = {}

    g_data = obj_lc[obj_lc['Filter'] == 'g'].sort_values('Time (MJD)')
    r_data = obj_lc[obj_lc['Filter'] == 'r'].sort_values('Time (MJD)')

    if len(g_data) < 5 or len(r_data) < 5:
        features['color_mag_correlation'] = np.nan
        features['color_mag_slope'] = np.nan
        features['bwb_strength'] = np.nan
        features['color_mag_scatter'] = np.nan
        return features

    # Pair up observations
    colors = []
    r_mags = []

    for _, g_row in g_data.iterrows():
        t = g_row['Time (MJD)']
        g_flux = g_row['Flux']

        # Find closest r observation
        dt = np.abs(r_data['Time (MJD)'].values - t)
        min_idx = np.argmin(dt)

        if dt[min_idx] < 3 and g_flux > 0 and r_data.iloc[min_idx]['Flux'] > 0:
            r_flux = r_data.iloc[min_idx]['Flux']

            g_r = -2.5 * np.log10(g_flux / r_flux)
            r_mag = -2.5 * np.log10(r_flux)

            if np.isfinite(g_r) and np.isfinite(r_mag):
                colors.append(g_r)
                r_mags.append(r_mag)

    if len(colors) >= 5:
        colors = np.array(colors)
        r_mags = np.array(r_mags)

        # Correlation
        corr, _ = stats.pearsonr(r_mags, colors)
        features['color_mag_correlation'] = corr

        # Slope (negative = bluer when brighter)
        coeffs = np.polyfit(r_mags, colors, 1)
        features['color_mag_slope'] = coeffs[0]

        # BWB strength (how much color changes per mag)
        # AGN typically: -0.1 to -0.3
        # Negative = BWB (bluer when brighter)
        features['bwb_strength'] = -coeffs[0]  # Positive if BWB

        # Scatter around relation
        predicted = coeffs[0] * r_mags + coeffs[1]
        features['color_mag_scatter'] = np.std(colors - predicted)
    else:
        features['color_mag_correlation'] = np.nan
        features['color_mag_slope'] = np.nan
        features['bwb_strength'] = np.nan
        features['color_mag_scatter'] = np.nan

    return features


def compute_decline_consistency(obj_lc: pd.DataFrame,
                                bands: List[str] = ['g', 'r', 'i']) -> Dict[str, float]:
    """
    Compute how consistent the decline is across bands.

    TDEs: Achromatic decline - all bands fade together
    SNe: Chromatic - different bands fade at different rates
    AGN: No systematic decline

    Returns:
        Cross-band decline correlation, decline rate ratios
    """
    features = {}

    decline_rates = {}
    decline_residuals = {}

    for band in bands:
        band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_data) < 5:
            decline_rates[band] = np.nan
            decline_residuals[band] = np.nan
            continue

        times = band_data['Time (MJD)'].values
        fluxes = band_data['Flux'].values

        # Post-peak data
        peak_idx = np.argmax(fluxes)
        post_times = times[peak_idx:]
        post_fluxes = fluxes[peak_idx:]

        if len(post_times) >= 4 and post_fluxes[0] > 0:
            # Normalize to peak
            norm_flux = post_fluxes / post_fluxes[0]
            rel_time = post_times - post_times[0]

            # Linear decline rate
            valid = norm_flux > 0
            if np.sum(valid) >= 3:
                coeffs = np.polyfit(rel_time[valid], norm_flux[valid], 1)
                decline_rates[band] = coeffs[0]  # Flux decline per day

                # Smoothness (residual from linear decline)
                predicted = coeffs[0] * rel_time[valid] + coeffs[1]
                decline_residuals[band] = np.std(norm_flux[valid] - predicted)
            else:
                decline_rates[band] = np.nan
                decline_residuals[band] = np.nan
        else:
            decline_rates[band] = np.nan
            decline_residuals[band] = np.nan

    # Cross-band consistency
    valid_bands = [b for b in bands if not np.isnan(decline_rates.get(b, np.nan))]

    if len(valid_bands) >= 2:
        rates = [decline_rates[b] for b in valid_bands]
        residuals = [decline_residuals[b] for b in valid_bands if not np.isnan(decline_residuals.get(b, np.nan))]

        # Decline rate consistency (CV - low for TDEs)
        features['decline_rate_cv'] = np.std(rates) / np.abs(np.mean(rates)) if np.mean(rates) != 0 else np.nan

        # Average smoothness
        if len(residuals) >= 2:
            features['decline_smoothness_avg'] = np.mean(residuals)
        else:
            features['decline_smoothness_avg'] = np.nan

        # Specific ratios
        if 'g' in valid_bands and 'r' in valid_bands:
            features['decline_ratio_g_r'] = decline_rates['g'] / decline_rates['r'] if decline_rates['r'] != 0 else np.nan
        else:
            features['decline_ratio_g_r'] = np.nan

        if 'r' in valid_bands and 'i' in valid_bands:
            features['decline_ratio_r_i'] = decline_rates['r'] / decline_rates['i'] if decline_rates['i'] != 0 else np.nan
        else:
            features['decline_ratio_r_i'] = np.nan
    else:
        features['decline_rate_cv'] = np.nan
        features['decline_smoothness_avg'] = np.nan
        features['decline_ratio_g_r'] = np.nan
        features['decline_ratio_r_i'] = np.nan

    return features


def compute_tde_power_law_deviation(obj_lc: pd.DataFrame,
                                    bands: List[str] = ['r']) -> Dict[str, float]:
    """
    Explicitly test how well the lightcurve matches TDE power law.

    TDE theoretical prediction: L ~ t^(-5/3) for mass fallback rate
    Monochromatic optical: F ~ t^(-5/12) approximately

    Lower deviation = more TDE-like

    Returns:
        Deviation from t^-5/3, deviation from t^-5/12, best-fit power law
    """
    features = {}

    TDE_ALPHA_BOLOMETRIC = -5/3  # -1.667
    TDE_ALPHA_OPTICAL = -5/12   # -0.417

    for band in bands:
        band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_data) < 8:
            features[f'{band}_tde_deviation_53'] = np.nan
            features[f'{band}_tde_deviation_512'] = np.nan
            features[f'{band}_best_power_law'] = np.nan
            features[f'{band}_power_law_chi2'] = np.nan
            continue

        times = band_data['Time (MJD)'].values
        fluxes = band_data['Flux'].values

        # Post-peak only
        peak_idx = np.argmax(fluxes)
        peak_time = times[peak_idx]
        peak_flux = fluxes[peak_idx]

        post_mask = times > peak_time + 10  # Start 10 days after peak
        post_times = times[post_mask]
        post_fluxes = fluxes[post_mask]

        if len(post_times) >= 5 and peak_flux > 0:
            dt = post_times - peak_time

            valid = post_fluxes > 0
            if np.sum(valid) >= 4:
                log_dt = np.log10(dt[valid])
                log_flux = np.log10(post_fluxes[valid])

                # Fit power law
                coeffs = np.polyfit(log_dt, log_flux, 1)
                best_alpha = coeffs[0]
                features[f'{band}_best_power_law'] = best_alpha

                # Deviations from TDE predictions
                features[f'{band}_tde_deviation_53'] = np.abs(best_alpha - TDE_ALPHA_BOLOMETRIC)
                features[f'{band}_tde_deviation_512'] = np.abs(best_alpha - TDE_ALPHA_OPTICAL)

                # Chi-squared of fit
                predicted = coeffs[0] * log_dt + coeffs[1]
                chi2 = np.mean((log_flux - predicted)**2)
                features[f'{band}_power_law_chi2'] = chi2
            else:
                features[f'{band}_tde_deviation_53'] = np.nan
                features[f'{band}_tde_deviation_512'] = np.nan
                features[f'{band}_best_power_law'] = np.nan
                features[f'{band}_power_law_chi2'] = np.nan
        else:
            features[f'{band}_tde_deviation_53'] = np.nan
            features[f'{band}_tde_deviation_512'] = np.nan
            features[f'{band}_best_power_law'] = np.nan
            features[f'{band}_power_law_chi2'] = np.nan

    return features


def compute_flux_stability_metrics(obj_lc: pd.DataFrame,
                                   bands: List[str] = ['r', 'g']) -> Dict[str, float]:
    """
    Compute flux stability and consistency metrics.

    TDEs: Smooth evolution, low point-to-point scatter
    AGN: High stochastic variability
    SNe: Smooth but with characteristic bumps

    Returns:
        Point-to-point scatter, monotonicity, noise level
    """
    features = {}

    for band in bands:
        band_data = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

        if len(band_data) < 10:
            features[f'{band}_pt_scatter'] = np.nan
            features[f'{band}_monotonicity'] = np.nan
            features[f'{band}_noise_ratio'] = np.nan
            features[f'{band}_smooth_score'] = np.nan
            continue

        times = band_data['Time (MJD)'].values
        fluxes = band_data['Flux'].values
        errors = band_data['Flux_err'].values if 'Flux_err' in band_data.columns else np.ones_like(fluxes) * 0.1

        # Post-peak (declining phase)
        peak_idx = np.argmax(fluxes)
        post_fluxes = fluxes[peak_idx:]
        post_errors = errors[peak_idx:]

        if len(post_fluxes) >= 5:
            # Point-to-point scatter (normalized)
            diffs = np.diff(post_fluxes)
            mean_flux = np.mean(post_fluxes)
            if mean_flux > 0:
                features[f'{band}_pt_scatter'] = np.std(diffs) / mean_flux
            else:
                features[f'{band}_pt_scatter'] = np.nan

            # Monotonicity (fraction of points that decrease)
            n_decreasing = np.sum(diffs < 0)
            features[f'{band}_monotonicity'] = n_decreasing / len(diffs)

            # Noise ratio (observed scatter vs expected from errors)
            expected_scatter = np.sqrt(np.mean(post_errors**2))
            observed_scatter = np.std(diffs) / np.sqrt(2)  # Factor for differencing
            if expected_scatter > 0:
                features[f'{band}_noise_ratio'] = observed_scatter / expected_scatter
            else:
                features[f'{band}_noise_ratio'] = np.nan

            # Smooth score (how well a smoothed curve fits)
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(post_fluxes, size=3)
            residuals = post_fluxes - smoothed
            features[f'{band}_smooth_score'] = 1 - np.std(residuals) / np.std(post_fluxes) if np.std(post_fluxes) > 0 else np.nan
        else:
            features[f'{band}_pt_scatter'] = np.nan
            features[f'{band}_monotonicity'] = np.nan
            features[f'{band}_noise_ratio'] = np.nan
            features[f'{band}_smooth_score'] = np.nan

    return features


def extract_high_snr_features_single(obj_lc: pd.DataFrame) -> Dict[str, float]:
    """Extract all high-SNR physics features for a single object."""
    features = {}

    # Structure function (AGN discriminator)
    features.update(compute_structure_function(obj_lc))

    # Color-magnitude relation (BWB pattern)
    features.update(compute_color_magnitude_relation(obj_lc))

    # Decline consistency across bands
    features.update(compute_decline_consistency(obj_lc))

    # TDE power law deviation
    features.update(compute_tde_power_law_deviation(obj_lc))

    # Flux stability metrics
    features.update(compute_flux_stability_metrics(obj_lc))

    return features


def extract_high_snr_features(
    lightcurves: pd.DataFrame,
    object_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract high-SNR physics features for multiple objects.

    Args:
        lightcurves: DataFrame with lightcurve data
        object_ids: Optional list of object IDs

    Returns:
        DataFrame with high-SNR physics features
    """
    if object_ids is None:
        object_ids = lightcurves['object_id'].unique()

    # Pre-group for efficiency
    grouped = {obj_id: group for obj_id, group in lightcurves.groupby('object_id')}

    all_features = []

    for i, obj_id in enumerate(object_ids):
        if (i + 1) % 500 == 0:
            print(f"    High-SNR Physics: {i+1}/{len(object_ids)} objects processed")

        obj_lc = grouped.get(obj_id, pd.DataFrame())
        if obj_lc.empty:
            continue

        features = extract_high_snr_features_single(obj_lc)
        features['object_id'] = obj_id
        all_features.append(features)

    return pd.DataFrame(all_features)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.data_loader import load_all_data

    print("Loading data...")
    data = load_all_data()

    print("\nExtracting high-SNR features for first 20 objects...")
    sample_ids = data['train_meta']['object_id'].head(20).tolist()
    snr_features = extract_high_snr_features(data['train_lc'], sample_ids)

    print(f"\nExtracted {len(snr_features.columns)-1} high-SNR physics features")
    print("\nFeature columns:")
    for col in snr_features.columns:
        if col != 'object_id':
            print(f"  {col}")
    print("\nSample values:")
    print(snr_features[['object_id', 'r_sf_slope', 'color_mag_correlation', 'decline_rate_cv', 'r_tde_deviation_512']].head())
