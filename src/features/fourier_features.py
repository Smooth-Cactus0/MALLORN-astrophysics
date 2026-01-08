"""
Fourier Features for MALLORN Competition

Extract frequency-domain features from lightcurves to detect periodicity.

Key insight: AGN often show quasi-periodic variability while TDEs and SNe
are aperiodic. Fourier transform reveals this in frequency space.

Based on PLAsTiCC 1st & 2nd place solutions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

def extract_fourier_features_single_band(times: np.ndarray,
                                         fluxes: np.ndarray,
                                         band: str) -> Dict[str, float]:
    """
    Extract Fourier features from a single-band lightcurve.

    Args:
        times: Time array (MJD)
        fluxes: Flux array
        band: Filter name (u, g, r, i, z, y)

    Returns:
        Dictionary of Fourier features
    """
    features = {}
    prefix = f'{band}_'

    # Need at least 10 points for meaningful FFT
    if len(times) < 10 or len(fluxes) < 10:
        features[f'{prefix}fourier_dominant_freq'] = np.nan
        features[f'{prefix}fourier_dominant_power'] = np.nan
        features[f'{prefix}fourier_power_ratio'] = np.nan
        features[f'{prefix}fourier_spectral_entropy'] = np.nan
        return features

    # Remove NaN values
    mask = np.isfinite(fluxes) & np.isfinite(times)
    times_clean = times[mask]
    fluxes_clean = fluxes[mask]

    if len(times_clean) < 10:
        features[f'{prefix}fourier_dominant_freq'] = np.nan
        features[f'{prefix}fourier_dominant_power'] = np.nan
        features[f'{prefix}fourier_power_ratio'] = np.nan
        features[f'{prefix}fourier_spectral_entropy'] = np.nan
        return features

    try:
        # 1. Interpolate to uniform time grid
        # FFT requires evenly spaced samples
        t_min, t_max = times_clean.min(), times_clean.max()
        n_samples = min(len(times_clean), 128)  # Cap at 128 for speed
        t_uniform = np.linspace(t_min, t_max, n_samples)

        # Linear interpolation
        flux_uniform = np.interp(t_uniform, times_clean, fluxes_clean)

        # 2. Remove mean (DC component)
        flux_centered = flux_uniform - np.mean(flux_uniform)

        # 3. Apply Hanning window to reduce spectral leakage
        window = np.hanning(len(flux_centered))
        flux_windowed = flux_centered * window

        # 4. Compute FFT
        fft_result = np.fft.fft(flux_windowed)
        power_spectrum = np.abs(fft_result) ** 2

        # Only keep positive frequencies (first half)
        n_freq = len(power_spectrum) // 2
        power_spectrum = power_spectrum[:n_freq]

        # Frequency array (Hz, but really 1/days)
        dt = (t_max - t_min) / (n_samples - 1)  # Sampling interval in days
        frequencies = np.fft.fftfreq(len(flux_windowed), d=dt)[:n_freq]

        # Skip DC component (first frequency)
        if len(frequencies) > 1:
            frequencies = frequencies[1:]
            power_spectrum = power_spectrum[1:]

        if len(power_spectrum) == 0 or np.max(power_spectrum) == 0:
            features[f'{prefix}fourier_dominant_freq'] = np.nan
            features[f'{prefix}fourier_dominant_power'] = np.nan
            features[f'{prefix}fourier_power_ratio'] = np.nan
            features[f'{prefix}fourier_spectral_entropy'] = np.nan
            return features

        # 5. Extract features

        # Dominant frequency (peak in power spectrum)
        dominant_idx = np.argmax(power_spectrum)
        dominant_freq = abs(frequencies[dominant_idx])
        dominant_power = power_spectrum[dominant_idx]

        # Power ratio (peak / mean power)
        mean_power = np.mean(power_spectrum)
        power_ratio = dominant_power / (mean_power + 1e-10)

        # Spectral entropy (measure of randomness)
        # Normalize power spectrum to probability distribution
        power_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)
        # Remove zeros to avoid log issues
        power_norm_nz = power_norm[power_norm > 1e-10]
        spectral_entropy = -np.sum(power_norm_nz * np.log2(power_norm_nz + 1e-10))

        # Normalize entropy by max possible (uniform distribution)
        max_entropy = np.log2(len(power_norm_nz))
        if max_entropy > 0:
            spectral_entropy = spectral_entropy / max_entropy

        features[f'{prefix}fourier_dominant_freq'] = dominant_freq
        features[f'{prefix}fourier_dominant_power'] = dominant_power
        features[f'{prefix}fourier_power_ratio'] = power_ratio
        features[f'{prefix}fourier_spectral_entropy'] = spectral_entropy

    except Exception as e:
        # If anything fails, return NaN
        features[f'{prefix}fourier_dominant_freq'] = np.nan
        features[f'{prefix}fourier_dominant_power'] = np.nan
        features[f'{prefix}fourier_power_ratio'] = np.nan
        features[f'{prefix}fourier_spectral_entropy'] = np.nan

    return features


def extract_fourier_features(lc_df: pd.DataFrame,
                             object_ids: List[str],
                             verbose: bool = True) -> pd.DataFrame:
    """
    Extract Fourier features for all objects.

    Args:
        lc_df: Lightcurve DataFrame (object_id, Time (MJD), Flux, Filter)
        object_ids: List of object IDs to process
        verbose: Print progress

    Returns:
        DataFrame with object_id + Fourier features (24 features: 4 per band × 6 bands)
    """
    all_features = []

    for i, obj_id in enumerate(object_ids):
        if verbose and (i + 1) % 500 == 0:
            print(f"    Fourier: {i+1}/{len(object_ids)} objects processed", flush=True)

        obj_lc = lc_df[lc_df['object_id'] == obj_id]

        if len(obj_lc) == 0:
            # Object not found - create empty features
            features = {'object_id': obj_id}
            for band in ['u', 'g', 'r', 'i', 'z', 'y']:
                features[f'{band}_fourier_dominant_freq'] = np.nan
                features[f'{band}_fourier_dominant_power'] = np.nan
                features[f'{band}_fourier_power_ratio'] = np.nan
                features[f'{band}_fourier_spectral_entropy'] = np.nan
            all_features.append(features)
            continue

        # Extract per-band features
        features = {'object_id': obj_id}

        for band in ['u', 'g', 'r', 'i', 'z', 'y']:
            band_lc = obj_lc[obj_lc['Filter'] == band]

            if len(band_lc) >= 10:
                times = band_lc['Time (MJD)'].values
                fluxes = band_lc['Flux'].values

                band_features = extract_fourier_features_single_band(times, fluxes, band)
                features.update(band_features)
            else:
                # Not enough data for this band
                features[f'{band}_fourier_dominant_freq'] = np.nan
                features[f'{band}_fourier_dominant_power'] = np.nan
                features[f'{band}_fourier_power_ratio'] = np.nan
                features[f'{band}_fourier_spectral_entropy'] = np.nan

        all_features.append(features)

    return pd.DataFrame(all_features)
