"""
MALLORN: Band-wise Peak Ordering Analysis

Physical Intuition:
- Different bands (u,g,r,i,z,y) = different wavelengths (blue to red)
- u,g = blue (hot emission)
- r,i = red
- z,y = near-infrared (cooler emission)

Questions:
1. Is there a consistent band ordering in peak times for TDE vs SN?
2. Do blue bands peak first (cooling source) or is there another pattern?
3. Can we use peak ordering as a discriminative feature?

Focus on CLEAR profiles (high SNR, single dominant peak) to see the true physics.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN: Band-wise Peak Ordering Analysis", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...", flush=True)

train_meta = pd.read_csv(base_path / 'data/raw/train_log.csv')

train_lcs = []
for i in range(1, 21):
    path = base_path / f'data/raw/split_{i:02d}/train_full_lightcurves.csv'
    if path.exists():
        train_lcs.append(pd.read_csv(path))
train_lc = pd.concat(train_lcs, ignore_index=True)

# Get TDE and SN objects
tde_ids = train_meta[train_meta['target'] == 1]['object_id'].tolist()
sn_types = ['SN Ia', 'SN II', 'SN Ibc', 'SLSN', 'SN IIn']
sn_ids = train_meta[train_meta['SpecType'].isin(sn_types)]['object_id'].tolist()

print(f"   TDEs: {len(tde_ids)}", flush=True)
print(f"   SNe: {len(sn_ids)}", flush=True)

# Band order (blue to red)
BANDS = ['u', 'g', 'r', 'i', 'z', 'y']
BAND_WAVELENGTHS = {'u': 354, 'g': 477, 'r': 623, 'i': 762, 'z': 913, 'y': 1004}  # nm

# ====================
# 2. IDENTIFY CLEAR PEAK PROFILES
# ====================
print("\n2. Identifying objects with clear peak profiles...", flush=True)

def get_peak_quality(obj_id, lc_data, band):
    """
    Assess if an object has a clear peak in a given band.
    Returns: (peak_time, peak_flux, quality_score) or (None, None, 0)

    Clear peak criteria:
    - Peak flux > 3x median absolute flux (prominence)
    - Single dominant peak (not multiple similar peaks)
    - At least 5 observations
    """
    band_lc = lc_data[(lc_data['object_id'] == obj_id) & (lc_data['Filter'] == band)]

    if len(band_lc) < 5:
        return None, None, 0

    band_lc = band_lc.sort_values('Time (MJD)')
    times = band_lc['Time (MJD)'].values
    flux = band_lc['Flux'].values

    # Find peaks
    peaks, properties = find_peaks(flux, prominence=0)

    if len(peaks) == 0:
        return None, None, 0

    # Get the highest peak
    peak_idx = peaks[np.argmax(flux[peaks])]
    peak_time = times[peak_idx]
    peak_flux = flux[peak_idx]

    # Quality metrics
    median_abs_flux = np.median(np.abs(flux))
    baseline = np.percentile(flux, 25)  # Lower quartile as baseline

    # Prominence: how much higher is peak vs baseline
    prominence = (peak_flux - baseline) / max(median_abs_flux, 0.1)

    # Is it a clear single peak? Check if second highest peak is much lower
    if len(peaks) > 1:
        sorted_peak_fluxes = sorted(flux[peaks], reverse=True)
        second_peak_ratio = sorted_peak_fluxes[1] / sorted_peak_fluxes[0] if sorted_peak_fluxes[0] > 0 else 1
    else:
        second_peak_ratio = 0  # Only one peak = very clear

    # Quality score (higher = clearer peak)
    # High prominence + low second peak ratio = clear profile
    quality = prominence * (1 - second_peak_ratio)

    return peak_time, peak_flux, quality

def analyze_object_peak_ordering(obj_id, lc_data, min_quality=2.0, min_bands=4):
    """
    Analyze peak ordering across bands for a single object.
    Returns dict with peak times per band, or None if not clear enough.
    """
    peak_data = {}

    for band in BANDS:
        peak_time, peak_flux, quality = get_peak_quality(obj_id, lc_data, band)
        if quality >= min_quality and peak_time is not None:
            peak_data[band] = {
                'time': peak_time,
                'flux': peak_flux,
                'quality': quality
            }

    # Need at least min_bands with clear peaks
    if len(peak_data) < min_bands:
        return None

    return peak_data

# Analyze TDEs
print("\n   Analyzing TDEs...", flush=True)
tde_peak_data = []
for obj_id in tde_ids:
    data = analyze_object_peak_ordering(obj_id, train_lc, min_quality=1.5, min_bands=3)
    if data:
        data['object_id'] = obj_id
        tde_peak_data.append(data)

print(f"   TDEs with clear profiles: {len(tde_peak_data)}/{len(tde_ids)}", flush=True)

# Analyze SNe
print("   Analyzing SNe...", flush=True)
sn_peak_data = []
for obj_id in sn_ids:
    data = analyze_object_peak_ordering(obj_id, train_lc, min_quality=1.5, min_bands=3)
    if data:
        data['object_id'] = obj_id
        sn_peak_data.append(data)

print(f"   SNe with clear profiles: {len(sn_peak_data)}/{len(sn_ids)}", flush=True)

# ====================
# 3. ANALYZE BAND ORDERING
# ====================
print("\n3. Analyzing Band Peak Ordering", flush=True)
print("=" * 80, flush=True)

def get_band_ordering(peak_data):
    """
    Get the ordering of bands by peak time.
    Returns list of bands sorted by peak time (earliest first).
    """
    bands_with_times = [(band, info['time']) for band, info in peak_data.items() if band in BANDS]
    bands_sorted = sorted(bands_with_times, key=lambda x: x[1])
    return [b[0] for b in bands_sorted]

def analyze_ordering_statistics(peak_data_list, class_name):
    """Analyze ordering statistics for a class."""

    # Track: which band peaks first, last, etc.
    first_band_counts = {b: 0 for b in BANDS}
    last_band_counts = {b: 0 for b in BANDS}

    # Track relative timing: blue vs red
    blue_first_count = 0  # u or g peaks before z or y
    red_first_count = 0

    # Track all orderings
    all_orderings = []

    # Track time differences between bands
    time_diffs = {f'{b1}_to_{b2}': [] for b1 in BANDS for b2 in BANDS if b1 != b2}

    for data in peak_data_list:
        ordering = get_band_ordering(data)
        all_orderings.append(ordering)

        if len(ordering) >= 2:
            first_band_counts[ordering[0]] += 1
            last_band_counts[ordering[-1]] += 1

        # Blue vs Red comparison
        blue_bands = set(['u', 'g']) & set(ordering)
        red_bands = set(['z', 'y']) & set(ordering)

        if blue_bands and red_bands:
            # Get earliest blue and red
            blue_times = [data[b]['time'] for b in blue_bands]
            red_times = [data[b]['time'] for b in red_bands]

            if min(blue_times) < min(red_times):
                blue_first_count += 1
            else:
                red_first_count += 1

        # Pairwise time differences
        for b1 in BANDS:
            for b2 in BANDS:
                if b1 != b2 and b1 in data and b2 in data:
                    diff = data[b2]['time'] - data[b1]['time']
                    time_diffs[f'{b1}_to_{b2}'].append(diff)

    print(f"\n{class_name} Peak Ordering Analysis (n={len(peak_data_list)})", flush=True)
    print("-" * 60, flush=True)

    print("\n   Which band peaks FIRST:", flush=True)
    for band in BANDS:
        pct = 100 * first_band_counts[band] / len(peak_data_list) if peak_data_list else 0
        bar = '*' * int(pct / 5)
        print(f"   {band}: {first_band_counts[band]:3d} ({pct:5.1f}%) {bar}", flush=True)

    print("\n   Which band peaks LAST:", flush=True)
    for band in BANDS:
        pct = 100 * last_band_counts[band] / len(peak_data_list) if peak_data_list else 0
        bar = '*' * int(pct / 5)
        print(f"   {band}: {last_band_counts[band]:3d} ({pct:5.1f}%) {bar}", flush=True)

    print(f"\n   Blue (u,g) vs Red (z,y) peak timing:", flush=True)
    total_br = blue_first_count + red_first_count
    if total_br > 0:
        print(f"   Blue peaks first: {blue_first_count} ({100*blue_first_count/total_br:.1f}%)", flush=True)
        print(f"   Red peaks first: {red_first_count} ({100*red_first_count/total_br:.1f}%)", flush=True)

    # Key time differences
    print(f"\n   Mean time differences (days):", flush=True)
    key_pairs = [('g', 'r'), ('r', 'i'), ('u', 'z'), ('g', 'i')]
    for b1, b2 in key_pairs:
        key = f'{b1}_to_{b2}'
        if time_diffs[key]:
            mean_diff = np.mean(time_diffs[key])
            std_diff = np.std(time_diffs[key])
            print(f"   {b1} -> {b2}: {mean_diff:+.1f} +/- {std_diff:.1f} days", flush=True)

    return {
        'first_band_counts': first_band_counts,
        'last_band_counts': last_band_counts,
        'blue_first': blue_first_count,
        'red_first': red_first_count,
        'time_diffs': time_diffs
    }

tde_stats = analyze_ordering_statistics(tde_peak_data, "TDE")
sn_stats = analyze_ordering_statistics(sn_peak_data, "SN")

# ====================
# 4. COMPARE TDE vs SN
# ====================
print("\n" + "=" * 80, flush=True)
print("COMPARISON: TDE vs SN Peak Ordering", flush=True)
print("=" * 80, flush=True)

print("\n   First-to-peak band comparison:", flush=True)
print(f"   {'Band':<6} {'TDE %':>10} {'SN %':>10} {'Difference':>12}", flush=True)
print("   " + "-" * 40, flush=True)

for band in BANDS:
    tde_pct = 100 * tde_stats['first_band_counts'][band] / len(tde_peak_data) if tde_peak_data else 0
    sn_pct = 100 * sn_stats['first_band_counts'][band] / len(sn_peak_data) if sn_peak_data else 0
    diff = tde_pct - sn_pct
    marker = "***" if abs(diff) > 10 else ""
    print(f"   {band:<6} {tde_pct:>10.1f} {sn_pct:>10.1f} {diff:>+12.1f} {marker}", flush=True)

print("\n   Blue vs Red timing:", flush=True)
tde_blue_pct = 100 * tde_stats['blue_first'] / (tde_stats['blue_first'] + tde_stats['red_first']) if (tde_stats['blue_first'] + tde_stats['red_first']) > 0 else 50
sn_blue_pct = 100 * sn_stats['blue_first'] / (sn_stats['blue_first'] + sn_stats['red_first']) if (sn_stats['blue_first'] + sn_stats['red_first']) > 0 else 50

print(f"   TDE: Blue first {tde_blue_pct:.1f}%, Red first {100-tde_blue_pct:.1f}%", flush=True)
print(f"   SN:  Blue first {sn_blue_pct:.1f}%, Red first {100-sn_blue_pct:.1f}%", flush=True)

# ====================
# 5. TIME DIFFERENCE ANALYSIS
# ====================
print("\n" + "=" * 80, flush=True)
print("TIME DIFFERENCE ANALYSIS (g -> r, the most common bands)", flush=True)
print("=" * 80, flush=True)

key = 'g_to_r'
if tde_stats['time_diffs'][key] and sn_stats['time_diffs'][key]:
    tde_g_to_r = np.array(tde_stats['time_diffs'][key])
    sn_g_to_r = np.array(sn_stats['time_diffs'][key])

    print(f"\n   g -> r peak time difference:", flush=True)
    print(f"   TDE: {np.mean(tde_g_to_r):+.1f} +/- {np.std(tde_g_to_r):.1f} days (median: {np.median(tde_g_to_r):+.1f})", flush=True)
    print(f"   SN:  {np.mean(sn_g_to_r):+.1f} +/- {np.std(sn_g_to_r):.1f} days (median: {np.median(sn_g_to_r):+.1f})", flush=True)

    # Statistical test
    from scipy.stats import mannwhitneyu
    stat, pval = mannwhitneyu(tde_g_to_r, sn_g_to_r, alternative='two-sided')
    print(f"\n   Mann-Whitney U test p-value: {pval:.4f}", flush=True)
    if pval < 0.05:
        print("   -> SIGNIFICANT difference in g->r timing between TDE and SN!", flush=True)
    else:
        print("   -> No significant difference", flush=True)

# Also check u -> i (wider wavelength span)
key = 'u_to_i'
if tde_stats['time_diffs'][key] and sn_stats['time_diffs'][key]:
    tde_u_to_i = np.array(tde_stats['time_diffs'][key])
    sn_u_to_i = np.array(sn_stats['time_diffs'][key])

    print(f"\n   u -> i peak time difference:", flush=True)
    print(f"   TDE: {np.mean(tde_u_to_i):+.1f} +/- {np.std(tde_u_to_i):.1f} days (median: {np.median(tde_u_to_i):+.1f})", flush=True)
    print(f"   SN:  {np.mean(sn_u_to_i):+.1f} +/- {np.std(sn_u_to_i):.1f} days (median: {np.median(sn_u_to_i):+.1f})", flush=True)

    stat, pval = mannwhitneyu(tde_u_to_i, sn_u_to_i, alternative='two-sided')
    print(f"\n   Mann-Whitney U test p-value: {pval:.4f}", flush=True)
    if pval < 0.05:
        print("   -> SIGNIFICANT difference in u->i timing between TDE and SN!", flush=True)

# ====================
# 6. CREATE VISUALIZATION
# ====================
print("\n6. Creating visualization...", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: First-to-peak distribution
ax1 = axes[0, 0]
x = np.arange(len(BANDS))
width = 0.35
tde_first = [tde_stats['first_band_counts'][b] / len(tde_peak_data) * 100 for b in BANDS]
sn_first = [sn_stats['first_band_counts'][b] / len(sn_peak_data) * 100 for b in BANDS]

ax1.bar(x - width/2, tde_first, width, label='TDE', color='blue', alpha=0.7)
ax1.bar(x + width/2, sn_first, width, label='SN', color='red', alpha=0.7)
ax1.set_xlabel('Band')
ax1.set_ylabel('% of objects')
ax1.set_title('Which band peaks FIRST')
ax1.set_xticks(x)
ax1.set_xticklabels(BANDS)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Last-to-peak distribution
ax2 = axes[0, 1]
tde_last = [tde_stats['last_band_counts'][b] / len(tde_peak_data) * 100 for b in BANDS]
sn_last = [sn_stats['last_band_counts'][b] / len(sn_peak_data) * 100 for b in BANDS]

ax2.bar(x - width/2, tde_last, width, label='TDE', color='blue', alpha=0.7)
ax2.bar(x + width/2, sn_last, width, label='SN', color='red', alpha=0.7)
ax2.set_xlabel('Band')
ax2.set_ylabel('% of objects')
ax2.set_title('Which band peaks LAST')
ax2.set_xticks(x)
ax2.set_xticklabels(BANDS)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: g->r time difference histogram
ax3 = axes[1, 0]
if tde_stats['time_diffs']['g_to_r'] and sn_stats['time_diffs']['g_to_r']:
    ax3.hist(tde_stats['time_diffs']['g_to_r'], bins=20, alpha=0.7, label='TDE', color='blue', density=True)
    ax3.hist(sn_stats['time_diffs']['g_to_r'], bins=20, alpha=0.7, label='SN', color='red', density=True)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('g -> r peak time (days)')
    ax3.set_ylabel('Density')
    ax3.set_title('g-band to r-band peak delay')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# Plot 4: u->i time difference histogram
ax4 = axes[1, 1]
if tde_stats['time_diffs']['u_to_i'] and sn_stats['time_diffs']['u_to_i']:
    ax4.hist(tde_stats['time_diffs']['u_to_i'], bins=20, alpha=0.7, label='TDE', color='blue', density=True)
    ax4.hist(sn_stats['time_diffs']['u_to_i'], bins=20, alpha=0.7, label='SN', color='red', density=True)
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('u -> i peak time (days)')
    ax4.set_ylabel('Density')
    ax4.set_title('u-band to i-band peak delay')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
save_path = base_path / 'visualizations/peak_ordering_analysis.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"   Saved: {save_path.name}", flush=True)
plt.close()

# ====================
# 7. SUMMARY
# ====================
print("\n" + "=" * 80, flush=True)
print("SUMMARY: Key Findings for Feature Engineering", flush=True)
print("=" * 80, flush=True)

print("""
   POTENTIAL DISCRIMINATIVE FEATURES:

   1. Band peak ordering features:
      - first_peak_band: which band peaks first (categorical or one-hot)
      - last_peak_band: which band peaks last

   2. Peak time differences:
      - g_to_r_peak_delay: time between g and r peak
      - u_to_i_peak_delay: time between u and i peak
      - blue_to_red_delay: earliest blue - earliest red peak

   3. Peak ordering consistency:
      - is_blue_first: binary, 1 if blue bands peak before red
      - peak_order_score: how much the ordering follows blue->red pattern

   These features capture the PHYSICS:
   - SNe cool as they expand -> blue peaks first, then red
   - TDEs have sustained heating -> different pattern expected
""", flush=True)

print("=" * 80, flush=True)
