"""
MALLORN: Lightcurve Visualization + Power Law Analysis

Part 1: Plot 20 random events per class (TDE, SN, AGN)
Part 2: Fit 10 different power law variations to decline phase
Part 3: Evaluate which power laws are significant features

Goal: Understand physics visually and find discriminative features
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN: Lightcurve Visualization + Power Law Analysis", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...", flush=True)

# Load metadata
train_meta = pd.read_csv(base_path / 'data/raw/train_log.csv')

# Load lightcurves from all splits
train_lcs = []
for i in range(1, 21):
    path = base_path / f'data/raw/split_{i:02d}/train_full_lightcurves.csv'
    if path.exists():
        train_lcs.append(pd.read_csv(path))
train_lc = pd.concat(train_lcs, ignore_index=True)

print(f"   Training objects: {len(train_meta)}", flush=True)
print(f"   Lightcurve points: {len(train_lc)}", flush=True)

# Group by class
# TDE: target=1, SN and AGN: target=0 but different SpecType
tde_objects = train_meta[train_meta['target'] == 1]['object_id'].tolist()
non_tde = train_meta[train_meta['target'] == 0]

# Get SNe and AGN from SpecType
sn_types = ['SN Ia', 'SN II', 'SN Ibc', 'SLSN', 'SN IIn']
agn_objects = non_tde[non_tde['SpecType'] == 'AGN']['object_id'].tolist()
sn_objects = non_tde[non_tde['SpecType'].isin(sn_types)]['object_id'].tolist()

print(f"\n   Class distribution:", flush=True)
print(f"   TDEs: {len(tde_objects)}", flush=True)
print(f"   SNe: {len(sn_objects)}", flush=True)
print(f"   AGN: {len(agn_objects)}", flush=True)

# ====================
# 2. VISUALIZATION: 20 EVENTS PER CLASS
# ====================
print("\n2. Creating visualizations...", flush=True)

# LSST band colors
band_colors = {
    'u': 'purple',
    'g': 'blue',
    'r': 'green',
    'i': 'orange',
    'z': 'red',
    'y': 'brown'
}

def plot_lightcurve(ax, obj_id, lc_data, title):
    """Plot multi-band lightcurve for a single object."""
    obj_lc = lc_data[lc_data['object_id'] == obj_id]

    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        band_lc = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')
        if len(band_lc) > 0:
            # Normalize time to start at 0
            t = band_lc['Time (MJD)'].values
            t = t - t.min()
            flux = band_lc['Flux'].values
            flux_err = band_lc['Flux_err'].values

            ax.errorbar(t, flux, yerr=flux_err, fmt='o-',
                       color=band_colors[band], label=band,
                       markersize=3, alpha=0.7, linewidth=0.5)

    ax.set_xlabel('Days since first obs')
    ax.set_ylabel('Flux (uJy)')
    ax.set_title(title, fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_class_examples(class_name, object_ids, lc_data, n_examples=20):
    """Plot n_examples lightcurves for a class in a grid."""
    np.random.seed(42)

    # Sample random objects
    sample_ids = np.random.choice(object_ids, min(n_examples, len(object_ids)), replace=False)

    # Create figure with 4x5 grid
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle(f'{class_name} Lightcurves (n={len(object_ids)})', fontsize=14)

    for idx, obj_id in enumerate(sample_ids):
        ax = axes[idx // 5, idx % 5]

        # Get spectral type for title
        spec_type = train_meta[train_meta['object_id'] == obj_id]['SpecType'].values
        spec_str = spec_type[0] if len(spec_type) > 0 else 'Unknown'

        plot_lightcurve(ax, obj_id, lc_data, f'{obj_id[:15]}...\n({spec_str})')

    # Add legend to first subplot
    axes[0, 0].legend(loc='upper right', fontsize=6)

    plt.tight_layout()

    # Save figure
    save_path = base_path / f'visualizations/{class_name.lower()}_examples.png'
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path.name}", flush=True)
    plt.close()

# Create visualizations
print("\n   Plotting TDE examples...", flush=True)
plot_class_examples('TDE', tde_objects, train_lc, n_examples=20)

print("   Plotting SN examples...", flush=True)
plot_class_examples('Supernova', sn_objects, train_lc, n_examples=20)

print("   Plotting AGN examples...", flush=True)
plot_class_examples('AGN', agn_objects, train_lc, n_examples=20)

# ====================
# 3. POWER LAW ANALYSIS
# ====================
print("\n3. Power Law Analysis...", flush=True)

# Define 10 different decay models
def powerlaw_5_3(t, A, t0):
    """TDE canonical: t^(-5/3)"""
    return A * np.power(np.maximum(t - t0, 0.1), -5/3)

def powerlaw_1(t, A, t0):
    """t^(-1)"""
    return A * np.power(np.maximum(t - t0, 0.1), -1)

def powerlaw_1_5(t, A, t0):
    """t^(-1.5)"""
    return A * np.power(np.maximum(t - t0, 0.1), -1.5)

def powerlaw_2(t, A, t0):
    """t^(-2)"""
    return A * np.power(np.maximum(t - t0, 0.1), -2)

def powerlaw_2_5(t, A, t0):
    """t^(-2.5)"""
    return A * np.power(np.maximum(t - t0, 0.1), -2.5)

def powerlaw_3(t, A, t0):
    """t^(-3)"""
    return A * np.power(np.maximum(t - t0, 0.1), -3)

def powerlaw_0_5(t, A, t0):
    """t^(-0.5) - slow decay"""
    return A * np.power(np.maximum(t - t0, 0.1), -0.5)

def exponential(t, A, tau, t0):
    """Exponential decay (SN-like)"""
    return A * np.exp(-np.maximum(t - t0, 0) / tau)

def linear(t, A, b, t0):
    """Linear decay"""
    return A - b * np.maximum(t - t0, 0)

def plateau_exp(t, A, tau, t_plateau, t0):
    """Plateau then exponential (SN II-like)"""
    dt = np.maximum(t - t0, 0)
    return np.where(dt < t_plateau, A, A * np.exp(-(dt - t_plateau) / tau))

# Models to test
MODELS = {
    'powerlaw_5_3': (powerlaw_5_3, ['A', 't0'], 'TDE canonical t^(-5/3)'),
    'powerlaw_1': (powerlaw_1, ['A', 't0'], 't^(-1)'),
    'powerlaw_1_5': (powerlaw_1_5, ['A', 't0'], 't^(-1.5)'),
    'powerlaw_2': (powerlaw_2, ['A', 't0'], 't^(-2)'),
    'powerlaw_2_5': (powerlaw_2_5, ['A', 't0'], 't^(-2.5)'),
    'powerlaw_3': (powerlaw_3, ['A', 't0'], 't^(-3)'),
    'powerlaw_0_5': (powerlaw_0_5, ['A', 't0'], 't^(-0.5) slow'),
    'exponential': (exponential, ['A', 'tau', 't0'], 'Exponential decay'),
    'linear': (linear, ['A', 'b', 't0'], 'Linear decay'),
}

def fit_decline_models(obj_id, lc_data, band='r'):
    """Fit all decline models to post-peak data for a single object."""
    obj_lc = lc_data[(lc_data['object_id'] == obj_id) & (lc_data['Filter'] == band)]

    if len(obj_lc) < 5:
        return {model: np.nan for model in MODELS}

    obj_lc = obj_lc.sort_values('Time (MJD)')
    t = obj_lc['Time (MJD)'].values
    flux = obj_lc['Flux'].values

    # Find peak
    peak_idx = np.argmax(flux)
    peak_time = t[peak_idx]
    peak_flux = flux[peak_idx]

    # Get post-peak data
    post_peak_mask = t > peak_time
    if np.sum(post_peak_mask) < 3:
        return {model: np.nan for model in MODELS}

    t_post = t[post_peak_mask] - peak_time  # Days since peak
    flux_post = flux[post_peak_mask]

    # Fit each model and compute residual
    results = {}

    for model_name, (model_func, params, desc) in MODELS.items():
        try:
            if len(params) == 2:  # A, t0
                popt, _ = curve_fit(model_func, t_post, flux_post,
                                   p0=[peak_flux, 0], maxfev=1000,
                                   bounds=([0, -10], [1e6, 10]))
            elif len(params) == 3:  # A, tau/b, t0
                if 'tau' in params:
                    popt, _ = curve_fit(model_func, t_post, flux_post,
                                       p0=[peak_flux, 30, 0], maxfev=1000,
                                       bounds=([0, 1, -10], [1e6, 500, 10]))
                else:
                    popt, _ = curve_fit(model_func, t_post, flux_post,
                                       p0=[peak_flux, 1, 0], maxfev=1000,
                                       bounds=([0, 0, -10], [1e6, 100, 10]))

            # Compute R^2
            pred = model_func(t_post, *popt)
            ss_res = np.sum((flux_post - pred) ** 2)
            ss_tot = np.sum((flux_post - np.mean(flux_post)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            results[model_name] = r2

        except Exception:
            results[model_name] = np.nan

    return results

def analyze_class(class_name, object_ids, lc_data, n_sample=100):
    """Analyze power law fits for a class."""
    np.random.seed(42)
    sample_ids = np.random.choice(object_ids, min(n_sample, len(object_ids)), replace=False)

    all_results = []
    for obj_id in sample_ids:
        result = fit_decline_models(obj_id, lc_data, band='r')
        result['object_id'] = obj_id
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    return results_df

print("\n   Analyzing TDEs...", flush=True)
tde_results = analyze_class('TDE', tde_objects, train_lc, n_sample=min(100, len(tde_objects)))

print("   Analyzing SNe...", flush=True)
sn_results = analyze_class('SN', sn_objects, train_lc, n_sample=100)

print("   Analyzing AGN...", flush=True)
agn_results = analyze_class('AGN', agn_objects, train_lc, n_sample=100)

# ====================
# 4. RESULTS SUMMARY
# ====================
print("\n" + "=" * 80, flush=True)
print("POWER LAW ANALYSIS RESULTS", flush=True)
print("=" * 80, flush=True)

print("\nMean R^2 by model and class:", flush=True)
print("-" * 70, flush=True)
print(f"{'Model':<20} {'TDE':>12} {'SN':>12} {'AGN':>12} {'TDE-SN diff':>12}", flush=True)
print("-" * 70, flush=True)

model_discriminative_power = {}

for model_name in MODELS.keys():
    tde_mean = tde_results[model_name].mean()
    sn_mean = sn_results[model_name].mean()
    agn_mean = agn_results[model_name].mean()

    # TDE vs SN difference (higher = more discriminative)
    diff = tde_mean - sn_mean if not (np.isnan(tde_mean) or np.isnan(sn_mean)) else 0
    model_discriminative_power[model_name] = diff

    print(f"{MODELS[model_name][2]:<20} {tde_mean:>12.3f} {sn_mean:>12.3f} {agn_mean:>12.3f} {diff:>+12.3f}", flush=True)

print("-" * 70, flush=True)

# Find most discriminative models
print("\n" + "=" * 80, flush=True)
print("MOST DISCRIMINATIVE POWER LAWS (TDE vs SN)", flush=True)
print("=" * 80, flush=True)

sorted_models = sorted(model_discriminative_power.items(), key=lambda x: abs(x[1]), reverse=True)
for i, (model, diff) in enumerate(sorted_models[:5], 1):
    direction = "TDE fits better" if diff > 0 else "SN fits better"
    print(f"   {i}. {MODELS[model][2]}: diff={diff:+.3f} ({direction})", flush=True)

# ====================
# 5. CREATE COMPARISON PLOT
# ====================
print("\n5. Creating comparison plot...", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (class_name, results) in zip(axes, [('TDE', tde_results), ('SN', sn_results), ('AGN', agn_results)]):
    means = [results[m].mean() for m in MODELS.keys()]
    stds = [results[m].std() for m in MODELS.keys()]
    labels = [MODELS[m][2][:15] for m in MODELS.keys()]

    x = np.arange(len(labels))
    ax.barh(x, means, xerr=stds, alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Mean R^2')
    ax.set_title(f'{class_name} (n={len(results)})')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = base_path / 'visualizations/powerlaw_comparison.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"   Saved: {save_path.name}", flush=True)
plt.close()

# ====================
# 6. SAVE FEATURES
# ====================
print("\n6. Extracting power law features for all training data...", flush=True)

def extract_powerlaw_features(obj_id, lc_data):
    """Extract power law R^2 values as features."""
    features = {'object_id': obj_id}

    for band in ['g', 'r', 'i']:  # Main optical bands
        results = fit_decline_models(obj_id, lc_data, band=band)
        for model_name, r2 in results.items():
            features[f'{band}_{model_name}_r2'] = r2

    return features

print("   Extracting features for all training objects...", flush=True)
all_features = []
train_ids = train_meta['object_id'].tolist()

for i, obj_id in enumerate(train_ids):
    if (i + 1) % 500 == 0:
        print(f"    Progress: {i+1}/{len(train_ids)}", flush=True)

    feat = extract_powerlaw_features(obj_id, train_lc)
    all_features.append(feat)

powerlaw_df = pd.DataFrame(all_features)

# Save
save_path = base_path / 'data/processed/powerlaw_features.pkl'
powerlaw_df.to_pickle(save_path)
print(f"   Saved: {save_path.name}", flush=True)
print(f"   Features: {len(powerlaw_df.columns) - 1}", flush=True)

print("\n" + "=" * 80, flush=True)
print("ANALYSIS COMPLETE", flush=True)
print("=" * 80, flush=True)
print("\nVisualization files:", flush=True)
print("   - visualizations/tde_examples.png", flush=True)
print("   - visualizations/supernova_examples.png", flush=True)
print("   - visualizations/agn_examples.png", flush=True)
print("   - visualizations/powerlaw_comparison.png", flush=True)
print("\nFeature file:", flush=True)
print("   - data/processed/powerlaw_features.pkl", flush=True)
print("=" * 80, flush=True)
