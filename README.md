# MALLORN Astronomical Classification Challenge

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Author**: Alexy Louis
**Competition**: [MALLORN Astronomical Classification Challenge](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)
**Best Score**: LB F1 = 0.6907 (v34a Bazin parametric features)
**Deadline**: January 30, 2026

---

## Overview

This repository contains my solution for the MALLORN Astronomical Classification Challenge, which aims to classify nuclear transients into three categories:

| Class | Description |
|-------|-------------|
| **TDE** | Tidal Disruption Events - stars torn apart by supermassive black holes |
| **Supernova** | Stellar explosions (Types Ia, II, IIn, Ib/c, SLSN) |
| **AGN** | Active Galactic Nuclei - persistently accreting black holes |

The challenge uses simulated LSST (Vera C. Rubin Observatory) multi-band lightcurves with only 30% labeled training data.

### Key Statistics
- **Total objects**: 10,178 simulated lightcurves
- **Training set**: 3,054 objects (30%) with labels
- **Test set**: 7,124 objects (70%) for evaluation
- **Bands**: u, g, r, i, z, y (LSST filters)
- **Class imbalance**: Only 64 TDEs (~2.1%) in training set

---

## Results Summary

| Version | OOF F1 | LB F1 | Approach | Key Innovation |
|---------|--------|-------|----------|----------------|
| v1 | 0.30 | 0.333 | XGBoost baseline | Statistical features only |
| v2 | 0.51 | 0.499 | +Color features | g-r, r-i colors at peak |
| v8 | 0.6262 | 0.6481 | Tuned ensemble | Hyperparameter optimization |
| v19 | 0.6626 | 0.6649 | Multi-band GP | 2D Gaussian Process kernel |
| v21 | 0.6708 | 0.6649 | XGBoost only | Simplified pipeline |
| **v34a** | 0.6667 | **0.6907** | **Bazin fitting** | **Parametric lightcurve model** |
| v60a | 0.6815 | ~0.69 | Two-stage classifier | AGN filter + TDE classifier |
| v61a | - | 0.6811 | Enhanced two-stage | Additional physics features |

See [BENCHMARKS.md](BENCHMARKS.md) for complete version history.

---

## Project Structure

```
MALLORN-astrophysics/
├── README.md                 # This file
├── BENCHMARKS.md             # Complete results history
├── CLAUDE.md                 # AI assistant context
├── PODIUM_ROADMAP.md         # Strategy for top 3 finish
├── TWO_STAGE_ROADMAP.md      # Two-stage classifier details
│
├── src/                      # Source code
│   ├── features/             # Feature engineering modules
│   │   ├── statistical.py    # Basic statistics per band
│   │   ├── colors.py         # Multi-band color features
│   │   ├── lightcurve_shape.py # Rise/decline times
│   │   ├── gaussian_process.py # GP interpolation
│   │   ├── multiband_gp.py   # 2D GP with wavelength kernel
│   │   ├── physics_based.py  # Temperature, blackbody fits
│   │   ├── tde_physics.py    # TDE-specific features
│   │   ├── cesium_features.py # Astronomy variability features
│   │   ├── fourier_features.py # FFT-based features
│   │   └── ...
│   │
│   ├── models/               # Model architectures
│   │   ├── lstm_classifier.py # LSTM for raw lightcurves
│   │   ├── transformer_classifier.py # Self-attention model
│   │   ├── atat.py           # ATAT transfer learning
│   │   └── focal_loss.py     # Custom loss for imbalance
│   │
│   └── utils/                # Utilities
│       └── data_loader.py    # Data loading pipeline
│
├── scripts/                  # Training scripts
│   ├── train_baseline.py     # v1: Initial baseline
│   ├── train_v2_colors.py    # v2: Color features
│   ├── train_v8_tuned.py     # v8: Tuned ensemble
│   ├── train_v19_multiband_gp.py # v19: Multi-band GP
│   ├── train_v34a_bazin.py   # v34a: Bazin fitting (BEST)
│   ├── train_v60_two_stage.py # v60: Two-stage approach
│   └── ...                   # 60+ experiment versions
│
├── visualizations/           # Analysis plots
│   ├── tde_examples.png      # Example TDE lightcurves
│   ├── supernova_examples.png # Example SN lightcurves
│   ├── agn_examples.png      # Example AGN lightcurves
│   └── powerlaw_comparison.png # Power law fits
│
├── data/                     # Data directory (gitignored)
│   ├── raw/                  # Original Kaggle data
│   └── processed/            # Feature caches
│
└── submissions/              # Submission files (gitignored)
```

---

## Methodology

### 1. Feature Engineering (The Key to Success)

The breakthrough came from physics-informed feature engineering:

#### Bazin Parametric Fitting (v34a - Best Model)
```python
# Bazin function models supernova-like lightcurves
f(t) = A * exp(-(t-t0)/τ_fall) / (1 + exp(-(t-t0)/τ_rise)) + B
```
- 8 parameters per band: A, t0, τ_rise, τ_fall, B, chi2, rise_fall_ratio, peak_flux
- 52 total features (6 bands × 8 + 4 cross-band)
- **+2.58% LB improvement** over baseline

#### Multi-band Gaussian Process (v19)
```python
# 2D kernel captures both time and wavelength correlations
k(t1, t2, λ1, λ2) = k_time(t1, t2) × k_wavelength(λ1, λ2)
```
- Matérn-3/2 kernel (as recommended in 2025 TDE paper)
- Enables interpolation of sparse lightcurves
- **+5.8% OOF improvement**

#### Color Features (v2)
```python
# TDEs stay hot (blue), SNe cool rapidly (redden)
g_minus_r_at_peak
g_minus_r_post_20d, _50d, _100d
color_evolution_rate = (g_r_post - g_r_peak) / delta_time
```
- **41.5% of total model importance**
- Physics: TDEs maintain ~20,000-40,000K from continuous accretion heating

### 2. Model Selection

| Model Type | Best OOF F1 | Notes |
|------------|-------------|-------|
| **XGBoost** | 0.6708 | Best performer, handles missing values |
| LightGBM | 0.65 | Faster training, similar performance |
| CatBoost | 0.62 | Good with categorical features |
| LSTM | 0.12 | Failed on raw lightcurves |
| Transformer | 0.11 | Self-attention didn't help |
| ATAT | 0.50 | Transfer learning improved DL |

**Key Learning**: Gradient boosting on engineered features dramatically outperforms deep learning on raw sequences for this problem.

### 3. Two-Stage Classifier (v60)

```
Stage 1: AGN Filter (97.4% accuracy)
    └── Remove objects with AGN_prob >= 0.99
        └── Only loses 4 TDEs (2.7%)

Stage 2: TDE vs Rest (on filtered set)
    └── Focus on TDE vs SN discrimination
        └── Main bottleneck: 43 TDEs missed (29.9% FN)
```

### 4. Handling Class Imbalance

With only 64 TDEs (2.1%) in training:
- Optimal threshold: ~0.07 (not 0.5!)
- Scale_pos_weight in XGBoost
- SMOTE attempted but hurt generalization
- Two-stage filtering to reduce negatives

---

## Installation

```bash
# Clone the repository
git clone https://github.com/alexylou/MALLORN-astrophysics.git
cd MALLORN-astrophysics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.7.0
lightgbm>=3.3.0
catboost>=1.1.0
scipy>=1.9.0
george>=0.4.0          # Gaussian Process
torch>=2.0.0           # For neural network models
matplotlib>=3.5.0
seaborn>=0.12.0
```

---

## Usage

### Training the Best Model (v34a)

```bash
python scripts/train_v34a_bazin.py
```

This will:
1. Load and preprocess lightcurves
2. Fit Bazin parametric model to each band
3. Extract 52 Bazin features + existing features
4. Train 5-fold XGBoost with optimized hyperparameters
5. Generate submission file

### Quick Start with Pre-computed Features

```python
from src.features.statistical import extract_statistical_features
from src.features.colors import extract_color_features
from src.features.gaussian_process import extract_gp_features

# Load your lightcurve data
features = pd.concat([
    extract_statistical_features(lightcurves),
    extract_color_features(lightcurves),
    extract_gp_features(lightcurves)
], axis=1)
```

---

## Key Insights

### What Worked
1. **Physics-informed features** - Color evolution, Bazin fitting, GP interpolation
2. **Gradient boosting** - XGBoost/LightGBM on engineered features
3. **Multi-band modeling** - 2D GP kernel for cross-band correlations
4. **Aggressive threshold tuning** - 0.07-0.30 instead of 0.5

### What Didn't Work
1. **Deep learning on raw sequences** - LSTM/Transformer achieved only F1~0.12
2. **External data (PLAsTiCC)** - Domain shift hurt performance
3. **SMOTE oversampling** - Synthetic samples didn't generalize
4. **Too many features** - 375 features performed worse than 172

### Lessons Learned
1. **Domain knowledge beats generic ML** - Understanding TDE physics led to best features
2. **Simpler is often better** - Single XGBoost beat complex ensembles
3. **Class imbalance is critical** - 2.1% positive rate requires careful handling
4. **Feature selection matters** - More features ≠ better performance

---

## References

### Competition Resources
- [MALLORN Kaggle Competition](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)
- [ArXiv Paper](https://arxiv.org/abs/2512.04946)
- [Data Production Colab](https://colab.research.google.com/drive/1oy96r29Zs4U5Hl-THsZPCnOQbuz21hl5)

### Related Work
- [PLAsTiCC Solutions](https://github.com/kozodoi/Kaggle_Astronomical_Classification) - 1st place used single LightGBM
- [2025 TDE Paper](https://arxiv.org/abs/2509.25902) - GP features, post-peak colors
- [ASTROMER](https://github.com/astromer-science/python-library) - Pre-trained embeddings
- [ATAT](https://github.com/alercebroker/ATAT) - Time series + tabular transformer

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- MALLORN competition organizers for the fascinating dataset
- PLAsTiCC competition winners for inspiring the feature engineering approach
- Claude Code for pair programming assistance
