# MALLORN Astronomical Classification Challenge

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)
[![Result](https://img.shields.io/badge/Result-4th%20%2F%20894-gold)](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/leaderboard)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Author**: Alexy Louis
**Final Result**: **4th / 894** on private leaderboard (Top 0.5%)
**Best Private LB Score**: F1 = 0.6684 (v92d XGBoost + Adversarial Validation)
**Competition**: [MALLORN Astronomical Classification Challenge](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)

---

## Overview

This repository contains my solution for the MALLORN Astronomical Classification Challenge, a binary classification task to identify **Tidal Disruption Events (TDEs)** among nuclear transients in simulated LSST lightcurves.

| Class | Description |
|-------|-------------|
| **TDE** | Tidal Disruption Events -- stars torn apart by supermassive black holes |
| **Non-TDE** | Supernovae (Types Ia, II, IIn, Ib/c, SLSN) + Active Galactic Nuclei (AGN) |

### Key Statistics
- **Total objects**: 10,178 simulated lightcurves
- **Training set**: 3,054 objects (148 TDE, 2,906 non-TDE)
- **Test set**: 7,124 objects
- **Bands**: u, g, r, i, z, y (LSST filters)
- **Extreme class imbalance**: ~4.8% TDE in training
- **Metric**: F1 Score

For a detailed breakdown of the winning approach, see [SOLUTION.md](SOLUTION.md).

---

## Final Results

### Selected Submissions (Private Leaderboard)

| Submission | Private F1 | Public F1 | Model | Technique |
|------------|-----------|-----------|-------|-----------|
| **v92d_baseline_adv** | **0.6684** | **0.6986** | XGBoost | **Adversarial validation weights** |
| v115c_extended_research | 0.6757 | 0.6840 | XGBoost | Adversarial weights + research features |
| v55_powerlaw | 0.6737 | 0.6873 | XGBoost | Power law decay features |
| v42_pseudolabel | 0.6735 | 0.6666 | XGBoost | Conservative pseudo-labeling |
| v104_seed_ensemble | 0.6700 | 0.6811 | XGBoost | 10-seed ensemble averaging |

### Competition Timeline

| Phase | Best Model | Public F1 | Key Innovation |
|-------|-----------|-----------|----------------|
| 1. Baseline (Dec 25-26) | v8 | 0.6481 | Statistical features + tuned ensemble |
| 2. Deep Learning (Dec 26-27) | LSTM | 0.12 | Failed -- feature engineering essential |
| 3. Gaussian Process (Dec 27-28) | v19 | 0.6649 | 2D multi-band GP kernel |
| 4. Bazin Fitting (Dec 29-Jan 5) | v34a | 0.6907 | Parametric lightcurve model |
| 5. Physics Features (Jan 5-15) | v55 | 0.6873 | Power law decay fitting |
| 6. Adversarial Validation (Jan 15-20) | v92d | 0.6986 | Train/test distribution alignment |
| 7. Final Optimization (Jan 20-29) | v115c | 0.6840 | Extended research features |

See [BENCHMARKS.md](BENCHMARKS.md) for the complete 126-version experiment history.

---

## Winning Solution: v92d (Adversarial Validation + XGBoost)

### The Core Insight

The key challenge in this competition was a **train-test distribution shift** -- the training and test sets were not drawn from the same distribution. Models that scored high on out-of-fold (OOF) cross-validation consistently performed **worse** on the leaderboard:

| Model | OOF F1 | Public LB F1 | Delta |
|-------|--------|-------------|-------|
| v92d (winner) | 0.6688 | 0.6986 | **+0.030** |
| v34a | 0.6667 | 0.6907 | +0.024 |
| v80a (overfit) | 0.7118 | 0.6666 | -0.045 |
| v108b (worst) | 0.6925 | 0.6325 | -0.060 |

**Higher OOF often meant worse LB.** The solution was adversarial validation.

### How Adversarial Validation Works

1. **Train a classifier** to distinguish train samples from test samples
2. **Extract prediction probabilities** -- samples the classifier is confident are "train-like" have high adversarial scores
3. **Down-weight train-only samples** -- reduce their influence during model training
4. **Up-weight test-like samples** -- focus the model on patterns present in both distributions

```python
# Adversarial weights: down-weight samples that look different from test
with open('data/processed/adversarial_validation.pkl', 'rb') as f:
    adv_results = pickle.load(f)
sample_weights = adv_results['sample_weights']
# weights range: [0.17, 1.93] -- test-like samples get ~2x weight
```

### Model Architecture

```python
xgb_params = {
    'objective': 'binary:logistic',
    'max_depth': 5,
    'learning_rate': 0.025,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'scale_pos_weight': 19.56,  # class imbalance ratio
    'tree_method': 'hist',
    'random_state': 42,
}
# 5-fold StratifiedKFold, single seed, 500 rounds with early stopping
```

**Notably simple**: no Optuna tuning, no multi-seed averaging, no ensemble. The adversarial weights alone bridged the train-test gap.

### Feature Set (222 features)

The v34a feature set (also used by v92d) consists of physics-informed features:

| Category | Count | Key Features |
|----------|-------|-------------|
| **Base statistical** | ~120 | Per-band flux statistics, skewness, kurtosis |
| **Bazin parametric fits** | ~50 | Rise/fall timescales, amplitudes per band |
| **TDE physics** | ~20 | Decay rates, power law indices |
| **Multi-band GP** | ~15 | 2D Gaussian Process (time + wavelength) |
| **Color evolution** | ~17 | g-r, r-i colors at 20d, 50d post-peak |

**Top 5 most important features** (by XGBoost gain):
1. `r_skew` -- r-band flux asymmetry (TDEs have characteristic asymmetric decline)
2. `gp_ri_color_50d` -- r-i color 50 days post-peak (TDEs stay blue)
3. `gp2d_wave_scale` -- GP wavelength correlation scale
4. `g_skew` -- g-band flux asymmetry
5. `r_rebrightening` -- r-band late-time rebrightening signal

---

## Honorable Mentions

### v115c: Extended Research Features (Private: 0.6757, Public: 0.6840)

Built on v92d's adversarial approach but added 11 research-derived features:
- Nuclear concentration and smoothness metrics
- Multi-band half-peak span ratios (10d, 30d, 100d)
- Color features at peak (g-r, r-i)

**Script**: `scripts/train_v115_xgb_research.py`

### v55: Power Law Decay Features (Private: 0.6737, Public: 0.6873)

Added 27 power law R-squared features across g, r, i bands. TDEs follow characteristic t^(-5/3) decay from accretion physics, distinguishing them from supernovae.

**Script**: `scripts/train_v55_powerlaw.py`

### v42: Conservative Pseudo-Labeling (Private: 0.6735, Public: 0.6666)

Used v34a predictions to pseudo-label high-confidence test samples (threshold > 0.99), then retrained with expanded training set. Inspired by PLAsTiCC 1st place solution.

**Script**: `scripts/train_v42_pseudolabel.py`

### v104: Seed Ensemble (Private: 0.6700, Public: 0.6811)

10-seed averaged v92d predictions to reduce variance. Interestingly, the single-seed v92d outperformed this on private LB, suggesting the "noise" in the single seed was actually signal.

**Script**: `non_successful_tests/scripts/train_v104_seed_ensemble.py`

---

## Project Structure

```
MALLORN-astrophysics/
|-- README.md                   # This file
|-- SOLUTION.md                 # Detailed solution writeup
|-- BENCHMARKS.md               # Complete 126-version experiment history
|-- CLAUDE.md                   # AI assistant context
|-- requirements.txt            # Python dependencies
|-- LICENSE                     # MIT License
|
|-- src/                        # Source code
|   |-- features/               # Feature engineering modules (25 modules)
|   |   |-- statistical.py      # Per-band flux statistics
|   |   |-- colors.py           # Multi-band color features
|   |   |-- lightcurve_shape.py # Rise/decline morphology
|   |   |-- gaussian_process.py # GP interpolation
|   |   |-- multiband_gp.py     # 2D GP with wavelength kernel
|   |   |-- tde_physics.py      # TDE-specific physics features
|   |   |-- bazin_fitting.py    # Bazin parametric fits
|   |   |-- physics_based.py    # Temperature, blackbody fits
|   |   |-- fourier_features.py # FFT-based features
|   |   `-- ...
|   |
|   |-- models/                 # Model architectures
|   |   |-- focal_loss.py       # Focal loss for class imbalance
|   |   |-- lstm_classifier.py  # LSTM (experimental, F1~0.12)
|   |   |-- transformer_classifier.py  # Transformer (experimental)
|   |   `-- atat.py             # ATAT transfer learning
|   |
|   `-- utils/
|       `-- data_loader.py      # Data loading pipeline
|
|-- scripts/                    # Training scripts (136 versions)
|   |-- train_v34a_bazin.py     # v34a: Bazin features (Public LB: 0.6907)
|   |-- train_v42_pseudolabel.py # v42: Pseudo-labeling
|   |-- train_v55_powerlaw.py   # v55: Power law features
|   |-- train_v115_xgb_research.py  # v115c: Extended research features
|   `-- ...
|
|-- non_successful_tests/       # Archived experiments (v81-v108)
|   |-- scripts/
|   |   |-- train_v92_focal_adversarial.py  # ** WINNING MODEL **
|   |   |-- train_v104_seed_ensemble.py     # Seed ensemble
|   |   |-- adversarial_validation.py       # Adversarial weight computation
|   |   `-- ...
|   |-- submissions_archive/    # Historical submissions (76 files)
|   `-- EXPERIMENT_SUMMARY.md   # Experiment notes
|
|-- visualizations/             # Analysis plots
|   |-- tde_examples.png        # Example TDE lightcurves
|   |-- supernova_examples.png  # Example SN lightcurves
|   `-- powerlaw_comparison.png # Power law fits
|
|-- data/                       # Data directory (gitignored)
|   |-- raw/                    # Kaggle competition data
|   `-- processed/              # Feature caches and artifacts
|
`-- submissions/                # Submission CSVs (gitignored)
```

---

## Reproduction Guide

### Prerequisites

```bash
git clone https://github.com/alexylou/MALLORN-astrophysics.git
cd MALLORN-astrophysics

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Step 1: Place Competition Data

Download the data from [Kaggle](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/data) and place it in `data/raw/`:

```
data/raw/
|-- train_log.csv
|-- test_log.csv
|-- sample_submission.csv
`-- split_01/ to split_20/
```

### Step 2: Extract Features

Feature extraction is cached automatically. Running any training script will compute and cache features on first run. The key caches are:
- `features_v4_cache.pkl` -- base statistical features
- `tde_physics_cache.pkl` -- TDE physics features
- `multiband_gp_cache.pkl` -- 2D Gaussian Process features
- `bazin_features_cache.pkl` -- Bazin parametric fits

### Step 3: Compute Adversarial Weights

```bash
python non_successful_tests/scripts/adversarial_validation.py
```

This trains a classifier to distinguish train from test samples and produces `data/processed/adversarial_validation.pkl`.

### Step 4: Train v34a (Feature Backbone)

```bash
python scripts/train_v34a_bazin.py
```

Produces `data/processed/v34a_artifacts.pkl` with feature names and importance rankings.

### Step 5: Train v92d (Winning Model)

```bash
python non_successful_tests/scripts/train_v92_focal_adversarial.py
```

Generates `submissions/submission_v92d_baseline_adv.csv` -- the winning submission.

---

## Key Insights

### What Worked
1. **Adversarial validation** -- Bridging the train-test distribution gap was the single biggest improvement (+0.008 LB over v34a)
2. **Physics-informed features** -- Bazin fitting, GP color evolution, power law decay rates
3. **Gradient boosting on engineered features** -- XGBoost dramatically outperformed deep learning
4. **Conservative approach** -- Simple model + right features > complex ensemble

### What Didn't Work
1. **Deep learning on raw sequences** -- LSTM/Transformer F1 ~ 0.12
2. **More features** -- Adding features beyond ~220 consistently hurt LB
3. **Ensembling optimized models** -- v125 ensemble (OOF 0.7003) scored only 0.6618 on LB
4. **Multi-seed averaging** -- Smoothed out useful variance (v104 < v92d on private LB)
5. **Optuna hyperparameter tuning** -- Optimized for OOF which anti-correlated with LB

### The OOF-LB Paradox

The most important lesson: **higher OOF F1 often meant worse LB F1**. This is characteristic of small datasets with distribution shift. The winning strategy was to deliberately accept lower OOF scores in favor of techniques that align training to the test distribution.

### Physics Insight

TDEs maintain hot blackbody temperatures (~20,000-40,000 K) due to continuous accretion disk heating. Supernovae cool as ejecta expand. AGN fluctuate stochastically. This makes **color evolution** the key physical discriminator -- TDEs stay blue while SNe redden over weeks-to-months timescales.

---

## References

### Competition Resources
- [MALLORN Kaggle Competition](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge)
- [ArXiv Paper: MALLORN Dataset](https://arxiv.org/abs/2512.04946)
- [Data Production Colab](https://colab.research.google.com/drive/1oy96r29Zs4U5Hl-THsZPCnOQbuz21hl5)

### Related Work
- [PLAsTiCC Solutions](https://github.com/kozodoi/Kaggle_Astronomical_Classification) -- 1st place used single LightGBM
- [2025 TDE Paper](https://arxiv.org/abs/2509.25902) -- GP features, post-peak colors
- [ATAT](https://github.com/alercebroker/ATAT) -- Astronomical time series + tabular transformer

---

## License

MIT License -- see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- MALLORN competition organizers for the fascinating dataset and challenge
- PLAsTiCC competition winners for inspiring the feature engineering approach
- Claude Code for pair programming assistance throughout the competition
