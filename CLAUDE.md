# CLAUDE.md - MALLORN Astronomical Classification Challenge

## Competition Overview

**Goal**: Classify nuclear transients into **TDEs** (Tidal Disruption Events), **Supernovae**, or **AGN** (Active Galactic Nuclei)

| Aspect | Details |
|--------|---------|
| **Metric** | F1 Score |
| **Prize** | €1,000 |
| **Deadline** | January 30, 2026 |
| **Dataset** | 10,178 simulated LSST lightcurves |
| **Training Split** | 30% (with labels) |
| **Test Split** | 70% (public + private) |

## Key Resources

- **Kaggle Competition**: https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge
- **ArXiv Paper**: https://arxiv.org/abs/2512.04946
- **Data Production Colab**: https://colab.research.google.com/drive/1oy96r29Zs4U5Hl-THsZPCnOQbuz21hl5
- **Data Usage Colab**: https://colab.research.google.com/drive/1N7Q1bxc2gxBuOv2eD3fTYsrxoLC2dQAP
- **PLAsTiCC Solutions (reference)**: https://github.com/kozodoi/Kaggle_Astronomical_Classification

## Transfer Learning Resources (Research Dec 2025)

| Resource | Link | Use Case |
|----------|------|----------|
| **ASTROMER** | `pip install ASTROMER==0.1.7` | Pre-trained embeddings for GBM |
| **ASTROMER GitHub** | github.com/astromer-science/python-library | Single-band transformer |
| **ATAT** | github.com/alercebroker/ATAT | Time series + tabular transformer |
| **SwinV2 for LCs** | github.com/dnlmoreno/VT_Model_for_LightCurves_Classification | Lightcurve→image approach |
| **2025 TDE Paper** | arxiv.org/abs/2509.25902 | GP features, post-peak colors |

---

## Development Plan (Revised Dec 2025)

### Current Best: v20c Optuna-tuned → OOF F1=0.6687 (+0.92% vs v19), LB pending

### Phase A: Enhanced GBM with Transfer Learning Features ✅ DONE
**Goal**: Push GBM score higher using research-backed features

1. **Physics-based features**
   - Gaussian Process length scales (confirmed key in 2025 TDE paper)
   - Blackbody temperature estimation from multi-band flux
   - Temperature evolution rate (dT/dt)
   - Structure functions for variability characterization

2. **Enhanced post-peak color features**
   - More time points: 10d, 30d, 50d, 75d, 100d, 150d post-peak
   - Color evolution derivatives (d(g-r)/dt)
   - Color dispersion/stability metrics

3. **ASTROMER embeddings as features**
   - `pip install ASTROMER==0.1.7`
   - Generate embeddings per band using pre-trained 'macho' weights
   - Feed embeddings to GBM alongside existing features
   - This gives GBM access to learned representations

### Phase B: ATAT Implementation
**Goal**: Compare proper transfer-learning DL to our homemade LSTM

1. Clone and adapt ATAT (github.com/alercebroker/ATAT)
2. Convert MALLORN data to ATAT format
3. Train and evaluate on same CV splits
4. Compare: ATAT vs LSTM (F1=0.12) vs GBM (F1=0.63)

### Phase C: SwinV2 Image Approach (Experimental)
**Goal**: Try novel lightcurve→image approach that beat all benchmarks

1. Clone VT_Model_for_LightCurves_Classification
2. Generate images from multi-band lightcurves
3. Fine-tune ImageNet-pretrained SwinV2
4. Potential to beat all other approaches (F1=84.1 on MACHO, 65.5 on ELAsTiCC)

---

## Original Development Plan (Completed Phases)

### Phase 1: ML Baseline ✅
1. Set up data loading pipeline
2. Implement basic statistical features
3. Train XGBoost baseline
4. Train LightGBM baseline
5. Establish validation strategy (stratified K-fold)

### Phase 2: RNN Development ✅ (Attempted, F1=0.12)
1. Design sequence data pipeline (raw lightcurves)
2. Implement LSTM/GRU architecture
3. Add attention mechanism
4. Train and evaluate

### Phase 3: Advanced Feature Engineering ✅
1. Gaussian Process fitting for lightcurve interpolation
2. Color features (g-r, r-i at peak and post-peak)
3. Color evolution slopes
4. Cross-band correlations
5. Physics-based features (temperature estimation, structure functions)

### Phase 4: Data Augmentation ✅ (Attempted, didn't help DL)
1. Time shifting
2. Flux scaling
3. Cadence resampling
4. Noise injection
5. Evaluate impact on each model

### Phase 5: Ensembling ✅ (v15 ensemble didn't beat v8)
1. Weighted averaging
2. Stacking with meta-learner
3. Blending on holdout
4. Rank averaging
5. Optimize ensemble weights

---

## Actual Data Structure (Downloaded)

Data location: `data/raw/`

### File Organization
```
data/raw/
├── train_log.csv          # Training metadata (3,054 objects)
├── test_log.csv           # Test metadata (7,124 objects)
├── sample_submission.csv  # Submission format
└── split_01/ to split_20/ # Lightcurve data in 20 splits
    ├── train_full_lightcurves.csv
    └── test_full_lightcurves.csv
```

### train_log.csv / test_log.csv Columns
| Column | Description | Example |
|--------|-------------|---------|
| `object_id` | Unique identifier (Tolkien-themed!) | `Dornhoth_fervain_onodrim` |
| `Z` | Redshift | `0.4324` |
| `Z_err` | Redshift error (empty for training) | - |
| `EBV` | Extinction E(B-V) | `0.058` |
| `SpecType` | Spectral type (training only) | `AGN`, `SN II`, `TDE`, etc. |
| `English Translation` | Name meaning | `moon + roof + noble maiden` |
| `split` | Which split folder | `split_01` |
| `target` | **Binary target** (0 or 1) | `0` = non-TDE, `1` = TDE |

### Lightcurve CSV Columns
| Column | Description | Example |
|--------|-------------|---------|
| `object_id` | Links to log file | `Dornhoth_fervain_onodrim` |
| `Time (MJD)` | Modified Julian Date | `63314.4662` |
| `Flux` | Flux in μJy | `10.49938934` |
| `Flux_err` | Flux error | `0.25386745` |
| `Filter` | LSST band | `u`, `g`, `r`, `i`, `z`, `y` |

### Key Technical Details
- **Training objects**: ~3,054 (30%)
- **Test objects**: ~7,124 (70%)
- **LSST bands**: u, g, r, i, z, y
- **Target**: Binary (TDE vs non-TDE)
- **SpecTypes in training**: AGN, SN II, SN Ia, TDE, SLSN, etc.

### Data Loading Strategy
```python
import pandas as pd
import os

# Load metadata
train_log = pd.read_csv('data/raw/train_log.csv')
test_log = pd.read_csv('data/raw/test_log.csv')

# Load all lightcurves (concatenate splits)
train_lcs = []
for i in range(1, 21):
    path = f'data/raw/split_{i:02d}/train_full_lightcurves.csv'
    if os.path.exists(path):
        train_lcs.append(pd.read_csv(path))
train_lightcurves = pd.concat(train_lcs, ignore_index=True)
```

---

## Feature Engineering Categories

### 1. Statistical Features (Baseline)
```python
# Per-band statistics
mean_flux, std_flux, min_flux, max_flux
skewness, kurtosis
number_of_observations
time_span, median_absolute_deviation
```

### 2. Lightcurve Shape Features
- Rise time (to peak)
- Fade time (from peak)
- Peak flux
- Amplitude (max-min)
- Asymmetry (rise_time / fade_time)

### 3. Color Features (Most Important!)
```python
# At peak
g_minus_r_at_peak, r_minus_i_at_peak

# Post-peak (20, 50, 100 days after)
g_minus_r_post_20d, g_minus_r_post_50d

# Color evolution
color_slope_gr = (g_r_post - g_r_peak) / delta_time
```

### 4. Gaussian Process Features
```python
# From GP fit to each lightcurve
gp_amplitude
gp_length_scale_time
gp_length_scale_wavelength
gp_fit_residuals
```

### 5. Physics-Motivated Features
- Blackbody temperature (from SED)
- Temperature evolution (dT/dt)
- Variability structure function
- Host offset (if available)

### 6. Cross-Band Features
```python
flux_ratio_g_r, flux_ratio_r_i
lag_g_r = mjd_peak_g - mjd_peak_r
correlation_g_r
```

### 7. Redshift-Corrected Features
```python
rest_frame_duration = observed_duration / (1 + z)
absolute_mag = apparent_mag - 5*log10(d_L) - K_correction
```

---

## Model Configurations

### XGBoost
- Handles missing values natively (sparse lightcurves)
- Good with heterogeneous features

### LightGBM
- Faster training
- Leaf-wise growth better for imbalanced data

### RNN/LSTM
- Learn temporal patterns from raw lightcurves
- No hand-crafted features needed
- Consider attention mechanism

### Ensemble Strategy
- Weighted averaging
- Stacking (model predictions as meta-features)
- Blending (different folds)
- Rank averaging

---

## Important Considerations

### Class Imbalance
TDEs are rare (~64 out of ~3000 training). Consider:
- Class weights in models
- SMOTE oversampling
- Focal loss for neural networks

### Validation Strategy
- Stratified K-fold (preserve class ratios)
- Track per-class metrics

### Key Insight: Why Colors Matter
TDEs maintain hot blackbody temperatures (~2-4×10⁴ K) because the accretion disk is continuously heated. SNe cool as they expand (adiabatic cooling). AGN colors fluctuate stochastically. This physics directly translates to g-r color behavior!

---

## Commands

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn plotly
pip install george  # For Gaussian Process fitting
pip install torch   # For RNN/LSTM

# Run training scripts (to be created)
python scripts/train_baseline.py
python scripts/train_rnn.py
python scripts/ensemble.py
```

---

## Project Structure (Recommended)

```
MALLORN astrophysics/
├── CLAUDE.md
├── data/
│   ├── raw/           # Downloaded Kaggle data
│   └── processed/     # Feature-engineered datasets
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_ensemble.ipynb
├── src/
│   ├── features/
│   │   ├── statistical.py
│   │   ├── lightcurve_shape.py
│   │   ├── colors.py
│   │   ├── gaussian_process.py
│   │   └── physics_based.py
│   ├── models/
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   └── rnn_model.py
│   └── utils/
│       ├── data_loader.py
│       └── evaluation.py
├── scripts/
│   ├── train_baseline.py
│   ├── train_rnn.py
│   └── ensemble.py
└── submissions/
    └── submission_v1.csv
```

---

## Progress Tracking

### Phase A: Enhanced GBM ⬅️ CURRENT
- [ ] GP length scale features
- [ ] Blackbody temperature features
- [ ] Enhanced post-peak colors
- [ ] ASTROMER embeddings

### Phase B: ATAT
- [ ] Clone and adapt ATAT repo
- [ ] Train and evaluate
- [ ] Compare to LSTM

### Phase C: SwinV2
- [ ] Implement lightcurve→image conversion
- [ ] Fine-tune SwinV2
- [ ] Evaluate

---

### Results History
| Date | Version | OOF F1 | LB F1 | Model | Notes |
|------|---------|--------|-------|-------|-------|
| Dec 25 | v1 | 0.30 | 0.333 | GBM | Statistical features only |
| Dec 25 | v2 | 0.51 | 0.499 | GBM | +Color features |
| Dec 25 | v3 | 0.56 | - | GBM | +Shape features |
| Dec 26 | v8 | 0.6262 | 0.6481 | GBM Ensemble | Rank 47/489 |
| Dec 26 | v11 | 0.12 | - | LSTM | Raw lightcurves |
| Dec 27 | v19 | 0.6626 | 0.6649 | GBM+Multi-band GP | +5.8% OOF |
| Dec 28 | v20c | 0.6687 | 0.6518 | GBM+Optuna tuned | +0.92% over v19 |
| Dec 28 | **v34a** | **0.6667** | **0.6907** | **XGB Optuna** | **BEST LB** |
| Jan 11 | v65 | 0.6780 | 0.6344 | MaxVar features | Severe overfit |
| Jan 11 | v71 | 0.6701 | 0.5800 | PLAsTiCC augment | Catastrophic overfit |
| Jan 11 | v72 | 0.6723 | 0.6500 | Top 100 features | Feature reduction |
| Jan 11 | v74 | 0.6856 | 0.6441 | Selective Cesium | Overfit |
| Jan 11 | v77 | 0.6886 | 0.6714 | LightGBM Optuna | Best OOF LGB |
| Jan 11 | v78 | 0.6921 | 0.6558 | XGB+LGB Ensemble | Ensemble hurt |
| Jan 11 | v79c | 0.6834 | 0.6891 | 70 physics features | 2nd best LB |
| Jan 11 | v80a | 0.7118 | 0.6666 | +Structure Function | Best OOF, overfit |
| Jan 11 | v80c | 0.7000 | 0.6682 | +SF+TDE+Decline | At 0.70 barrier |

### Key Learnings (Updated Jan 2026)

**Critical Pattern: Higher OOF F1 = WORSE LB F1**
- v34a: OOF 0.6667 → LB 0.6907 (best)
- v77: OOF 0.6886 → LB 0.6714
- v80a: OOF 0.7118 → LB 0.6666

**Feature Engineering Insights:**
- **Structure Function features**: +1.5% OOF (captures AGN damped random walk)
- **Color-magnitude relation**: +1.3% OOF (AGN "bluer when brighter")
- **TDE power law deviation**: +1.1% OOF (tests t^-5/3 decay)
- **70 physics features optimal**: More features = worse LB
- **PLAsTiCC augmentation catastrophic**: 0.58 LB (train/test shift)

**Algorithm Insights:**
- XGBoost generalizes better than LightGBM despite worse OOF
- Ensembling HURTS (v78 ensemble worse than solo models)
- CatBoost poor performance (0.56 OOF)

**What Didn't Work:**
- Adding more features (always hurt LB)
- Augmentation (domain shift)
- Ensembling (combined overfitting)
- Cesium time-series features (noise)
- **More features ≠ better** - v20 (375 features) < v19 (172 features) in OOF F1
- **Selective feature addition wins** - Only add features that prove useful through benchmarking
- **Optuna tuning is essential** - v20c gained +1.5% from proper hyperparameter search
- **ATAT transformer works** - 0.50 F1, +38% over naive LSTM, but still below GBM
- **Single XGBoost beats ensemble on OOF** - 0.6708 vs 0.6687, simpler may generalize better
- **Optimal threshold ~0.07** (not 0.5!) due to class imbalance

---

## Quick Start

```bash
# Navigate to project folder
cd "C:\Users\alexy\Documents\Claude_projects\Kaggle competition\MALLORN astrophysics"

# Launch Claude Code
claude
```

Then say: "Let's start with Phase 1 - explore the data and build the first baseline model"
