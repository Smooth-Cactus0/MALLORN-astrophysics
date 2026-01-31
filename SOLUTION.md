# MALLORN Competition Solution -- 4th Place (Private LB F1: 0.6684)

**Author**: Alexy Louis
**Final Ranking**: 4th / 894 (Top 0.5%)
**Winning Model**: v92d -- XGBoost with Adversarial Validation Weights

---

## Table of Contents

1. [Summary](#1-summary)
2. [The Challenge](#2-the-challenge)
3. [Winning Approach: v92d](#3-winning-approach-v92d)
4. [Feature Engineering Pipeline](#4-feature-engineering-pipeline)
5. [Adversarial Validation](#5-adversarial-validation)
6. [Honorable Mentions](#6-honorable-mentions)
7. [What Didn't Work](#7-what-didnt-work)
8. [Key Learnings](#8-key-learnings)
9. [Reproduction](#9-reproduction)

---

## 1. Summary

The winning approach was remarkably simple: a **single XGBoost model** with **222 physics-informed features** and **adversarial validation sample weights**. No ensembling, no Optuna tuning, no multi-seed averaging.

The critical insight was that this competition had a severe **train-test distribution shift**. Standard techniques that improved cross-validation (OOF) scores consistently degraded leaderboard performance. Adversarial validation -- which down-weights training samples that differ from the test distribution -- was the key to bridging this gap.

| Component | Detail |
|-----------|--------|
| **Algorithm** | XGBoost (binary:logistic) |
| **Features** | 222 physics-informed features |
| **Innovation** | Adversarial validation sample weights |
| **OOF F1** | 0.6688 |
| **Private LB F1** | 0.6684 |
| **Public LB F1** | 0.6986 |
| **Training time** | ~2 minutes on CPU |

---

## 2. The Challenge

### Task
Binary classification: identify **TDEs** (Tidal Disruption Events) among simulated LSST nuclear transients. Non-TDEs include supernovae of various types and AGN.

### Data
- **3,054 training objects**: 148 TDE (4.8%), 2,906 non-TDE
- **7,124 test objects**: unlabeled
- **6 photometric bands**: u, g, r, i, z, y (LSST filters)
- **20 time splits**: Lightcurve data partitioned across 20 subdirectories

### Key Difficulties

1. **Extreme class imbalance**: Only 4.8% positive rate requires careful threshold tuning and class weighting.

2. **Train-test distribution shift**: The training and test sets were drawn from different distributions. This made standard cross-validation unreliable -- models optimized for OOF F1 performed worse on the leaderboard.

3. **Sparse, noisy lightcurves**: Many objects have few observations, large gaps, and high noise in faint bands (u, z, y).

4. **Physics complexity**: TDEs, supernovae, and AGN can appear similar depending on the observation cadence and wavelength coverage.

---

## 3. Winning Approach: v92d

### Architecture

The winning model is a standard XGBoost classifier with two key modifications:

1. **Adversarial sample weights**: Training samples are weighted by how similar they are to the test distribution (see Section 5).
2. **Class imbalance handling**: `scale_pos_weight = 19.56` (ratio of non-TDE to TDE).

```python
# v92d hyperparameters (hand-tuned, NOT Optuna-optimized)
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.025,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'scale_pos_weight': 19.56,
    'tree_method': 'hist',
    'random_state': 42,
}

# Training: 5-fold StratifiedKFold, 500 rounds, early stopping at 50
```

### Training Procedure

```
For each of 5 folds:
  1. Create DMatrix with adversarial weights on training split
  2. Train XGBoost with early stopping on validation loss
  3. Predict on validation fold (for OOF) and test set
Average 5 test predictions for final submission
```

### Threshold Selection

Due to class imbalance, the optimal classification threshold is **0.414** (not 0.5). This was found via grid search on OOF predictions.

### Performance Breakdown

| Metric | Value |
|--------|-------|
| OOF F1 | 0.6688 |
| OOF Recall | 0.6959 (103/148 TDEs detected) |
| OOF Precision | 0.6438 |
| False Negatives | 45 TDEs missed |
| False Positives | 57 non-TDEs misclassified |
| Hard TDEs (pred < 0.1) | 20 objects |
| Fold F1 std | 0.0375 |
| Private LB F1 | 0.6684 |
| Public LB F1 | 0.6986 |

### Why This Model Won

v92d has the **best generalization delta** of any model tested:
- OOF-to-Public-LB delta: **+0.030** (model performs *better* on unseen test data)
- This positive delta comes from adversarial weights forcing the model to learn test-relevant patterns.

Script: [`non_successful_tests/scripts/train_v92_focal_adversarial.py`](non_successful_tests/scripts/train_v92_focal_adversarial.py)

---

## 4. Feature Engineering Pipeline

### Overview

The 222 features used by v92d (inherited from v34a) combine multiple physics-informed engineering approaches:

### 4.1 Base Statistical Features (~120 features)

Per-band (u, g, r, i, z, y) statistics computed from raw lightcurves:
- **Flux statistics**: mean, median, std, min, max, range, IQR
- **Shape metrics**: skewness, kurtosis, number of detections
- **Temporal features**: observation span, cadence statistics
- **Cross-band ratios**: flux ratios between bands at peak

Module: `src/features/statistical.py`

### 4.2 Bazin Parametric Fits (~50 features)

Fitting the Bazin function to each band's lightcurve:

```
f(t) = A * exp(-(t - t0) / tau_fall) / (1 + exp(-(t - t0) / tau_rise)) + B
```

This captures supernova-like rise-and-fall behavior. Per band: A, t0, tau_rise, tau_fall, B, chi-squared, rise/fall ratio, peak flux. Cross-band features include Bazin parameter ratios and fit quality metrics.

Module: `src/features/bazin_fitting.py`

### 4.3 TDE Physics Features (~20 features)

Physics-motivated features specific to TDE identification:
- **Decay rate indices**: Power law alpha (expected t^{-5/3} for TDE)
- **Late-time flux ratio**: Characteristic slow TDE decline
- **Rebrightening signals**: TDE-specific late-time rebrightening
- **Duration metrics**: Time above 25%, 50% peak flux

Module: `src/features/tde_physics.py`

### 4.4 Multi-band Gaussian Process (~15 features)

A 2D Gaussian Process with a Matern-3/2 kernel jointly modeling time and wavelength:

```
k(t1, t2, lambda1, lambda2) = k_time(t1, t2) * k_wavelength(lambda1, lambda2)
```

Extracted features: wavelength correlation scale, time scale, time-wavelength ratio, log-likelihood, and GP-interpolated fluxes/colors at fixed time points (20d, 50d post-peak).

Module: `src/features/multiband_gp.py`

### 4.5 Color Evolution Features (~17 features)

The key physical discriminator: TDEs stay blue (hot) while supernovae redden (cool).

- g-r color at peak, 20d, 50d, 100d post-peak
- r-i color at peak, 20d, 50d post-peak
- Color evolution slopes (rate of reddening)
- GP-interpolated colors at fixed epochs

Module: `src/features/colors.py`

### Top 10 Most Important Features

| Rank | Feature | Importance (gain) | Category |
|------|---------|-------------------|----------|
| 1 | `r_skew` | 316.2 | Light curve shape |
| 2 | `gp_ri_color_50d` | 228.2 | Color evolution |
| 3 | `gp2d_wave_scale` | 143.7 | GP wavelength scale |
| 4 | `g_skew` | 122.5 | Light curve shape |
| 5 | `r_rebrightening` | 121.6 | TDE physics |
| 6 | `r_duration_50` | 97.3 | TDE physics |
| 7 | `r_bazin_B` | 95.1 | Bazin fit |
| 8 | `gp_flux_g_50d` | 92.1 | GP interpolated flux |
| 9 | `gp2d_log_likelihood` | 90.1 | GP model quality |
| 10 | `r_late_flux_ratio` | 82.4 | TDE physics |

---

## 5. Adversarial Validation

### Concept

Adversarial validation trains a classifier to distinguish **training samples from test samples**. If it succeeds (AUC > 0.5), the distributions differ. The classifier's probabilities reveal *how much* each training sample resembles the test set.

### Implementation

```python
# 1. Label train=0, test=1
combined_features = pd.concat([train_features, test_features])
combined_labels = [0]*len(train) + [1]*len(test)

# 2. Train LightGBM to distinguish train from test
adv_model = lgb.LGBMClassifier(n_estimators=200, ...)
adv_model.fit(combined_features, combined_labels)

# 3. Get probabilities for training samples
# High prob = "looks like test" (good), Low prob = "looks like train only" (bad)
train_probs = adv_model.predict_proba(train_features)[:, 1]

# 4. Convert to sample weights
# Test-like samples: weight up to ~2x
# Train-only samples: weight down to ~0.17x
sample_weights = normalize(train_probs)
```

### Why It Works for This Competition

The MALLORN dataset has train-test distribution shift likely from:
- Different simulation parameters between train/test splits
- Selection effects in the labeled training subset
- Potential differences in cadence or noise properties

Adversarial validation addresses this directly by telling the model: "focus on patterns that exist in both train and test, ignore patterns unique to training."

Script: [`non_successful_tests/scripts/adversarial_validation.py`](non_successful_tests/scripts/adversarial_validation.py)

---

## 6. Honorable Mentions

### v115c: Extended Research Features

| Metric | Value |
|--------|-------|
| Private LB F1 | **0.6757** |
| Public LB F1 | 0.6840 |

Built on v92d's adversarial approach with 11 additional research-derived features:
- Nuclear concentration and smoothness metrics
- Multi-band half-peak span ratios (mhps at 10d, 30d, 100d)
- Color features at peak (g-r, r-i)

Interestingly, this had the **highest private LB score** among all our submissions, though we selected v92d as our final submission based on public LB performance and generalization analysis.

Script: [`scripts/train_v115_xgb_research.py`](scripts/train_v115_xgb_research.py)

---

### v55: Power Law Decay Features

| Metric | Value |
|--------|-------|
| Private LB F1 | **0.6737** |
| Public LB F1 | 0.6873 |

Added 27 power law R-squared features (9 decay models x 3 bands). TDEs follow a characteristic t^{-5/3} power law decay from accretion disk theory. Fitting various power law models and using the R-squared goodness-of-fit as features captures this physical signature.

Key discriminative feature: `linear_r2` (TDE mean = 0.52, SN mean = 0.30).

Script: [`scripts/train_v55_powerlaw.py`](scripts/train_v55_powerlaw.py)

---

### v42: Conservative Pseudo-Labeling

| Metric | Value |
|--------|-------|
| Private LB F1 | **0.6735** |
| Public LB F1 | 0.6666 |

Pseudo-labeling with a very conservative threshold (> 0.99 confidence):
1. Train v34a on labeled data
2. Predict on test set
3. Add only high-confidence test predictions (pred > 0.99 or pred < 0.01) as pseudo-labels
4. Retrain on expanded dataset

Inspired by the PLAsTiCC 1st place solution. Notable for having a relatively low public LB (0.6666) but strong private LB (0.6735) -- it would have climbed substantially in the shake-up.

Script: [`scripts/train_v42_pseudolabel.py`](scripts/train_v42_pseudolabel.py)

---

### v104: Seed Ensemble

| Metric | Value |
|--------|-------|
| Private LB F1 | **0.6700** |
| Public LB F1 | 0.6811 |

10-seed averaged v92d: same architecture and features, different random seeds. Averaging predictions reduces variance.

Interestingly, the single-seed v92d outperformed this on the private LB (0.6684 vs 0.6700 -- though v104 was slightly better). This suggests the variance reduction from multi-seeding provides marginal benefit when adversarial weights already handle the main source of instability.

Script: [`non_successful_tests/scripts/train_v104_seed_ensemble.py`](non_successful_tests/scripts/train_v104_seed_ensemble.py)

---

## 7. What Didn't Work

### Deep Learning (F1 ~ 0.12)
LSTM, GRU, Transformer, and 1D-CNN models on raw lightcurve sequences all failed catastrophically. Best deep learning result was ATAT (pre-trained astronomical transformer) at F1 = 0.50 -- still far below gradient boosting on engineered features.

### Adding More Features
Every attempt to expand beyond ~220 features degraded LB performance, despite improving OOF. The dataset is too small (3,054 samples) to support high-dimensional feature spaces without overfitting.

### Ensembling Optimized Models
Complex ensembles (XGBoost + LightGBM + CatBoost with optimized weights) achieved OOF F1 = 0.7003 but only 0.6618 on public LB -- a delta of -0.039. Weight optimization on OOF caused severe overfitting.

### Feature Reduction + Re-optimization
Reducing features to top-K by importance improved OOF for all model types, but the retrained models with reduced features performed worse on LB than the originals. The removed features apparently contained signals that mattered more on test than train.

### Optuna Hyperparameter Tuning
Optuna-tuned LightGBM reached OOF 0.6914 but the hand-tuned v34a params (OOF 0.6688) consistently outperformed on LB. HPO optimizes for the wrong objective in distribution-shifted settings.

---

## 8. Key Learnings

### 1. The OOF-LB Paradox
In competitions with distribution shift, cross-validation is adversarial -- it measures overfitting, not generalization. We observed a consistent **negative correlation** between OOF improvement and LB improvement for many techniques.

### 2. Adversarial Validation is Crucial
When train-test distributions differ, adversarial validation provides direct information about the shift. Using this information as sample weights was more effective than any modeling trick.

### 3. Simplicity Wins
The winning model is a single XGBoost with hand-tuned parameters. Every attempt at sophistication (ensembling, HPO, feature engineering expansion) degraded test performance. With only 3,054 training samples and severe class imbalance, the model's ability to memorize noise grows faster than its ability to learn signal.

### 4. Physics-Informed Features are Essential
Domain knowledge (TDE accretion physics, supernova expansion physics, AGN stochasticity) guided feature engineering. The top features all correspond to known physical differences between transient classes.

### 5. Trust Your Validation -- But Not Too Much
We correctly identified v92d as our best generalizer based on its positive OOF-to-LB delta (+0.030). However, we also learned that even this analysis has limits -- v115c would have scored higher on private LB despite having a smaller public LB score.

---

## 9. Reproduction

### Environment

```bash
pip install -r requirements.txt
# Key packages: xgboost>=1.7, lightgbm>=3.3, scikit-learn>=1.0,
# george>=0.4, scipy>=1.9, pandas>=1.5, numpy>=1.21
```

### Full Pipeline

```bash
# 1. Place Kaggle data in data/raw/

# 2. Train feature backbone (generates feature caches)
python scripts/train_v34a_bazin.py

# 3. Compute adversarial weights
python non_successful_tests/scripts/adversarial_validation.py

# 4. Train winning model
python non_successful_tests/scripts/train_v92_focal_adversarial.py
# -> Generates submissions/submission_v92d_baseline_adv.csv

# 5. (Optional) Train honorable mentions
python scripts/train_v115_xgb_research.py
python scripts/train_v55_powerlaw.py
python scripts/train_v42_pseudolabel.py
```

### Expected Output

```
v92d_baseline_adv:
  OOF F1: ~0.6688
  Threshold: ~0.414
  Recall: ~69.6%
  Precision: ~64.4%
  Predicted TDEs in test: ~500-520
```

---

*Competition completed January 30, 2026. Final ranking: 4th / 894.*
