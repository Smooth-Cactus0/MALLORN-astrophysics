# CURRENT BEST MODEL - v34a (December 31, 2024)

## 🏆 Performance Summary

| Metric | Score | Rank/Context |
|--------|-------|--------------|
| **Leaderboard F1** | **0.6907** | **Best submission to date** |
| **OOF F1** | 0.6667 | Cross-validation score |
| **Improvement vs v21** | +2.58% LB | Previous best: 0.6649 |
| **Competition Rank** | TBD | Previous: 23/496 |

---

## 🎯 Model Architecture

### Core: XGBoost with Bazin Parametric Features

**Feature Set (224 features total):**
1. **v21 Baseline Features (172)**:
   - 120 selected statistical features (from 200+ candidates)
   - TDE physics features (temperature, color evolution)
   - Multi-band Gaussian Process features (2D Matérn kernel)

2. **Bazin Parametric Features (52)** - NEW in v34a:
   - **Per-band features** (6 bands × 8 features = 48):
     - `bazin_A`: Peak amplitude above baseline
     - `bazin_t0`: Time of peak brightness
     - `bazin_tau_rise`: Rise timescale (days)
     - `bazin_tau_fall`: Decay timescale (days)
     - `bazin_B`: Baseline flux
     - `bazin_fit_chi2`: Fit quality (reduced χ²)
     - `bazin_rise_fall_ratio`: τ_rise / τ_fall
     - `bazin_peak_flux`: A + B (total peak flux)

   - **Cross-band consistency features** (4):
     - `bazin_rise_consistency`: Rise time consistency across g,r,i bands
     - `bazin_fall_consistency`: Fall time consistency across g,r,i bands
     - `bazin_avg_fit_chi2`: Average fit quality
     - `bazin_fit_quality_dispersion`: Fit quality variation

**Bazin Function:**
```python
f(t) = A * exp(-(t-t0)/τ_fall) / (1 + exp(-(t-t0)/τ_rise)) + B
```

This parametric model captures the characteristic rise-and-fall shape of transient lightcurves:
- Sigmoid rise controlled by τ_rise
- Exponential decay controlled by τ_fall
- More robust than raw statistics on sparse lightcurves

---

## 📊 Feature Importance Analysis

**Top 10 Features (by XGBoost gain):**
1. `gp_flux_g_50d` (100.3) - Gaussian Process interpolated g-band flux at peak+50d
2. `r_power_law_alpha` (73.0) - Power law decay index in r-band
3. `r_skew` (66.3) - Skewness of r-band lightcurve
4. **`r_bazin_B` (28.1)** ⭐ - **Top Bazin feature at rank #7**
5. `gp2d_wave_scale` (55.1) - 2D GP wavelength scale
6. **`r_bazin_tau_fall` (40.9)** ⭐ - **Rank #5**
7. `g_skew` (28.3)
8. `gp_ri_color_50d` (26.7)
9. `r_rebrightening` (22.3)
10. `gp2d_time_wave_ratio` (21.9)

**Bazin Features Account for 24.8% of Total Model Importance**
- Despite being only 23% of features (52/224)
- Top Bazin feature ranks #4 overall
- 3 Bazin features in top 15

---

## ⚙️ Model Hyperparameters

```python
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,                    # Conservative depth to prevent overfitting
    'learning_rate': 0.025,            # Slow learning for stability
    'subsample': 0.8,                  # Row sampling
    'colsample_bytree': 0.8,           # Feature sampling
    'min_child_weight': 3,             # Minimum samples per leaf
    'reg_alpha': 0.2,                  # L1 regularization
    'reg_lambda': 1.5,                 # L2 regularization
    'scale_pos_weight': ~20,           # Handle 148:2895 class imbalance
    'tree_method': 'hist',             # Fast histogram-based splits
    'random_state': 42,
    'num_boost_round': 500,
    'early_stopping_rounds': 50
}
```

**Validation Strategy:**
- 5-fold Stratified K-Fold (preserves 4.9% TDE ratio)
- Random state: 42 (reproducible splits)
- Threshold optimization: 0.39 (tuned on OOF predictions)

---

## 🔬 Why Bazin Features Work for TDE Detection

### Physical Interpretation

**TDEs have distinctive temporal signatures:**

1. **Rapid Rise (τ_rise ~ 20-40 days)**:
   - Tidal disruption creates sudden accretion disk
   - Fast increase in luminosity
   - Shorter than most SNe (τ_rise ~ 50-100 days)

2. **Slow Decay (τ_fall ~ 100-300 days)**:
   - Accretion disk gradually dissipates
   - Longer decay than SNe Ia (τ_fall ~ 30-50 days)
   - Distinctive τ_rise/τ_fall ratio

3. **Consistent Cross-Band Behavior**:
   - TDEs maintain hot blackbody temperatures
   - All bands peak at similar times (low dispersion)
   - AGN show stochastic variations (high dispersion)

**Model Learning:**
The Bazin function compresses sparse, noisy lightcurves (10-50 observations) into 5 physically meaningful parameters that XGBoost can effectively learn from, rather than trying to learn patterns from raw flux values.

---

## 📈 Training Performance

**Cross-Validation Results:**
```
Fold 1: F1 = 0.6486 @ threshold = 0.414
Fold 2: F1 = 0.7059 @ threshold = 0.229
Fold 3: F1 = 0.7419 @ threshold = 0.362
Fold 4: F1 = 0.6667 @ threshold = 0.188
Fold 5: F1 = 0.7353 @ threshold = 0.362

Overall OOF F1: 0.6667 @ threshold = 0.39
```

**Confusion Matrix:**
- True Positives (TP): 107
- False Positives (FP): 66
- False Negatives (FN): 41
- True Negatives (TN): 2,829

**Metrics:**
- Precision: 0.6185 (62% of predicted TDEs are real TDEs)
- Recall: 0.7230 (72% of real TDEs are detected)

**Bazin Fit Success Rate:**
- Training: 97.7% successful fits (2,975/3,043)
- Test: 98.1% successful fits (6,997/7,135)

---

## 🚫 What We Tried That Didn't Work

### v34b: Bazin + SMOTE Oversampling
- **OOF F1**: 0.6585 (-1.22% vs v34a)
- **LB F1**: 0.6430 (-4.77% vs v34a)
- **Issue**: Aggressive synthetic oversampling (25% minority target)
  - Generated 579 TDEs per fold (80% synthetic)
  - Over-smoothed minority class distribution
  - High fold variance (F1: 0.60-0.78)
- **Conclusion**: With only 118 real TDEs per fold, synthetic samples lack natural diversity

### v34b_conservative: Bazin + Conservative SMOTE
- **OOF F1**: 0.6407 (-3.90% vs v34a)
- **Issue**: Even "conservative" 10% target (231 TDEs, 49% synthetic) hurt
- **Conclusion**: SMOTE not viable for this dataset size

### v34c: Bazin + Probability Calibration
- **OOF F1**: 0.6748 (+1.22% vs v34a)
- **LB F1**: 0.6698 (-2.09% vs v34a)
- **Issue**: Isotonic calibration improved OOF but didn't generalize to LB
  - Stabilized threshold (0.08 vs 0.39)
  - Improved recall but overfit to validation folds
- **Conclusion**: Calibration helps OOF but may overfit

---

## 🎯 Key Insights

1. **Simpler is better**: v34a (no augmentation, no calibration) generalized best to LB
2. **Physics matters**: Bazin features capture real TDE signatures (24.8% importance)
3. **Small data limitations**: Only 148 TDEs makes synthetic oversampling harmful
4. **Parametric models win**: 5 Bazin parameters > raw flux statistics on sparse data
5. **OOF-LB correlation**: v34a had lower OOF but higher LB - validation strategy worked

---

## 📁 Code Location

**Training Script**: `scripts/train_v34a_bazin.py`
**Feature Module**: `src/features/bazin_fitting.py`
**Artifacts**: `data/processed/v34a_artifacts.pkl`
**Submission**: `submissions/submission_v34a_bazin.csv`

---

## 🔜 Next Steps: Technique #4 (Cesium Astronomy Features)

Building on v34a foundation with domain-specific features:
- `stetson_j`, `stetson_k` (correlated variability)
- `beyond_1_std` (fraction of outliers)
- `flux_percentile_ratios` (shape characterization)
- `maximum_slope` (fastest rise/fall)
- Expected gain: +1-3% (PLAsTiCC winners used this)
- Target: LB F1 > 0.71

---

## 📊 Progress Toward Podium Goal

**Current Status:**
- **Target for Top 3**: LB F1 > 0.73
- **Current Best (v34a)**: LB F1 = 0.6907
- **Gap to Close**: +3.93 percentage points
- **Techniques Completed**: 1/8 (Bazin)
- **Techniques Remaining**: 7 (Cesium, Fourier, Adversarial Validation, Focal Loss, etc.)

**Projected Path:**
- v35a (Bazin + Cesium): Target 0.70-0.71
- v35b (+ Adversarial Validation): Target 0.71-0.72
- v35c (+ Fourier): Target 0.72-0.73
- v36a (+ Focal Loss): Target 0.73-0.74 → **PODIUM RANGE**

---

**Generated**: December 31, 2024
**Status**: Active Best Model
**Next Update**: After Cesium features (v35a)
