# MALLORN Competition Plan

**Competition**: MALLORN Astronomical Classification Challenge
**Goal**: Classify nuclear transients as TDEs vs non-TDEs
**Metric**: F1 Score
**Prize**: EUR 1,000
**Deadline**: January 30, 2026
**Team**: Alexy Louis + Claude (AI collaboration)

---

## Current Standing

| Date | Version | Public F1 | Rank | Features | Notes |
|------|---------|-----------|------|----------|-------|
| Dec 25 | v1 Baseline | 0.3333 | 410/484 | 124 | Statistical features only |
| Dec 25 | v2 Colors | 0.4989 | 305/486 | 175 | +Color features, threshold opt |
| Dec 25 | v3 Shapes | 0.5995 | 157/486 | 240 | +Shape features (rise/fade, power-law) |
| Dec 26 | v4 Physics | 0.6075 | 131/486 | 271 | +Stetson, structure function, temperature |
| Dec 26 | v5 CatBoost | 0.6174 | 120/486 | 273 | +CatBoost, optimized ensemble weights |
| Dec 26 | v6 FeatureSel | *pending* | - | 100 | Feature selection: 100 > 273! |

**Improvement so far: +85% from baseline (and climbing)!**

---

## 1. Problem Understanding

### 1.1 What are TDEs?
**Tidal Disruption Events (TDEs)** occur when a star passes too close to a supermassive black hole and gets torn apart by tidal forces. The stellar debris forms an accretion disk that produces a characteristic flare.

### 1.2 Why is this classification hard?
- **Class imbalance**: Only 4.9% of training data are TDEs (148/3,043)
- **Similar light curves**: AGN variability and some SNe can mimic TDE behavior
- **Sparse observations**: LSST cadence means gaps in time coverage
- **Redshift effects**: Time dilation and K-corrections affect observed properties

### 1.3 Physics-Based Distinguishing Features

| Property | TDEs | Supernovae | AGN |
|----------|------|------------|-----|
| **Color evolution** | Stay blue (hot disk ~30,000K) | Redden as they cool | Stochastic, no trend |
| **Timescale** | Weeks to months rise, slow fade | Days to weeks | Years of variability |
| **Lightcurve shape** | t^(-5/3) power-law decline | Exponential decline | Stochastic |
| **Temperature** | Constant high (~20-40kK) | Cooling over time | Variable |
| **Variability** | Correlated across bands | Correlated | More stochastic |

---

## 2. Development Progress

### Phase 1: ML Baseline - COMPLETE
- [x] Data loading pipeline
- [x] Statistical feature extraction (124 features)
- [x] XGBoost + LightGBM ensemble
- [x] First submission (F1: 0.3333)

### Phase 2: Color Features - COMPLETE
- [x] Colors at peak (g-r, r-i, u-g, i-z)
- [x] Colors at post-peak epochs (+20, +50, +100 days)
- [x] Color evolution slopes
- [x] Threshold optimization for F1
- [x] Submission v2 (F1: 0.4989)

### Phase 3: Lightcurve Shape Features - COMPLETE
- [x] Rise time to peak (per band)
- [x] Fade time from peak (to 50%, 25% of peak)
- [x] Asymmetry ratio (rise/fade)
- [x] Power-law decay fitting (alpha parameter)
- [x] Duration above flux thresholds
- [x] Submission v3 (F1: 0.5995)

### Phase 4: Physics-Based Features - COMPLETE
- [x] Stetson J/K variability indices
- [x] Structure function (variability vs timescale)
- [x] Rest-frame corrected timescales
- [x] Blackbody temperature estimation
- [x] Bazin-like lightcurve parameters
- [x] Excess variance
- [x] Submission v4 (OOF F1: 0.585, awaiting LB result)

### Phase 5: Advanced Techniques - PLANNED
- [ ] RNN/LSTM for temporal patterns
- [ ] Gaussian Process interpolation
- [ ] Hyperparameter optimization (Optuna)
- [ ] Advanced ensemble (stacking, weighted averaging)
- [ ] CatBoost as third model

---

## 3. Feature Engineering Summary

### 3.1 Feature Categories (Current: ~272 features)

| Category | Count | Key Features | Source |
|----------|-------|--------------|--------|
| **Statistical** | 124 | mean, std, skew, kurtosis, amplitude per band | Standard |
| **Color** | 49 | g-r at peak, color slopes, color variability | Physics |
| **Shape** | 65 | rise_time, fade_time, asymmetry, power_law_alpha | Physics |
| **Physics** | 32 | Stetson J/K, structure function, temperature | Literature |

### 3.2 Top Performing Features (from v4)

| Rank | Feature | Type | Importance |
|------|---------|------|------------|
| 1 | r_skew | Statistical | 0.042 |
| 2 | g_skew | Statistical | 0.021 |
| 3 | r_fade_time_25 | Shape | 0.016 |
| 4 | r_bazin_fall_approx | Physics | 0.015 |
| 5 | r_power_law_alpha | Shape | 0.015 |
| 6 | r_i_post_50d | Color | 0.013 |
| 7 | r_sf_tau_30 | Physics | 0.008 |
| 8 | temp_evolution | Physics | 0.007 |

**Physics features contribute 13.8% of total importance**

### 3.3 Physics Features (New in v4)

| Feature | Physical Meaning | Why it helps |
|---------|------------------|--------------|
| **Stetson J** | Correlated variability between bands | TDEs/SNe vary together, AGN more random |
| **Stetson K** | Kurtosis of variability | Distinguishes noise from real variability |
| **Structure Function** | Variability amplitude vs timescale | Different transients have different SF slopes |
| **Rest-frame timescales** | Timescales corrected for z | Removes redshift bias |
| **Temperature** | Blackbody temp from colors | TDEs are hotter than cooling SNe |
| **Temp evolution** | dT/dt | TDEs stay hot, SNe cool rapidly |
| **Bazin params** | Parametric lightcurve shape | Rise/fall timescales in standard form |
| **Excess variance** | Intrinsic variability beyond noise | Stronger for real transients |

---

## 4. Research Findings

### 4.1 From PLAsTiCC Competition (Kaggle 2018)

Key techniques from winning solutions:
- **Gaussian Process interpolation** for irregular time series
- **Bazin function fitting** for parametric lightcurve features
- **tsfresh library** for automated feature extraction
- **Multi-model ensembles** (XGBoost, LightGBM, neural networks)

Sources:
- [PLAsTiCC GitHub Solutions](https://github.com/aerdem4/kaggle-plasticc)
- [PLAsTiCC Results Paper](https://arxiv.org/abs/2012.12392)

### 4.2 From TDE Classification Literature

Key features identified in recent papers:
- **Post-peak colors** and **GP hyperparameters** are most predictive
- **Temperature evolution** distinguishes TDEs from cooling SNe
- Early-phase classifiers achieve 76% recall
- Half of TDEs can be flagged before peak!

Sources:
- [TDE Classifier for Rubin LSST (2025)](https://arxiv.org/abs/2509.25902)
- [FLEET Algorithm](https://par.nsf.gov/biblio/10418418)

### 4.3 Key Insight: Stetson Indices

From [Stetson 1996] and subsequent work:
- **J index**: Measures correlated variability across two bands
- **K index**: Kurtosis-like measure, K~0.798 for Gaussian noise
- High J + high K = real astrophysical variability
- Used extensively in variable star and transient classification

---

## 5. Model Configuration

### 5.1 Current Ensemble: XGBoost + LightGBM

```python
# XGBoost (v4 params)
xgb_params = {
    'max_depth': 5,
    'learning_rate': 0.015,
    'n_estimators': 1200,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'scale_pos_weight': 19.56,  # Class imbalance
}

# LightGBM (v4 params)
lgb_params = {
    'max_depth': 5,
    'learning_rate': 0.015,
    'n_estimators': 1200,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'scale_pos_weight': 19.56,
}

# Ensemble
ensemble_probs = (xgb_probs + lgb_probs) / 2
predictions = (ensemble_probs > optimal_threshold).astype(int)
```

### 5.2 Threshold Optimization

Optimal thresholds have been much lower than 0.5:
- v2: threshold = 0.07
- v3: threshold = 0.17
- This reflects the class imbalance - need to be more aggressive predicting TDEs

### 5.3 Potential Improvements

1. **Weighted ensemble**: Weight models by inverse CV variance
2. **Stacking**: Use OOF predictions as meta-features
3. **Add CatBoost**: Third GBM model for diversity
4. **Hyperparameter tuning**: Optuna for systematic optimization
5. **Feature selection**: Remove low-importance features

---

## 6. Validation Strategy

### 6.1 Cross-Validation
- **Stratified 5-fold CV** to preserve 4.9% TDE ratio
- Track per-fold F1 for stability assessment
- Use OOF predictions for threshold optimization

### 6.2 OOF vs LB Correlation

| Version | OOF F1 | Public F1 | Difference |
|---------|--------|-----------|------------|
| v1 | 0.30 | 0.333 | +11% |
| v2 | 0.51 | 0.499 | -2% |
| v3 | 0.56 | 0.600 | +7% |
| v4 | 0.585 | 0.6075 | +4% |
| v5 | 0.601 | 0.6174 | +3% |
| v6 | 0.627 | *pending* | *TBD* |

Good correlation! OOF consistently underestimates LB by ~5% on average.

---

## 7. Next Steps

### Immediate (v4+)
1. Submit v4 with physics features
2. Analyze which physics features help most
3. Try weighted ensemble (0.6*XGB + 0.4*LGB based on CV)

### Short-term (v5-v6)
1. Hyperparameter tuning with Optuna
2. Add CatBoost to ensemble
3. Feature selection (drop low-importance)

### Medium-term (v7+)
1. RNN/LSTM for temporal patterns
2. Gaussian Process interpolation features
3. Advanced stacking ensemble

---

## 8. File Structure

```
MALLORN astrophysics/
├── CLAUDE.md                       # Project guidance
├── COMPETITION_PLAN.md             # This document
├── data/
│   ├── raw/                        # Original Kaggle data
│   └── processed/                  # Cached features, models
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory analysis
│   └── progression.ipynb           # Progress tracking
├── src/
│   ├── features/
│   │   ├── statistical.py         # Statistical features (124)
│   │   ├── colors.py              # Color features (49)
│   │   ├── lightcurve_shape.py    # Shape features (65)
│   │   └── physics_based.py       # Physics features (32) [NEW]
│   └── utils/
│       └── data_loader.py         # Data loading
├── scripts/
│   ├── train_baseline.py          # v1 training
│   ├── train_v2_colors.py         # v2 training
│   ├── train_v3_shapes.py         # v3 training
│   └── train_v4_physics.py        # v4 training [NEW]
└── submissions/
    ├── submission_baseline.csv    # v1: F1=0.333
    ├── submission_v2_colors.csv   # v2: F1=0.499
    ├── submission_v3_shapes.csv   # v3: F1=0.600
    └── submission_v4_physics.csv  # v4: pending
```

---

## 9. Collaboration Notes

### Team Roles
- **Alexy**: Domain expertise, strategic decisions, critical review
- **Claude**: Implementation, research, systematic exploration

### Key Decisions Made
1. Focus on physics-based features over RNN (higher ROI for now)
2. Use threshold optimization aggressively (0.07-0.17 range)
3. Simple probability averaging for ensemble (works well)

### Open Questions
1. Is RNN worth the investment given strong GBM performance?
2. How much more can we gain from feature engineering?
3. Should we try stacking or keep simple averaging?

---

*Last updated: December 25, 2024 (v4 complete)*
