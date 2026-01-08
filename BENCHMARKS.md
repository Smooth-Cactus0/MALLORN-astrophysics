# MALLORN Competition Benchmarks

**Author**: Alexy Louis
**Last Updated**: January 8, 2026
**Current Best**: LB F1 = 0.6907 (v34a)
**Top Competitor**: LB F1 = 0.735

---

## Complete Version History

### Phase 1: Baseline Development (Dec 25-26, 2024)

| Version | OOF F1 | LB F1 | Model | Features | Notes |
|---------|--------|-------|-------|----------|-------|
| v1 | 0.30 | 0.333 | XGBoost | Statistical only | Initial baseline |
| v2 | 0.51 | 0.499 | XGBoost | +Colors | g-r, r-i at peak (+51% OOF) |
| v3 | 0.56 | - | XGBoost | +Shape | Rise/decline times |
| v4 | 0.58 | 0.557 | XGBoost | +Physics | Temperature estimates |
| v5 | 0.55 | - | CatBoost | Same as v4 | CatBoost test |
| v6 | 0.60 | 0.591 | XGBoost | Top 100 features | Feature selection |
| v6b | 0.61 | - | XGBoost | Top 120 features | Extended selection |
| v7 | 0.59 | 0.583 | XGBoost | +TDE physics | Viscous timescale |
| **v8** | **0.6262** | **0.6481** | **Ensemble** | Tuned + ensemble | **First major milestone** |
| v9 | 0.55 | - | XGBoost | +DTW features | Dynamic time warping |

### Phase 2: Deep Learning Experiments (Dec 26-27, 2024)

| Version | OOF F1 | LB F1 | Model | Architecture | Notes |
|---------|--------|-------|-------|--------------|-------|
| v10 | 0.12 | - | LSTM | 2-layer LSTM | Raw lightcurves |
| v11 | 0.12 | - | LSTM | +Augmentation | Time shift, noise |
| v12 | 0.11 | - | LSTM | +Balanced aug | Oversampled TDEs |
| v13 | 0.11 | - | Transformer | Self-attention | 4 heads, 2 layers |
| v14 | 0.47 | - | MLP | Feature-based | Better than raw DL |
| v15 | 0.63 | 0.6463 | Ensemble | GBM+NN+LSTM | Didn't beat v8 |
| v16 | 0.12 | - | LSTM | +PLAsTiCC | External data hurt |
| v17 | 0.10 | - | 1D-CNN | Heavy aug | Augmentation backfired |

**Key Learning**: Deep learning on raw sequences failed spectacularly (F1~0.12). Feature engineering is essential.

### Phase 3: Gaussian Process Features (Dec 27-28, 2024)

| Version | OOF F1 | LB F1 | Model | Features | Notes |
|---------|--------|-------|-------|----------|-------|
| v18 | 0.6130 | - | XGBoost | Per-band GP | GP hurt (-2.1%) |
| **v19** | **0.6626** | **0.6649** | **XGBoost** | **Multi-band GP** | **2D kernel (+5.8%)** |
| v20 | 0.6432 | - | XGBoost | ALL features | Too many (375) hurt |
| v20b | 0.6535 | - | XGBoost | Selective | Better feature selection |
| v20c | 0.6687 | 0.6518 | XGBoost | +Optuna tuned | HPO gains |
| **v21** | **0.6708** | **0.6649** | **XGBoost only** | Same as v20c | **Best single model OOF** |

**Key Learning**: Multi-band GP with 2D kernel dramatically outperforms per-band GP.

### Phase 4: Transfer Learning (Dec 28, 2024)

| Version | OOF F1 | LB F1 | Model | Approach | Notes |
|---------|--------|-------|-------|----------|-------|
| v22 | 0.5053 | - | ATAT | Transformer | +38.5% over naive LSTM |
| v23 | 0.52 | - | XGBoost | +TDE features | Specialized features |
| v24 | 0.48 | - | XGBoost | Oracle transfer | Overfitted |
| v25 | 0.64 | - | Ensemble | Multi-model | No improvement |
| v26 | 0.45 | - | ASTROMER | Pre-trained | Embeddings didn't help |
| v27 | 0.15 | - | GRU | Simpler RNN | Still failed |

### Phase 5: Regularization & Pseudo-labeling (Dec 28-29, 2024)

| Version | OOF F1 | LB F1 | Model | Technique | Notes |
|---------|--------|-------|-------|-----------|-------|
| v28a | 0.62 | - | XGBoost | Strong reg | Underfitted |
| v28b | 0.55 | - | XGBoost | Pseudo-label | 0.85 threshold too aggressive |
| v29a | 0.65 | - | XGBoost | Mild reg | Balanced |

### Phase 6: Advanced Physics Features (Dec 29-30, 2024)

| Version | OOF F1 | LB F1 | Model | Features | Notes |
|---------|--------|-------|-------|----------|-------|
| v30 | 0.6432 | - | XGBoost | Advanced physics | Structure function |
| v30b | 0.6535 | - | XGBoost | Selective physics | Better selection |
| v31 | 0.62 | - | CatBoost | Same features | CatBoost comparison |
| v32 | 0.63 | - | XGBoost | +Interactions | Feature interactions |
| v33 | 0.64 | - | Ensemble | Diverse models | No improvement |

### Phase 7: Bazin Parametric Fitting (Dec 31, 2024)

| Version | OOF F1 | LB F1 | Model | Features | Notes |
|---------|--------|-------|-------|----------|-------|
| **v34a** | **0.6667** | **0.6907** | **XGBoost** | **+Bazin params** | **CURRENT BEST LB (+2.58%)** |
| v34b | 0.6489 | - | XGBoost | +SMOTE | Oversampling hurt |
| v34b_cons | 0.6392 | - | XGBoost | Conservative SMOTE | Still hurt |
| v34c | 0.6748 | 0.6698 | XGBoost | +Calibration | Better OOF, worse LB |

**Key Learning**: Bazin parametric fitting = 24.8% of model importance. Physics-based parametric models work!

### Phase 8: Additional Techniques (Jan 1-4, 2026)

| Version | OOF F1 | LB F1 | Model | Features | Notes |
|---------|--------|-------|-------|----------|-------|
| v35a | 0.65 | - | XGBoost | +Cesium | Astronomy features |
| v37a | 0.63 | - | XGBoost | TDE physics model | Custom parametric |
| v38a | 0.66 | - | Ensemble | Rank averaging | Ensemble test |
| v39b | 0.6688 | - | XGBoost | +Adversarial val | Sample weights |
| v40 | 0.65 | - | XGBoost | +Fourier | FFT features |
| v41 | 0.64 | - | XGBoost | Focal loss | Custom loss |
| v42 | 0.56 | - | XGBoost | Pseudo-label v2 | 0.99 threshold |

### Phase 9: Model Variations (Jan 4, 2026)

| Version | OOF F1 | LB F1 | Model | Approach | Notes |
|---------|--------|-------|-------|----------|-------|
| v43 | 0.64 | - | XGBoost | Cesium v2 | Revised features |
| v44 | 0.63 | - | CatBoost | Bazin features | CatBoost test |
| v45 | 0.61 | - | CatBoost | +Categorical | Host galaxy type |
| v46 | 0.65 | - | LightGBM | Bazin features | LightGBM test |
| v47 | 0.66 | - | XGBoost | Enhanced colors | More color epochs |
| v48 | 0.65 | - | XGBoost | Time-to-decline | Decline metrics |
| v49 | 0.65 | - | LightGBM | Decline features | LightGBM variant |

### Phase 10: Augmentation & Ensemble (Jan 4, 2026)

| Version | OOF F1 | LB F1 | Model | Approach | Notes |
|---------|--------|-------|-------|----------|-------|
| v50 | 0.64 | - | XGBoost | Augmented train | Noise injection |
| v51 | 0.64 | - | LightGBM | Augmented train | Same approach |
| v52 | 0.65 | - | XGBoost | TDE augmentation | TDE-specific aug |
| v53 | 0.65 | - | LightGBM | TDE augmentation | LightGBM variant |
| v54a | 0.66 | - | Ensemble | Simple average | Multi-model |
| v54b | 0.67 | - | Ensemble | v34a threshold | Optimized threshold |
| v54c | 0.66 | - | Ensemble | Weighted average | Inverse error |
| v54d | 0.67 | - | Ensemble | Rank average | Rank-based |
| v54e | 0.65 | - | Ensemble | Voting | Majority vote |
| v54f | 0.67 | - | Ensemble | v34a dominant | Heavy v34a weight |

### Phase 11: Physics Exploration (Jan 4, 2026)

| Version | OOF F1 | LB F1 | Model | Features | Notes |
|---------|--------|-------|-------|----------|-------|
| v55 | 0.65 | - | XGBoost | Power law fits | t^(-5/3) decay |
| v56 | 0.64 | - | XGBoost | AGN peak ordering | Multi-band peak sequence |
| v57a | 0.66 | - | XGBoost | Extinction corrected | EBV correction |
| v57b | 0.66 | - | XGBoost | Top 200 features | Feature selection |
| v58 | 0.65 | - | XGBoost | FWHM features | Full width half max |
| v59a | 0.64 | - | XGBoost | Structure function | Variability |
| v59b | 0.65 | - | XGBoost | Temp at FWHM | Temperature features |
| v59c | 0.65 | - | XGBoost | Stetson indices | Correlated variability |

### Phase 12: Two-Stage Classifier (Jan 4-5, 2026)

| Version | OOF F1 | LB F1 | Model | Approach | Notes |
|---------|--------|-------|-------|----------|-------|
| v60a | 0.6815 | ~0.69 | Two-stage | Hard AGN filter | 0.99 threshold |
| v60b | 0.68 | - | Two-stage | Soft combination | Weighted probs |
| v61a | - | 0.6811 | Two-stage | Enhanced features | High recall variant |

---

## Best Models by Category

### By LB Score
1. **v34a**: LB F1 = 0.6907 (Bazin features)
2. **v60a**: LB F1 ≈ 0.69 (Two-stage)
3. **v61a**: LB F1 = 0.6811 (Enhanced two-stage)
4. **v21**: LB F1 = 0.6649 (XGBoost only)
5. **v8**: LB F1 = 0.6481 (Tuned ensemble)

### By OOF Score
1. **v60a**: OOF F1 = 0.6815 (Two-stage)
2. **v34c**: OOF F1 = 0.6748 (Calibrated)
3. **v21**: OOF F1 = 0.6708 (XGBoost only)
4. **v39b**: OOF F1 = 0.6688 (Adversarial)
5. **v20c**: OOF F1 = 0.6687 (Optuna tuned)

---

## Feature Importance Analysis

### Top 20 Features (v34a)

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | gp2d_wave_scale | 0.0398 | GP |
| 2 | gp_flux_g_50d | 0.0383 | GP |
| 3 | gp_ri_color_50d | 0.0367 | Color |
| 4 | bazin_r_tau_fall | 0.0356 | Bazin |
| 5 | gp2d_time_wave_ratio | 0.0342 | GP |
| 6 | gp_gr_color_50d | 0.0318 | Color |
| 7 | bazin_g_amplitude | 0.0301 | Bazin |
| 8 | all_asymmetry | 0.0289 | Shape |
| 9 | gp_gr_color_20d | 0.0276 | Color |
| 10 | r_mean_flux | 0.0265 | Statistical |
| 11 | bazin_i_peak_flux | 0.0254 | Bazin |
| 12 | Z | 0.0243 | Metadata |
| 13 | gp_flux_r_100d | 0.0231 | GP |
| 14 | all_rise_time | 0.0219 | Shape |
| 15 | bazin_g_tau_rise | 0.0207 | Bazin |
| 16 | gp_ri_color_100d | 0.0195 | Color |
| 17 | g_std_flux | 0.0183 | Statistical |
| 18 | bazin_z_chi2 | 0.0171 | Bazin |
| 19 | all_peak_flux | 0.0159 | Shape |
| 20 | EBV | 0.0147 | Metadata |

### Feature Category Importance

| Category | Total Importance | Count | Avg Importance |
|----------|------------------|-------|----------------|
| GP features | 31.2% | 48 | 0.0065 |
| Color features | 24.8% | 36 | 0.0069 |
| Bazin features | 24.8% | 52 | 0.0048 |
| Shape features | 10.1% | 24 | 0.0042 |
| Statistical | 7.2% | 48 | 0.0015 |
| Metadata | 1.9% | 2 | 0.0095 |

---

## Techniques Evaluation Summary

### What Worked (+% gain)
| Technique | Gain | Version |
|-----------|------|---------|
| Color features | +51% | v2 |
| Multi-band GP | +5.8% | v19 |
| Bazin fitting | +2.6% | v34a |
| Two-stage AGN filter | +2.0% | v60a |
| Hyperparameter tuning | +0.9% | v20c |
| Adversarial validation | +0.3% | v39b |

### What Didn't Work
| Technique | Result | Version | Reason |
|-----------|--------|---------|--------|
| LSTM/Transformer | F1=0.12 | v10-v13 | Raw sequences too sparse |
| SMOTE oversampling | -2% to -4% | v34b | Synthetic samples don't generalize |
| PLAsTiCC external data | F1=0.12 | v16 | Domain shift |
| Heavy augmentation | F1=0.10 | v17 | Degraded signal |
| Too many features | -4% | v20 | Overfitting |
| Per-band GP | -2% | v18 | Misses cross-band correlations |
| Probability calibration | -2% LB | v34c | Calibration overfit OOF |

---

## Computational Resources

| Version | Training Time | Memory | Hardware |
|---------|---------------|--------|----------|
| v34a | ~15 min | 8 GB | CPU (i7) |
| v60a | ~25 min | 10 GB | CPU (i7) |
| v22 (ATAT) | ~2 hours | 16 GB | GPU (RTX 3080) |
| v13 (Transformer) | ~1 hour | 12 GB | GPU (RTX 3080) |

---

## Submission Log

| Date | Version | LB F1 | Rank | Notes |
|------|---------|-------|------|-------|
| Dec 25 | v1 | 0.333 | ~400 | Initial submission |
| Dec 25 | v2 | 0.499 | ~300 | Color features |
| Dec 26 | v8 | 0.6481 | ~50 | Tuned ensemble |
| Dec 27 | v19 | 0.6649 | 23 | Multi-band GP |
| Dec 28 | v21 | 0.6649 | 23 | XGBoost only |
| Dec 31 | **v34a** | **0.6907** | ~15 | **Bazin (BEST)** |
| Jan 5 | v61a | 0.6811 | ~20 | Two-stage |

---

## Gap Analysis

**Current Best**: LB F1 = 0.6907 (v34a)
**Top Competitor**: LB F1 = 0.735
**Gap**: 0.044 F1 points

### Potential Sources of Gap

1. **Undiscovered features** - Competitor may have astronomy domain expertise
2. **Better ensemble** - Multiple diverse models
3. **Advanced pseudo-labeling** - Iterative training on high-confidence predictions
4. **Custom loss functions** - Beyond focal loss
5. **Model stacking** - Meta-learner on OOF predictions

### Remaining Techniques to Try

| Technique | Expected Gain | Effort |
|-----------|---------------|--------|
| Cesium features (proper) | +1-2% | Medium |
| Conservative pseudo-labeling (0.99) | +1-2% | Medium |
| Focal loss (tuned γ) | +1-2% | Low |
| Three-stage cascade | +1-2% | High |
| Temperature evolution | +1-2% | Medium |
| Stacking meta-learner | +1-3% | High |

---

## Reproducibility

All experiments use:
- **Random seed**: 42
- **CV strategy**: 5-fold stratified
- **Evaluation**: OOF F1 score
- **Hardware**: Intel i7 + RTX 3080
- **Python**: 3.9+
