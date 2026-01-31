# MALLORN Competition Benchmarks

**Author**: Alexy Louis
**Last Updated**: January 31, 2026
**Final Ranking**: 4th / 894 (Private LB F1 = 0.6684, v92d)
**Best Public LB**: F1 = 0.6986 (v92d)
**Competition Winner**: ~0.75 F1

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
| **v34a** | **0.6667** | **0.6907** | **XGBoost** | **+Bazin params** | **Bazin breakthrough (+2.58%)** |
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

### Phase 13: Feature Engineering & Algorithm Experiments (Jan 5-15, 2026)

| Version | OOF F1 | Public LB | Model | Features | Notes |
|---------|--------|-----------|-------|----------|-------|
| v62 | 0.66 | - | XGBoost | Blackbody radius | Temperature evolution |
| v63-65 | 0.66-0.67 | - | XGBoost | R_bb, MaxVar features | Incremental improvements |
| v66-v75 | 0.65-0.68 | - | Various | Feature experiments | Many small tests |
| v76 | 0.65 | - | LightGBM | Bazin features | LightGBM comparison |
| v77 | 0.6886 | 0.6714 | LightGBM | Optuna tuned | High OOF, lower LB (overfit) |
| v78 | 0.66 | - | Ensemble | XGB + LGBM | Naive ensemble worse |
| v79c | 0.6834 | 0.6891 | XGBoost | 70 physics features | Curated set |
| v80a | 0.7118 | 0.6666 | XGBoost | Aggressive | Severe overfit! |

### Phase 14: Adversarial Validation & Focal Loss (Jan 15-20, 2026)

| Version | OOF F1 | Public LB | Private LB | Model | Notes |
|---------|--------|-----------|------------|-------|-------|
| v81-v91 | various | - | - | XGBoost | Regularization, focal loss experiments |
| v92a | 0.6327 | - | - | XGBoost | Focal g=1 + adv weights |
| v92b | 0.6495 | - | - | XGBoost | Focal g=2 + adv weights |
| v92c | 0.6477 | - | - | XGBoost | Focal g=2, alpha=0.9 + adv weights |
| **v92d** | **0.6688** | **0.6986** | **0.6684** | **XGBoost** | **Adv weights, no focal = WINNER** |
| v93-v103 | various | - | - | Various | Ensemble, pseudo-label, distillation |
| v104 | 0.6866 | 0.6811 | 0.6700 | XGBoost | 10-seed v92d ensemble |
| v105b | 0.6832 | 0.6862 | - | XGBoost | Cross-feature interactions |
| v106b | 0.6435 | 0.6851 | - | XGBoost | MixUp augmentation |
| v108b | 0.6925 | 0.6325 | - | XGBoost | Knowledge distillation (worst LB!) |

**Key Learning**: v92d proves adversarial validation is the single most important technique. Higher OOF = worse LB confirmed (v108b).

### Phase 15: LightGBM & XGBoost Research (Jan 20-27, 2026)

| Version | OOF F1 | Public LB | Private LB | Model | Notes |
|---------|--------|-----------|------------|-------|-------|
| v110 | 0.6609 | - | - | LightGBM | Heavy regularization |
| v111 | 0.6608 | - | - | LightGBM | DART boosting |
| v112 | 0.6914 | - | - | LightGBM | Optuna constrained search |
| v114a | 0.6542 | 0.6542 | - | LightGBM | Best research features |
| v114d | 0.6786 | 0.6797 | - | LightGBM | Minimal research features |
| v115a | 0.68 | 0.6894 | - | XGBoost | v34a + adv weights (no extras) |
| v115b | 0.68 | 0.6682 | - | XGBoost | + minimal research features |
| **v115c** | **0.68** | **0.6840** | **0.6757** | **XGBoost** | **+ extended research features** |

### Phase 16: CatBoost & Ensembles (Jan 27-29, 2026)

| Version | OOF F1 | Public LB | Model | Notes |
|---------|--------|-----------|-------|-------|
| v118 | 0.6289 | - | CatBoost | Optuna-tuned, 230 features |
| v123 | 0.6882 | - | CatBoost | **Optimized: 75 features (+0.06!)** |
| v124_conservative | 0.6857 | 0.6976 | Ensemble | 4-model blend (good generalization) |
| v124_best | 0.6859 | 0.6894 | Ensemble | Optimized weight blend |
| v125_optimized | 0.7003 | 0.6618 | Ensemble | **Overfit: 60% CatBoost weight** |
| v125_equal | 0.6786 | 0.6746 | Ensemble | Equal 4-model blend |
| v126_heavy | 0.6744 | 0.6528 | Ensemble | v92d-heavy blend (overfit) |

**Key Learning**: Feature reduction helped all models dramatically (CatBoost: +0.06, LightGBM: +0.04). But ensemble weight optimization overfit severely.

---

## Final Private Leaderboard Results

| Submission | Public LB | Private LB | Delta (Pub-Priv) | Selected? |
|------------|-----------|------------|-------------------|-----------|
| **v92d_baseline_adv** | **0.6986** | **0.6684** | -0.030 | **Yes (Final)** |
| v115c_extended | 0.6840 | 0.6757 | -0.008 | No |
| v55_powerlaw | 0.6873 | 0.6737 | -0.014 | No |
| v42_pseudolabel | 0.6666 | 0.6735 | +0.007 | No |
| v104_seed_ensemble | 0.6811 | 0.6700 | -0.011 | No |

---

## Best Models by Category

### By Private LB Score (Final)
1. **v115c**: Private F1 = 0.6757 (Extended research features)
2. **v55**: Private F1 = 0.6737 (Power law decay features)
3. **v42**: Private F1 = 0.6735 (Conservative pseudo-labeling)
4. **v104**: Private F1 = 0.6700 (10-seed ensemble)
5. **v92d**: Private F1 = 0.6684 (Adversarial weights -- selected as final)

### By Public LB Score
1. **v92d**: Public F1 = 0.6986 (Adversarial weights)
2. **v124_conservative**: Public F1 = 0.6976 (4-model ensemble)
3. **v34a**: Public F1 = 0.6907 (Bazin features)
4. **v115a**: Public F1 = 0.6894 (v34a + adv weights)
5. **v124_best**: Public F1 = 0.6894 (Optimized ensemble)

### By OOF Score (Unreliable -- inversely correlated with LB!)
1. **v80a**: OOF F1 = 0.7118 -- Public LB only 0.6666 (OVERFIT)
2. **v125_optimized**: OOF F1 = 0.7003 -- Public LB only 0.6618 (OVERFIT)
3. **v108b**: OOF F1 = 0.6925 -- Public LB only 0.6325 (WORST)
4. **v112**: OOF F1 = 0.6914
5. **v77**: OOF F1 = 0.6886 -- Public LB 0.6714 (overfit)

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

| Date | Version | Public LB | Private LB | Rank | Notes |
|------|---------|-----------|------------|------|-------|
| Dec 25 | v1 | 0.333 | - | ~400 | Initial submission |
| Dec 25 | v2 | 0.499 | - | ~300 | Color features |
| Dec 26 | v8 | 0.6481 | - | ~50 | Tuned ensemble |
| Dec 27 | v19 | 0.6649 | - | 23 | Multi-band GP |
| Dec 28 | v21 | 0.6649 | - | 23 | XGBoost only |
| Dec 31 | **v34a** | **0.6907** | - | ~15 | Bazin features |
| Jan 5 | v61a | 0.6811 | - | ~20 | Two-stage |
| Jan 15 | v55 | 0.6873 | **0.6737** | ~15 | Power law features |
| Jan 20 | **v92d** | **0.6986** | **0.6684** | ~8 | **Adversarial validation** |
| Jan 27 | v115c | 0.6840 | **0.6757** | ~12 | Extended research |
| Jan 28 | v124_conservative | 0.6976 | - | ~8 | 4-model ensemble |
| Jan 29 | v125_optimized | 0.6618 | - | ~30 | Overfit ensemble |

---

## Post-Competition Analysis

**Final Ranking**: 4th / 894 (Top 0.5%)
**Winner**: ~0.75 F1 (private)

### The OOF-LB Anti-Correlation

The defining pattern of this competition: higher OOF F1 reliably predicted **worse** leaderboard performance. This table shows the anti-correlation across all models with LB scores:

| Model | OOF F1 | Public LB | Delta |
|-------|--------|-----------|-------|
| v92d | 0.6688 | 0.6986 | **+0.030** |
| v34a | 0.6667 | 0.6907 | +0.024 |
| v79c | 0.6834 | 0.6891 | +0.006 |
| v77 | 0.6886 | 0.6714 | -0.017 |
| v80a | 0.7118 | 0.6666 | -0.045 |
| v108b | 0.6925 | 0.6325 | **-0.060** |

**Lesson**: In distribution-shifted competitions, OOF optimization is counterproductive. Techniques that align training to test distribution (adversarial validation) matter more than model complexity.

---

## Reproducibility

All experiments use:
- **Random seed**: 42
- **CV strategy**: 5-fold stratified
- **Evaluation**: OOF F1 score
- **Hardware**: Intel i7 + RTX 3080
- **Python**: 3.9+
