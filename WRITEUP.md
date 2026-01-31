# MALLORN Challenge Writeup -- 4th Place (Alexy Louis)


**Code**: https://github.com/Smooth-Cactus0/MALLORN-astrophysics

---

## Background

I have an MSc in astrophysics, working as a research engineer in computational neurosciences.

---

## Summary of your approach

**Did you use features for classification?** Yes

**What type of ML algorithm did you apply?** Gradient Boosted Decision Trees (XGBoost), with adversarial validation sample weighting, for my best submission. I also tried LGBM, Catboost, and Neural Networks.

### Features extracted and how

The final model used **222 physics-informed features** across five categories:

1. **Base statistical features (~120)**: Per-band (u, g, r, i, z, y) flux statistics -- mean, median, std, skewness, kurtosis, IQR, observation span, cadence metrics, and cross-band flux ratios at peak.

2. **Bazin parametric fits (~50)**: For each band, I fit the Bazin function `f(t) = A * exp(-(t-t0)/tau_fall) / (1 + exp(-(t-t0)/tau_rise)) + B` to model the lightcurve rise and decline. Extracted parameters per band: amplitude (A), reference time (t0), rise timescale (tau_rise), fall timescale (tau_fall), baseline flux (B), chi-squared goodness-of-fit, rise/fall ratio, and peak flux. Cross-band Bazin parameter ratios and fit quality dispersion were also included.

3. **TDE physics features (~20)**: Motivated by TDE accretion physics -- power law decay indices (testing for the theoretical t^{-5/3} dependence), late-time flux ratios, rebrightening signals, and duration metrics (time above 25% and 50% of peak flux).

4. **Multi-band Gaussian Process features (~15)**: A 2D GP with a Matern-3/2 kernel jointly modeling time and wavelength: `k(t1, t2, lambda1, lambda2) = k_time(t1, t2) * k_wavelength(lambda1, lambda2)`. Extracted features: wavelength correlation scale, time scale, time-wavelength ratio, log-likelihood, and GP-interpolated fluxes and colors at fixed epochs (20d, 50d post-peak).

5. **Color evolution features (~17)**: g-r and r-i colors at peak and at 20, 50, and 100 days post-peak, plus color evolution slopes. This exploits the physical difference that TDEs maintain hot temperatures (~20,000-40,000 K) from continuous accretion heating, while supernovae cool as ejecta expand.

### Feature importance chart

The top 20 features by XGBoost gain importance:

| Rank | Feature | Gain | Category |
|------|---------|------|----------|
| 1 | `r_skew` | 316.2 | Light curve shape |
| 2 | `gp_ri_color_50d` | 228.2 | Color evolution |
| 3 | `gp2d_wave_scale` | 143.7 | GP wavelength scale |
| 4 | `g_skew` | 122.5 | Light curve shape |
| 5 | `r_rebrightening` | 121.6 | TDE physics |
| 6 | `r_duration_50` | 97.3 | TDE physics |
| 7 | `r_bazin_B` | 95.1 | Bazin parametric fit |
| 8 | `gp_flux_g_50d` | 92.1 | GP interpolated flux |
| 9 | `gp2d_log_likelihood` | 90.1 | GP model quality |
| 10 | `r_late_flux_ratio` | 82.4 | TDE physics |
| 11 | `r_decay_alpha` | 72.1 | TDE physics |
| 12 | `gp_gr_color_20d` | 71.8 | Color evolution |
| 13 | `flux_p25` | 71.8 | Statistical |
| 14 | `gp_gr_color_50d` | 66.4 | Color evolution |
| 15 | `r_power_law_alpha` | 63.0 | TDE physics |
| 16 | `r_bazin_tau_fall` | 62.1 | Bazin parametric fit |
| 17 | `gp2d_time_wave_ratio` | 53.2 | GP scale ratio |
| 18 | `u_g_peak` | 52.4 | Color at peak |
| 19 | `Z` | 49.3 | Redshift |
| 20 | `g_bazin_B` | 47.8 | Bazin parametric fit |

Feature importance by category: Color evolution ~28%, TDE physics ~22%, Bazin fits ~21%, GP features ~18%, Statistical ~11%.

### Use of redshifts and flux uncertainties

**Redshift (Z)**: Included directly as a feature. It ranked 19th in importance. Redshift provides context for luminosity and distance, though it was not the dominant discriminator.

**Flux uncertainties**: Not used directly as features. The flux measurements were used as-is. I considered uncertainty-weighted statistics early on, but they did not improve performance in my experiments.

### Data augmentation

**No data augmentation was used in the final model.** I tested several augmentation approaches during development:

- **Noise injection** (adding Gaussian noise to lightcurves): No improvement
- **SMOTE / ADASYN oversampling**: Consistently hurt generalization (-2% to -4% LB)
- **PLAsTiCC external data**: Domain shift between PLAsTiCC and MALLORN simulations made external data counterproductive
- **MixUp augmentation**: Marginal OOF improvement but worse LB

The class imbalance (148 TDE vs 2,906 non-TDE) was handled through XGBoost's `scale_pos_weight` parameter (set to 19.56, the class ratio) and threshold optimization.

---

## Additional details

### The key innovation: Adversarial Validation

The single most impactful technique was **adversarial validation sample weighting**. I trained a classifier to distinguish training samples from test samples, revealing a distribution shift between the two sets. Training samples that "looked like" test data received higher weights (~2x), while train-only samples were down-weighted (~0.17x).

This addressed the core challenge: models optimized for cross-validation (OOF) consistently performed **worse** on the leaderboard. For example, my best OOF model (F1=0.7003) scored only 0.6618 on the public LB, while v92d (OOF=0.6688) scored 0.6986. Adversarial weighting forced the model to focus on patterns shared between train and test.

### Model details

- **Algorithm**: XGBoost (binary:logistic)
- **Hyperparameters**: Hand-tuned (max_depth=5, learning_rate=0.025, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.5)
- **Validation**: 5-fold StratifiedKFold (seed=42)
- **Training**: 500 boosting rounds with early stopping (patience=50)
- **Threshold**: 0.414 (optimized on OOF predictions)
- **Single seed, no ensembling**: The simplicity was intentional -- every attempt at ensembling or multi-seed averaging degraded LB performance

### What I tried that didn't work

| Approach | Effect on OOF | Effect on LB | Lesson |
|----------|--------------|--------------|--------|
| Ensembling 4 models (XGB+LGB+CatBoost) | +0.03 OOF | -0.04 LB | Ensemble weight optimization overfit |
| Feature reduction (222 to 120) | +0.009 OOF | Worse LB | Removed features mattered on test |
| Optuna hyperparameter tuning | +0.02 OOF | Worse LB | HPO optimizes wrong objective |
| Multi-seed averaging (5 seeds) | Stable OOF | Worse LB | Smoothed useful variance |
| Deep learning (LSTM, Transformer) | F1~0.12 | N/A | Raw sequences too sparse |
| CatBoost / LightGBM | ~0.69 OOF | ~0.68 LB | Good diversity but didn't beat XGB |

### Code

Full code repository with reproduction instructions: https://github.com/Smooth-Cactus0/MALLORN-astrophysics

Key files:
- `non_successful_tests/scripts/train_v92_focal_adversarial.py` -- Winning model
- `non_successful_tests/scripts/adversarial_validation.py` -- Adversarial weight computation
- `scripts/train_v34a_bazin.py` -- Feature backbone (Bazin fitting)
- `src/features/` -- All 25 feature engineering modules
- `SOLUTION.md` -- Detailed technical writeup

---

## Participant Feedback



I was drawn to the MALLORN challenge because it combines machine learning with real astrophysics and I miss that field of research. The dataset seemed very approachable so it was a great way for me to test my skills in a domain that I love and in which I'm not involved in my daily job.

Playing with astronomical data was a big appeal, the possibility of contributing to the project was a great motivation to try and get into that top three (so close yet so far)

Usually you have a 'code' tab in the Kaggle competitions where people share a lot of what they do, it's a good way to learn for people getting started and a good way to give back when you're a bit more experienced, this could have been a nice addition.
