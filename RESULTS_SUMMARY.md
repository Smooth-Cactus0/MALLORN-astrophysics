# MALLORN Competition Results Summary

## Competition Overview
- **Goal**: Classify TDEs (Tidal Disruption Events) vs non-TDEs
- **Metric**: F1 Score
- **Best LB Score**: 0.6907 (v34a XGBoost)
- **Leader Score**: ~0.735
- **Gap to Close**: ~0.028

---

## Key Learnings

### 1. The Overfitting Pattern
**Critical Finding**: Higher OOF F1 consistently leads to WORSE LB F1

| Model | OOF F1 | LB F1 | Gap |
|-------|--------|-------|-----|
| v34a XGBoost | 0.6667 | **0.6907** | +0.024 |
| v77 LightGBM | 0.6886 | 0.6714 | -0.017 |
| v80a SF Features | 0.7118 | 0.6666 | -0.045 |

**Implication**: Models that "underfit" on training data generalize better to test.

### 2. Feature Engineering Results

#### Features That Worked (Improved OOF)
- **Structure Function (SF)**: +1.51% OOF - captures AGN's damped random walk
- **Color-Magnitude Relation**: +1.30% OOF - "bluer when brighter" pattern
- **TDE Power Law Deviation**: +1.14% OOF - tests against t^-5/3 decay
- **Curated Physics Features**: Set C (70 features) from v79

#### Features That Didn't Help LB
- MaxVar features (v65): OOF 0.6780 -> LB 0.6344 (severe overfit)
- PLAsTiCC Augmentation (v71): OOF 0.6701 -> LB 0.58 (catastrophic)
- Cesium Features (v73-74): Added noise, worse LB
- All High-SNR features combined: More features = worse

### 3. Algorithm Comparison

| Algorithm | Best OOF F1 | Best LB F1 | Notes |
|-----------|-------------|------------|-------|
| XGBoost (v34a) | 0.6667 | **0.6907** | Best LB, Optuna-tuned |
| LightGBM (v77) | 0.6886 | 0.6714 | Good OOF, worse LB |
| CatBoost (v75) | 0.5597 | - | Poor performance |
| Ensemble (v78) | 0.6921 | 0.6558 | Ensembling hurt |

### 4. Feature Count Sweet Spot

| Features | Model | OOF F1 | LB F1 |
|----------|-------|--------|-------|
| 224 | v34a XGBoost | 0.6667 | **0.6907** |
| 100 | v72 Top100 | 0.6723 | 0.6500 |
| 70 | v79 Set C | 0.6834 | 0.6891 |
| 82 | v80a SF | 0.7118 | 0.6666 |

**Insight**: ~70-100 physics-focused features seems optimal for LB generalization.

---

## Experiment Timeline

### Phase 1: Feature Addition (v66-v74)
- v66-v67: Lean MaxVar approaches - didn't improve
- v68-v70: Two-stage AGN filtering - marginal gains
- v71: PLAsTiCC augmentation - severe overfit (0.58 LB)
- v72: Feature subtraction - found top 100 optimal
- v73-v74: Cesium features - added noise

### Phase 2: Algorithm Exploration (v75-v78)
- v75: CatBoost - poor OOF (0.5597)
- v76: LightGBM default - decent (0.6583 OOF)
- v77: LightGBM Optuna - best OOF (0.6886)
- v78: XGB+LGB ensemble - hurt LB (0.6558)

### Phase 3: Physics Feature Engineering (v79-v80)
- v79: Curated physics sets (A/B/C) - Set C best LB (0.6891)
- v80: High-SNR physics features - best OOF (0.7118) but overfit

---

## Feature Importance Analysis

### Top 10 Most Stable Features (across XGBoost & LightGBM)

| Rank | Feature | Stability | Category |
|------|---------|-----------|----------|
| 1 | r_skew | 100.0 | Light curve shape |
| 2 | gp_gr_color_50d | 59.5 | Color evolution |
| 3 | r_bazin_B | 53.2 | Bazin amplitude |
| 4 | gp2d_wave_scale | 43.0 | GP wavelength scale |
| 5 | r_bazin_tau_rise | 30.4 | Rise timescale |
| 6 | r_bazin_tau_fall | 27.9 | Decline timescale |
| 7 | gp_gr_color_20d | 27.3 | Early color |
| 8 | gp_ri_color_50d | 27.0 | r-i color |
| 9 | g_skew | 26.3 | g-band asymmetry |
| 10 | g_bazin_B | 21.4 | g-band amplitude |

### Physics Feature Categories by Importance

| Category | Avg Stability | Count | Key Examples |
|----------|---------------|-------|--------------|
| Shape | 7.63 | 32 | skew, duration, rise/fall times |
| Color | 7.23 | 38 | g-r colors at various epochs |
| GP Physics | 6.47 | 17 | 2D GP wavelength/time scales |
| Bazin | 6.10 | 40 | Rise/fall parameters |
| Variability | 3.83 | 12 | beyond_1std, rebrightening |

---

## New Features Implemented

### high_snr_physics.py

1. **Structure Function (SF)**
   - SF at multiple timescales (1, 5, 10, 20, 50, 100 days)
   - SF slope and amplitude
   - DRW timescale estimation
   - *Purpose*: Distinguish AGN (damped random walk) from TDE (systematic decline)

2. **Color-Magnitude Relation**
   - Correlation between brightness and color
   - "Bluer when brighter" (BWB) strength
   - Color-magnitude scatter
   - *Purpose*: AGN shows BWB, TDEs don't

3. **Decline Consistency**
   - Cross-band decline rate CV
   - Decline smoothness
   - Band-to-band decline ratios
   - *Purpose*: TDEs decline achromatically

4. **TDE Power Law Deviation**
   - Deviation from t^-5/3 (bolometric)
   - Deviation from t^-5/12 (monochromatic)
   - Best-fit power law index
   - *Purpose*: Direct TDE physics test

5. **Flux Stability Metrics**
   - Point-to-point scatter
   - Monotonicity (fraction decreasing)
   - Noise ratio
   - Smoothness score
   - *Purpose*: TDEs are smooth, AGN are noisy

---

## Benchmark Results

### Individual Feature Group Impact (added to Set C baseline)

| Feature Group | OOF F1 | Delta | Fold Std | Verdict |
|--------------|--------|-------|----------|---------|
| Baseline (Set C) | 0.6811 | - | 0.0336 | Reference |
| + Structure Function | 0.6962 | +0.0151 | 0.0334 | **Best single add** |
| + Color-Magnitude | 0.6941 | +0.0130 | 0.0541 | High variance |
| + TDE Power Law | 0.6925 | +0.0114 | 0.0387 | Good stability |
| + Decline Consistency | 0.6914 | +0.0102 | 0.0459 | Moderate |
| + Flux Stability | 0.6889 | +0.0078 | 0.0476 | Minor gain |
| + All New | 0.6801 | -0.0010 | 0.0531 | Too noisy |

**Key Insight**: Adding ALL new features hurt performance - selective addition is crucial.

---

## Hypotheses for Future Work

### Why OOF != LB?

1. **Train/Test Distribution Shift**: Despite similar redshift distributions, test objects may have subtler differences in noise, cadence, or class proportions

2. **Threshold Sensitivity**: Optimal threshold varies between OOF and LB

3. **Feature Noise**: Features that improve training fit may capture training-specific noise

4. **Regularization Gap**: Models may need stronger regularization than what optimizes OOF

### Potential Directions

1. **Stronger Regularization**: Deliberately underfit to improve generalization
2. **Probability Calibration**: Adjust prediction distributions
3. **Adversarial Validation**: Identify train/test differences
4. **Pseudo-labeling**: Use high-confidence predictions to expand training
5. **Different Threshold Selection**: Optimize for stability, not max OOF

---

## File Structure

```
scripts/
├── analyze_feature_selection.py  # Feature importance analysis
├── benchmark_new_features.py     # Feature group benchmarking
├── train_v66-v80_*.py           # All experiment scripts
src/features/
├── high_snr_physics.py          # New physics features
├── plasticc_augmentation.py     # Augmentation (didn't help)
data/processed/
├── feature_selection_analysis.pkl
├── feature_sets.pkl
├── feature_benchmark_results.pkl
├── high_snr_features_cache.pkl
├── v77_artifacts.pkl            # Best LightGBM
├── v79_artifacts.pkl            # Curated features
├── v80_artifacts.pkl            # High-SNR features
```

---

## Best Models Summary

| Rank | Model | Features | OOF F1 | LB F1 | Status |
|------|-------|----------|--------|-------|--------|
| 1 | v34a XGBoost | 224 | 0.6667 | **0.6907** | Best LB |
| 2 | v79 Set C LGB | 70 | 0.6834 | 0.6891 | 2nd best LB |
| 3 | v77 LightGBM | 223 | 0.6886 | 0.6714 | Best Optuna LGB |
| 4 | v80c SF+TDE+Decline | 90 | 0.7000 | 0.6682 | At 0.70 OOF barrier |

---

*Last updated: January 2026*
