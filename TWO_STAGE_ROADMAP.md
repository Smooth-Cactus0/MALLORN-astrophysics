# MALLORN Two-Stage Classifier Improvement Roadmap

**Current Best**: v60a (LB 0.69) using two-stage approach with cautious AGN filtering

## Executive Summary

The two-stage classifier achieves significant improvement over baseline (LB 0.6907 → 0.69) by:
1. **Stage 1**: AGN vs Rest classifier (97.4% accuracy)
2. **AGN Filter**: Remove objects with AGN_prob >= 0.99 (loses only 4 TDEs)
3. **Stage 2**: TDE vs Rest classifier on filtered set

**Error Analysis Results:**
- Stage 1: Near-perfect (4 TDEs misclassified, 2.7% loss)
- Stage 2: Bottleneck (43 TDEs missed, 29.9% FN rate)

---

## Stage 1: AGN Classifier Analysis

### Current Performance
- **Threshold**: 0.99
- **TDEs Lost**: 4 / 148 (2.7%)
- **AGN Filtered**: ~1650 / 1786 (92.4%)

### The 4 Problematic TDEs
These TDEs have AGN-like features:
| Object ID | AGN_prob |
|-----------|----------|
| hervess_hend_lam | ~0.99+ |
| Mithrim_alae_cund | ~0.99+ |
| cabed_mereth_iphant | ~0.99+ |
| yll_merilin_rach | ~0.99+ |

### Feature Comparison (Misclassified vs Correct TDEs)
| Feature | Misclassified TDEs | Correct TDEs | Interpretation |
|---------|-------------------|--------------|----------------|
| gp2d_time_wave_ratio | 4.1 | 34.2 | Low ratio = AGN-like behavior |
| r_skew | 0.56 | 2.0 | Lower skewness = more symmetric |
| gp2d_wave_scale | 397 | 126 | Higher = less wavelength-coherent |

### Potential Improvements

#### Priority 1: TDE-Specific Physics Features
Add features that capture TDE physics that AGN cannot mimic:
- **Temperature consistency**: TDEs maintain ~20,000-40,000K, AGN fluctuate
- **Rise coherence**: TDEs have coherent rise across all bands
- **Color stability**: TDEs maintain blue colors (g-r < 0), AGN colors vary

#### Priority 2: Lower Threshold with Safety Net
- Current: 0.99 threshold
- Consider: 0.97 threshold + secondary TDE rescue classifier
- Trade-off: Filter more AGN but need to rescue the ~4-8 borderline TDEs

#### Priority 3: Ensemble AGN Classifier
- Train multiple AGN classifiers with different feature subsets
- Use voting/averaging to be more robust
- May reduce false positive rate on TDEs

---

## Stage 2: TDE vs SN Classifier Analysis (MAIN FOCUS)

### Current Performance (on filtered set)
- **TDE Recall**: 70.1% (101/144 TDEs detected)
- **TDEs Missed**: 43 (false negatives)
- **False Positives**: 61 non-TDEs predicted as TDE

### False Positive Breakdown by Type
| SpecType | Count | % of FPs | Notes |
|----------|-------|----------|-------|
| **AGN** | 28 | 45.9% | AGN that slipped through Stage 1 |
| **SN IIn** | 13 | 21.3% | Type IIn SNe often mimic TDEs! |
| **SN II** | 9 | 14.8% | Core-collapse SNe |
| **SN Ia** | 5 | 8.2% | Thermonuclear SNe |
| **SLSN-I** | 4 | 6.6% | Superluminous SNe |
| **SN Ib/c** | 2 | 3.3% | Stripped-envelope SNe |

### Key Insight: SN IIn Problem
**SN IIn (Type IIn Supernovae)** are the second biggest source of false positives after AGN.

Why SN IIn mimic TDEs:
- Both show narrow emission lines (IIn = "narrow")
- Both can have slow decline rates
- Both can be blue at peak
- Both occur in galaxy nuclei sometimes

**Distinguishing features we should add:**
1. **Hydrogen emission line equivalent width** (IIn stronger)
2. **Rise time** (TDEs typically rise faster)
3. **Late-time color evolution** (IIn become redder faster)
4. **Host galaxy properties** (IIn in star-forming regions)

### Top TDE-Discriminating Features (Stage 2)
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | gp2d_wave_scale | 0.0398 |
| 2 | gp_flux_g_50d | 0.0383 |
| 3 | gp_ri_color_50d | 0.0367 |
| 4 | gp2d_time_wave_ratio | 0.0342 |
| 5 | gp_gr_color_50d | 0.0318 |

**Observation**: GP-based features dominate! Multi-band Gaussian Process interpolation is crucial.

---

## Improvement Action Plan

### Phase 1: Quick Wins (v61)
**Goal**: Reduce false positives from residual AGN and SN IIn

1. **Tighter AGN threshold in Stage 2**
   - Add AGN_prob as a feature to Stage 2 classifier
   - Let the model learn to down-weight high-AGN-prob objects

2. **SN IIn-specific features**
   - Rise time ratio (rise_time / decline_time)
   - Color change rate in first 30 days
   - Peak-to-50d flux ratio

3. **Re-test earlier features on filtered dataset**
   - Power law features (v55) - may help distinguish SN cooling
   - Extinction-corrected colors (v57) - may help with SN IIn

### Phase 2: Feature Engineering (v62)
**Goal**: Better TDE vs SN separation

1. **Temperature evolution features**
   - T_peak, T_30d, T_50d, T_100d
   - Cooling rate (dT/dt) - SNe cool, TDEs stay hot
   - Temperature variance

2. **Rise-to-decline asymmetry**
   - TDEs: Often symmetric or slow decline
   - SNe: Usually faster decline than rise
   - Feature: rise_time / decline_time ratio

3. **Late-time behavior**
   - Flux ratio: F_100d / F_peak
   - TDEs decline as t^(-5/3) (canonical law)
   - SNe decline faster (exponential for Ia)

### Phase 3: Model Architecture (v63)
**Goal**: Optimize two-stage pipeline

1. **Cascade classifier**
   - Stage 1: AGN filter (current)
   - Stage 2a: SN IIn filter (new!)
   - Stage 2b: TDE vs remaining

2. **Probability calibration**
   - Calibrate Stage 1 and Stage 2 probabilities
   - Use Platt scaling or isotonic regression
   - Better threshold selection

3. **Soft two-stage**
   - Don't hard-filter, use AGN_prob as weight
   - Final_prob = TDE_prob × (1 - AGN_prob)^α
   - Tune α on validation set

### Phase 4: Ensemble (v64)
**Goal**: Combine best approaches

1. **Multi-model ensemble**
   - v60a (hard filter two-stage)
   - v60b (soft combination)
   - v34a (baseline)
   - Rank averaging

2. **Threshold optimization**
   - Different thresholds for different models
   - Optimize ensemble weights on OOF

---

## Specific Next Steps

### Immediate (v61a)
```python
# Add to Stage 2 training:
1. Include agn_prob as a feature
2. Add rise_time / decline_time ratio
3. Add 30d color change rate
4. Re-test power law fit residuals
```

### This Week (v61b-v62)
```python
# Temperature evolution
1. Extract T at multiple epochs (peak, 30d, 50d, 100d)
2. Compute cooling rate
3. Add t^(-5/3) power law fit quality
```

### Validation Strategy
- Always compare to v34a baseline (OOF 0.6667, LB 0.6907)
- Always compare to v60a (OOF 0.6815, LB 0.69)
- Submit promising improvements to LB

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Stage 1 threshold too aggressive | Low | High | Use 0.99 (conservative) |
| SN IIn features overfit | Medium | Medium | Cross-validate carefully |
| Ensemble overfits OOF | Medium | High | Use rank averaging |
| Feature count explosion | Medium | Medium | Limit to top 250 features |

---

## Success Metrics

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| OOF F1 | 0.6815 | 0.70 | 0.72 |
| LB F1 | 0.69 | 0.71 | 0.73 |
| TDE Recall | 70.1% | 75% | 80% |
| False Positives | 61 | 50 | 40 |

---

## Timeline

| Week | Version | Focus |
|------|---------|-------|
| Week 1 | v61a | Quick wins (AGN prob + rise ratio) |
| Week 1 | v61b | Re-test power law on filtered data |
| Week 2 | v62 | Temperature evolution features |
| Week 2 | v63 | Cascade classifier (SN IIn filter) |
| Week 3 | v64 | Ensemble optimization |
| Week 4 | Final | Best ensemble submission |

---

## References

- Error analysis: `scripts/analyze_two_stage_errors.py`
- Two-stage baseline: `scripts/train_v60_two_stage.py`
- Best single model: `scripts/train_v34a_bazin.py` (cached features)
