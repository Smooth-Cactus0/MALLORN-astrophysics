# MALLORN PODIUM ROADMAP - 8 Critical Techniques

**Goal**: Top 3 Finish (€1,000 prize)
**Current**: v34a - OOF F1=0.6667, LB F1=0.6907 (NEW BEST +2.58%)
**Previous Best**: v21 - OOF F1=0.6708, LB F1=0.6649 (Rank 23/496)
**Target**: OOF F1 > 0.75, LB F1 > 0.73 (Top 3)
**Deadline**: January 30, 2026
**Last Updated**: December 31, 2024

---

## MASTER CHECKLIST: 8 PROVEN TECHNIQUES

### ✅ Tier 1: Foundation (Week 1) - Target: +6-15%

#### 1. Bazin Function Parametric Fitting ⭐⭐⭐
- **Status**: ✅ COMPLETE (v34a)
- **Actual Gain**: +2.58% LB (+0.0258 F1)
- **Result**: LB F1 = 0.6907 (NEW BEST)
- **PLAsTiCC**: 1st & 2nd place both used
- **Implementation**:
  - Model: `f(t) = A * exp(-(t-t0)/τ_fall) / (1 + exp(-(t-t0)/τ_rise)) + B`
  - 8 features per band: A, t0, τ_rise, τ_fall, B, chi2, rise_fall_ratio, peak_flux
  - Library: `scipy.optimize.curve_fit`
  - Output: 52 features (6 bands × 8 + 4 cross-band)
- **Why it works**: Parametric model more robust than raw statistics on sparse lightcurves
- **Key Finding**: Bazin features = 24.8% of model importance, top feature at rank #4
- **Version**: v34a
- **Completed**: December 31, 2024

#### 2. SMOTE/ADASYN Oversampling ⭐⭐⭐
- **Status**: ❌ SKIPPED (not viable for our data)
- **Attempted**: v34b (25% target), v34b_conservative (10% target)
- **Results**: Both failed (-1.8% to -4.5% vs v34a)
- **PLAsTiCC**: 2nd place used (but had more minority samples)
- **Implementation**:
  - Library: `imbalanced-learn`
  - Methods: `SMOTE` (basic), `ADASYN` (adaptive)
  - Strategy: Generate 2-3× synthetic TDE samples (64 → 128-192)
  - Apply in feature space, not raw lightcurves
- **Why it works**: Model learns from 3× more minority class examples
- **Current issue**: Only 64 TDEs vs 2990 non-TDEs (2.1% positive)
- **Version**: v34b
- **Timeline**: Days 3-4

#### 3. Probability Calibration ⭐⭐⭐
- **Status**: ⚠️ TESTED (improved OOF but not LB)
- **Result**: v34c - OOF F1=0.6748 (+1.2%), LB F1=0.6698 (-2.1% vs v34a)
- **PLAsTiCC**: 1st & 2nd place both used
- **Implementation**:
  - Method 1: Platt Scaling (sigmoid fit)
  - Method 2: Isotonic Regression (non-parametric)
  - Method 3: Beta Calibration (for skewed distributions)
  - Library: `sklearn.calibration.CalibratedClassifierCV`
  - Apply to OOF predictions, then test
- **Why it works**: Maps model scores → true probabilities, stabilizes threshold
- **Current issue**: Threshold=0.30 is fragile
- **Version**: v34c
- **Timeline**: Days 5-6

#### 4. Cesium Astronomy Features ⭐⭐⭐
- **Status**: 🔄 NEXT (building on v34a)
- **Expected Gain**: +1-3%
- **PLAsTiCC**: All top 3 used
- **Target**: LB F1 > 0.71
- **Implementation**:
  - Library: `cesium` or port key functions
  - Key features:
    - `stetson_j`, `stetson_k` (correlated variability)
    - `beyond_1_std` (fraction of points >1σ from mean)
    - `flux_percentile_ratio_mid20`, `_mid35`, `_mid50`, `_mid65`, `_mid80`
    - `percent_amplitude` (max flux / median flux)
    - `maximum_slope` (steepest rise/fall)
    - `linear_trend` (linear fit slope)
    - `anderson_darling` (normality test)
- **Why it works**: Astronomy-specific domain knowledge
- **Version**: v35a
- **Timeline**: Week 2, Days 8-10

---

### ✅ Tier 2: Optimization (Week 2-3) - Target: +4-9%

#### 5. Adversarial Validation ⭐⭐⭐
- **Status**: ✅ COMPLETE (v39b) - Ready for LB test
- **Actual Gain**: +0.32% OOF (+0.0021 F1)
- **Result**: v39b OOF F1 = 0.6688, LB = TBD
- **PLAsTiCC**: 1st & 3rd place used
- **Implementation**:
  - Trained classifier on v4 baseline features (267 features)
  - AUC = 0.6097 (SIGNIFICANT distribution shift detected)
  - Created adversarial sample weights: [1.34, 1.92], mean = 1.52
  - Applied weights to XGBoost training via DMatrix weight parameter
- **Key Findings**:
  - **all_rise_time**: Test +66.4% longer (267 → 445 days)
  - **all_asymmetry**: Test +80.5% higher (28.14 → 50.80)
  - Metadata (Z, EBV) identical, but derived temporal features differ dramatically
- **Root Cause**: Systematic bias in lightcurve temporal characteristics
- **Model Impact**: Small OOF improvement, predicts 414 TDEs (vs 420 for v34a)
- **Version**: v39b (Bazin + adversarial weights)
- **Completed**: January 1, 2026

#### 6. Focal Loss ⭐⭐
- **Status**: ⏳ PENDING
- **Expected Gain**: +1-3%
- **Used in**: Fraud detection, medical diagnosis, object detection
- **Implementation**:
  - Formula: `FL(p_t) = -(1 - p_t)^γ * log(p_t)` where γ=2
  - Custom XGBoost objective function
  - Down-weights easy negatives (high confidence non-TDEs)
  - Up-weights hard examples (low confidence TDEs)
- **Why it works**: Forces model to focus on rare, hard-to-classify TDEs
- **Better than**: Standard log loss (treats all examples equally)
- **Version**: v36a
- **Timeline**: Week 3, Days 15-16

#### 7. Conservative Pseudo-Labeling ⭐⭐
- **Status**: ⏳ PENDING (v28b failed with 0.85 threshold)
- **Expected Gain**: +1-2%
- **PLAsTiCC**: 1st place used with >0.99 threshold
- **Implementation**:
  - Step 1: Train on labeled data
  - Step 2: Predict test set
  - Step 3: Add only >0.99 confidence predictions to training
  - Step 4: Retrain with expanded dataset
  - Step 5: Repeat 2-3 times
- **Why previous attempt failed**: 0.85 threshold too aggressive (190 TDEs + 950 non-TDEs)
- **New strategy**: Start with 0.99, expect ~20-50 high-confidence samples
- **Version**: v36b
- **Timeline**: Week 3, Days 17-18

#### 8. Fourier Features ⭐⭐
- **Status**: 🔄 IN PROGRESS (v39a)
- **Expected Gain**: +1-2%
- **PLAsTiCC**: 1st & 2nd place used
- **Implementation**:
  - FFT per band: `np.fft.fft(fluxes)`
  - Interpolate to uniform time grid (FFT requirement)
  - Apply Hanning window to reduce spectral leakage
  - Features extracted:
    - `dominant_frequency` (peak in power spectrum)
    - `dominant_power` (height of peak)
    - `power_ratio` (peak / mean power)
    - `spectral_entropy` (randomness in spectrum)
  - 4 features × 6 bands = 24 features
- **Why it works**: Detects periodicity (AGN signature vs aperiodic TDE/SN)
- **Version**: v39a
- **Status**: Ready to train
- **Timeline**: January 1, 2026

---

### ✅ Tier 3: Physics-Based Innovation (Week 3-4) - Target: +3-8%

#### 9. Custom TDE Physics Parametric Model ⭐⭐⭐⭐
- **Status**: ⏳ NEXT (after Cesium results)
- **Expected Gain**: +2-5% (potentially higher than Bazin)
- **Motivation**: Bazin achieved +2.58% but was designed for SNe, not TDEs
- **Implementation**:
  - Design TDE-specific parametric function based on physics:
    - **Elliptical accretion disk model**: Varying eccentricity evolution
    - **Super-Eddington accretion**: L(t) ~ Ṁ(t) with viscous timescale
    - **Reprocessing model**: UV → optical conversion with time lag
    - **Alternative forms**:
      1. **Guillochon model**: L ~ t^(-5/3) power law with exponential cutoff
      2. **Piecewise model**: Linear rise + exponential decay (asymmetric)
      3. **Double-exponential**: τ_rise_early + τ_rise_late for stalled rise
  - Parameters per band: 5-7 parameters describing TDE-specific physics
  - Library: `scipy.optimize.curve_fit` (same as Bazin)
  - Output: 35-50 features (5-7 params × 6 bands + cross-band)
- **Why it could work better**:
  - Bazin assumes symmetric exponential decay (SN-like)
  - TDEs have distinctive t^(-5/3) fallback rate from relativistic dynamics
  - TDEs show stalled rise (circularization timescale) not captured by Bazin
  - Custom model can encode viscous timescale and Eddington ratio evolution
- **Research sources**:
  - Guillochon & Ramirez-Ruiz (2013): t^(-5/3) power law
  - Lodato & Rossi (2011): Super-Eddington accretion disk model
  - van Velzen et al. (2020): Observational TDE lightcurve templates
- **Version**: v37a (after testing other techniques)
- **Timeline**: Week 3-4, Days 19-22

**Note**: This is a high-risk, high-reward approach. If it works, could be significant contribution beyond just competition success. Will implement after validating Cesium features (Technique #4) to ensure we have stable foundation.

---

## PROGRESS TRACKING

| Week | Days | Techniques | Target OOF | Target LB | Actual LB | Status |
|------|------|------------|-----------|-----------|-----------|--------|
| **Baseline** | - | v21 | 0.6708 | 0.6649 | 0.6649 | ✅ Complete |
| **Dec 31** | 1 | #1 (Bazin) | >0.70 | >0.68 | **0.6907** | ✅ **BEAT TARGET!** |
| **Next** | 2 | #4 (Cesium) | >0.70 | >0.71 | TBD | 🔄 In Progress |
| **Week 2** | 3-7 | #5, #8 (AdvVal, Fourier) | >0.72 | >0.72 | TBD | ⏳ Pending |
| **Week 3** | 8-14 | #6-7 (Focal, Pseudo) | >0.74 | >0.74 | TBD | ⏳ Pending |
| **Week 4** | 15-30 | Ensemble + Final HPO | >0.75 | >0.75 | TBD | ⏳ Pending |

---

## VERSION NAMING CONVENTION

- **v34a**: Bazin fitting
- **v34b**: Bazin + SMOTE
- **v34c**: Bazin + SMOTE + Calibration (Week 1 complete)
- **v35a**: v34c + Cesium features
- **v35b**: v35a + Adversarial validation
- **v35c**: v35b + Fourier features (Week 2 complete)
- **v36a**: v35c + Focal loss
- **v36b**: v36a + Pseudo-labeling (Week 3 complete)
- **v37**: Best ensemble (Week 4)

---

## CUMULATIVE GAIN PROJECTION

| Version | Techniques Added | Expected Cumulative F1 | LB Rank Estimate |
|---------|------------------|------------------------|------------------|
| v21 | Baseline | 0.6708 | 23/496 |
| v34a | +Bazin | 0.69-0.70 | 15-20 |
| v34b | +SMOTE | 0.71-0.73 | 8-15 |
| v34c | +Calibration | 0.72-0.74 | 5-10 |
| v35a | +Cesium | 0.73-0.75 | 3-8 |
| v35b | +AdvVal | 0.73-0.76 | 2-6 |
| v35c | +Fourier | 0.74-0.76 | 2-5 |
| v36a | +Focal | 0.75-0.77 | 1-4 |
| v36b | +Pseudo | 0.76-0.78 | 1-3 🏆 |

**Conservative estimate**: 0.75-0.76 → **Top 3 PODIUM**
**Optimistic estimate**: 0.77-0.78 → **1st PLACE**

---

## SUCCESS CRITERIA

✅ **COMPLETE** when ALL 8 techniques implemented and evaluated
🏆 **PODIUM** when OOF F1 > 0.75 AND LB F1 > 0.73
🥇 **WINNER** when LB F1 > 0.75

---

## RISK MITIGATION

1. **If technique fails** (gain <0.5%):
   - Document why it failed
   - Move to next technique
   - Don't combine with other features if it hurts

2. **If overfitting detected** (OOF >> LB):
   - Increase regularization
   - Remove technique
   - Use calibration

3. **If stuck** (no progress for 3 days):
   - Review PLAsTiCC solutions in detail
   - Ask for help in competition discussion
   - Try simpler version of technique

4. **Leaderboard usage**:
   - Maximum 2 submissions per week
   - Only submit versions that show +2% OOF improvement
   - Save final submission slots for Week 4

---

## KEY PRINCIPLES

1. **One technique at a time**: Isolate gains
2. **Trust CV**: OOF-LB correlation is <0.7%
3. **Incremental progress**: Each step must show +1-2%
4. **Learn from winners**: PLAsTiCC roadmap is proven
5. **Class imbalance focus**: All techniques handle 64:2990 ratio
6. **Systematic approach**: Track everything in this document

---

**CURRENT TASK**: Implement Technique #1 (Bazin Fitting) → v34a
**NEXT UPDATE**: After v34a training complete (Day 2)
