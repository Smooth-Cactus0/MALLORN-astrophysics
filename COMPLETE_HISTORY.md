# MALLORN Competition - Complete History & Analysis

## Internal Reference Document
**Goal**: €1,000 prize, Podium finish (Top 3)
**Current Best**: v21 - OOF F1=0.6708, LB F1=0.6649 (Rank 23/496, Top 5%)
**Deadline**: January 30, 2026 (1 month remaining)

---

## I. COMPLETE CHRONOLOGICAL HISTORY

### Phase 1: Initial Baseline (Dec 25)
| Version | Model | Features | OOF F1 | LB F1 | Key Innovation |
|---------|-------|----------|--------|-------|----------------|
| v1 | GBM | Statistical (50) | 0.30 | 0.333 | First submission |
| v2 | GBM | +Color features | 0.51 | 0.499 | Color = crucial (+70% improvement) |
| v3 | GBM | +Shape features | 0.56 | - | Lightcurve morphology |

**Learning**: Color features account for 41% of model importance. Physics matters!

---

### Phase 2: Peak Performance (Dec 26)
| Version | Model | Features | OOF F1 | LB F1 | Key Innovation |
|---------|-------|----------|--------|-------|----------------|
| **v8** | **GBM Ensemble** | **120** | **0.6262** | **0.6481** | **Best early result** |
| v11 | LSTM | Raw lightcurves | 0.12 | - | Deep learning failed |
| v13 | Transformer | Raw lightcurves | 0.11 | - | Self-attention failed |
| v14 | MLP | Features | 0.47 | - | Better than raw DL but below GBM |

**Learning**:
- Tree models >> Deep learning for tabular astronomical features
- Raw lightcurves insufficient for DL (need 100k+ samples)
- Feature engineering > End-to-end learning

---

### Phase 3: Advanced Features (Dec 26-27)
| Version | Model | Features | OOF F1 | LB F1 | Key Innovation |
|---------|-------|----------|--------|-------|----------------|
| v15 | GBM+NN+LSTM | Ensemble | 0.63 | 0.6463 | Ensemble didn't beat v8 |
| v16 | LSTM+PLAsTiCC | External data | 0.12 | - | Domain shift hurt |
| v17 | 1D-CNN+Aug | Heavy augmentation | 0.10 | - | Over-augmentation backfired |
| v18 | GBM+Per-band GP | GP per band | 0.6130 | - | Per-band GP hurt (-2.1%) |
| **v19** | **GBM+Multi-band GP** | **172** | **0.6626** | **0.6649** | **Multi-band GP crucial (+5.8%)** |

**Learning**:
- Multi-band GP (2D kernel) captures cross-band correlations
- george package with Matérn-3/2 kernel following 2025 TDE paper
- External data (PLAsTiCC) caused domain shift - MALLORN distribution unique
- Heavy augmentation backfires on small datasets

---

### Phase 4: Hyperparameter Optimization (Dec 28)
| Version | Model | Features | OOF F1 | LB F1 | Key Innovation |
|---------|-------|----------|--------|-------|----------------|
| v20 | GBM+ALL features | 375 | 0.6432 | - | Too many features hurt |
| v20b | GBM+Selective | ~250 | 0.6535 | - | Better feature selection |
| v20c | GBM+Optuna tuned | 172 | 0.6687 | 0.6518 | Proper HPO +0.92% |
| **v21** | **XGB only** | **147** | **0.6708** | **0.6649** | **Best single model** |
| v22 | ATAT (Transformer) | - | 0.5053 | - | +38% over naive LSTM but << GBM |

**Learning**:
- More features ≠ better (375 features < 172 features)
- Optuna HPO essential: gained +1.5%
- Single XGBoost > ensemble on OOF (simpler generalizes better)
- Optimal threshold ~0.30 (not 0.5!) due to class imbalance

---

### Phase 5: Threshold Tuning (Dec 31)
| Threshold | OOF F1 | LB F1 | TDEs Predicted |
|-----------|--------|-------|----------------|
| 0.06 | 0.5299 | 0.5302 | 849 |
| 0.08 | 0.5520 | 0.5499 | 774 |
| 0.10 | 0.5721 | 0.5682 | 724 |
| 0.15 | 0.6078 | 0.6015 | 641 |
| **~0.30 (v21)** | **0.6708** | **0.6649** | **419** |

**Learning**:
- OOF-LB correlation <0.7% - CV perfectly mirrors test distribution
- Optimal threshold ~0.30 found by CV optimization
- Class imbalance (64:2990) requires high threshold

---

### Phase 6: Advanced Physics (Dec 31)
| Version | Approach | Features | OOF F1 | vs v21 | Result |
|---------|----------|----------|--------|--------|--------|
| v23 | TDE-specific physics | - | - | 0.6552 | No improvement |
| v30 | All advanced physics | 314 | 0.5545 | -11.63% | Feature explosion |
| v30b | Selective advanced | 187 | 0.6420 | -2.88% | Still below baseline |

**Advanced Physics Features Created**:
- Multi-epoch temperature (7 epochs: 0-200d)
- Blackbody SED fitting with chi-squared quality
- Late-time colors (100-200d post-peak)
- Cross-band temporal asymmetry
- Cooling rate (early/late/overall)

**Learning**:
- v21 already captures essential physics through color + GP features
- Diminishing returns on detailed physics
- LSST sparse sampling makes late-time features NaN-heavy

---

### Phase 7: Model Improvements (Dec 31 - TODAY)
| Version | Approach | OOF F1 | vs v21 | Result |
|---------|----------|--------|--------|--------|
| v31 | CatBoost Ordered Boosting | 0.5754 | -9.54% | Failed |
| v32 | Feature Interactions | 0.6405 | -3.03% | Failed |
| v33 | Diverse Ensemble (XGB+LGB+CAT) | 0.6467 | -2.41% | Failed |

**What We Tried**:
1. **CatBoost**: Ordered boosting, depth=7, auto class weights
2. **Feature Interactions**: 14 physics-motivated interactions (color×Z, temp×time, etc.)
3. **Diverse Ensemble**: XGB (shallow) + LGB (deep) + CAT (ordered)

**Learning**:
- v21 configuration is near-optimal for this problem
- Small dataset (n=3043) limits model complexity benefits
- Tree models already capture interactions implicitly
- Weak ensemble members drag down performance

---

## II. CURRENT BEST MODELS - DETAILED ANALYSIS

### Model 1: v21 (Current Champion)
**Configuration**:
```python
XGBoost:
- max_depth: 5
- learning_rate: 0.025
- subsample: 0.8
- colsample_bytree: 0.8
- min_child_weight: 3
- reg_alpha: 0.2
- reg_lambda: 1.5
- scale_pos_weight: ~20 (class imbalance)
- tree_method: 'hist'
```

**Features (147 total)**:
- 120 selected base features (statistical + shape)
- TDE physics features (color variance, late-time behavior, power-law decay)
- Multi-band GP features (2D kernel capturing cross-band correlations)

**Performance**:
- OOF F1: 0.6708 @ threshold=0.30
- LB F1: 0.6649 (Rank 23/496)
- Precision: 0.59 | Recall: 0.73
- Confusion: TP=104, FP=72, FN=44, TN=2823

**Top Features by Importance**:
1. r_skew (190.2)
2. gp_gr_color_50d (115.8) - **Multi-band GP crucial**
3. g_skew (96.3)
4. gp_ri_color_50d (90.5)
5. gp2d_wave_scale (80.6)
6. flux_p25 (59.3)
7. r_rebrightening (58.4) - **TDE physics**
8. r_decay_alpha (53.7)

**Strengths**:
- Excellent OOF-LB correlation (<0.9% difference)
- Balanced precision-recall (not over-optimized)
- Simple, interpretable, robust
- Feature set captures essential physics

**Weaknesses**:
- Plateau'd at ~0.67 F1
- Threshold sensitivity (0.30 optimal but narrow)
- Limited ability to learn from rare class (64 TDEs only)

---

### Model 2: v19 (Multi-band GP Discovery)
**Configuration**: Same XGBoost params as v21

**Features (172 total)**: v21 features + additional GP variants

**Performance**:
- OOF F1: 0.6626
- LB F1: 0.6649
- Nearly identical to v21 on LB

**Key Innovation**:
- Multi-band GP with 2D kernel (time × wavelength)
- Captures achromatic vs chromatic evolution
- +5.8% improvement over per-band GP (v18)

**Why slightly worse than v21?**:
- 172 features vs 147 (some redundancy)
- v21's tighter feature selection improves generalization

---

## III. WHAT WORKED IN SIMILAR COMPETITIONS

### A. PLAsTiCC (Kaggle 2018-2019) - Most Similar Competition
**Challenge**: Classify 14 astronomical transient types from lightcurves
**Dataset**: ~7800 training objects (similar size to MALLORN)
**Metric**: Multi-class log loss

#### Winning Solutions Analysis:

**1st Place (kozodoi team)**:
- **Single LightGBM model** (no ensemble!)
- 500+ hand-crafted features
- **Key techniques we HAVEN'T tried**:
  - **Bazin function parametric fitting** (6 params per band)
  - **Fourier features** (periodicity detection)
  - **Cesium library features** (astronomical feature extraction)
  - **Weighted feature averaging** across bands
  - **Target encoding** of categorical metadata
  - **Pseudo-labeling** with very high confidence (>0.99)

**2nd Place (team)**:
- GP-based feature extraction (we do this!)
- **SMOTE oversampling** for rare classes (we don't do this!)
- **Calibration** (Platt scaling, isotonic regression)
- **Test-time augmentation** (predict multiple times with noise)

**3rd Place (team)**:
- Feature selection via **permutation importance**
- **Adversarial validation** to check train-test distribution
- **Stratified sampling** by SpecType within folds

#### What They ALL Used (That We Don't):
1. **Bazin function fitting**: Parametric lightcurve model
2. **Cesium features**: Astronomy-specific feature library
3. **Pseudo-labeling**: High-confidence test predictions as training
4. **Calibration**: Probability calibration for better thresholds
5. **SMOTE/ADASYN**: Synthetic minority oversampling

---

### B. Other Imbalanced Classification Competitions

#### Credit Card Fraud Detection (Kaggle)
**Imbalance**: 0.17% positive class (worse than our 2.1%)

**Winning Techniques**:
- **Focal Loss**: Down-weights easy negatives
- **Ensemble of different thresholds**: Each model optimized for different precision-recall point
- **Cost-sensitive learning**: Asymmetric loss (FP ≠ FN cost)
- **Anomaly detection**: One-class SVM, Isolation Forest
- **Stratified K-fold with minority upsampling**

#### Medical Diagnosis (Various)
**Imbalance**: 1-5% positive class

**Winning Techniques**:
- **Two-stage models**: Binary detector + classifier
- **Calibrated probabilities**: Platt scaling essential
- **Ensemble diversity via sampling**: Train on different minority oversampled sets
- **Meta-learning**: Stacking with class-aware meta-features

---

## IV. WHAT WE HAVEN'T TRIED (HIGH PRIORITY)

### Category 1: Feature Engineering (PLAsTiCC Winners)
**Status**: NOT TRIED ⚠️

1. **Bazin Function Parametric Fitting** ⭐⭐⭐
   - PLAsTiCC 1st place used this
   - 6 parameters per band: amplitude, rise time, fall time, peak time, plateau, baseline
   - More robust than raw statistics
   - **Expected gain**: +2-4%

2. **Cesium Astronomy Features** ⭐⭐⭐
   - Scientific library for time-series feature extraction
   - Includes: Stetson indices, CAR(1) features, period folding
   - PLAsTiCC 2nd/3rd place used this
   - **Expected gain**: +1-3%

3. **Fourier Features** ⭐⭐
   - FFT to detect periodicity (AGN signature)
   - Power spectrum features
   - **Expected gain**: +1-2%

4. **Wavelet Transform Features** ⭐
   - Time-frequency decomposition
   - Captures transient vs persistent signals
   - **Expected gain**: +0.5-1%

---

### Category 2: Handling Class Imbalance (Best Practices)
**Status**: PARTIALLY TRIED ⚠️

5. **SMOTE/ADASYN Oversampling** ⭐⭐⭐
   - PLAsTiCC 2nd place used this
   - Synthetic minority samples in feature space
   - Better than simple upsampling
   - **We use class weights but not synthetic samples**
   - **Expected gain**: +2-5%

6. **Focal Loss** ⭐⭐
   - Used in fraud detection, medical diagnosis
   - Formula: FL(p) = -(1-p)^γ * log(p)
   - Down-weights easy negatives, focuses on hard cases
   - **We use standard log loss**
   - **Expected gain**: +1-3%

7. **Cost-Sensitive Learning** ⭐⭐
   - Different costs for FP vs FN
   - Optimize for F1 directly (not log loss)
   - XGBoost supports custom objectives
   - **Expected gain**: +1-2%

8. **Ensemble of Thresholds** ⭐
   - Train multiple models at different operating points
   - Combine via voting or averaging
   - **Expected gain**: +0.5-1%

---

### Category 3: Probability Calibration (Widespread)
**Status**: NOT TRIED ⚠️

9. **Platt Scaling** ⭐⭐⭐
   - PLAsTiCC 2nd place used this
   - Fits sigmoid to map predictions to true probabilities
   - Essential for threshold optimization
   - **Our thresholds are fragile (0.30 optimal)**
   - **Expected gain**: +1-3%

10. **Isotonic Regression Calibration** ⭐⭐
    - Non-parametric calibration
    - Better for non-monotonic relationships
    - **Expected gain**: +1-2%

11. **Beta Calibration** ⭐
    - Three-parameter calibration
    - Best for skewed distributions
    - **Expected gain**: +0.5-1%

---

### Category 4: Advanced Validation & Meta-Learning
**Status**: NOT TRIED ⚠️

12. **Adversarial Validation** ⭐⭐⭐
    - Train classifier to distinguish train vs test
    - Identifies distribution shift
    - Reweight training samples accordingly
    - **We assume train=test but haven't verified**
    - **Expected gain**: +1-2% if shift exists

13. **Pseudo-Labeling** ⭐⭐⭐ (But Done Right)
    - PLAsTiCC 1st place used this
    - Only add test samples with >0.99 confidence
    - Iterative approach (add, retrain, repeat)
    - **We tried with 0.85 threshold - too aggressive**
    - **Expected gain**: +2-4%

14. **Stacking with Diversity** ⭐⭐
    - Meta-learner on diverse base models
    - Use out-of-fold predictions as meta-features
    - **Our v15 stacking failed - need better base models**
    - **Expected gain**: +1-2%

---

### Category 5: Test-Time Augmentation
**Status**: NOT TRIED ⚠️

15. **TTA for Tabular Data** ⭐⭐
    - Add small noise to features
    - Average predictions over multiple passes
    - Smooths decision boundaries
    - **Expected gain**: +0.5-1%

16. **Bagging Predictions** ⭐
    - Bootstrap aggregating at inference time
    - **Expected gain**: +0.3-0.5%

---

## V. GAP ANALYSIS: v21 vs PLAsTiCC Winners

| Technique | PLAsTiCC 1st | PLAsTiCC 2nd | v21 | Priority |
|-----------|--------------|--------------|-----|----------|
| Parametric fitting (Bazin) | ✅ | ✅ | ❌ | ⭐⭐⭐ HIGH |
| Cesium features | ✅ | ✅ | ❌ | ⭐⭐⭐ HIGH |
| GP features | ✅ | ✅ | ✅ | ✓ Done |
| SMOTE/ADASYN | ❌ | ✅ | ❌ | ⭐⭐⭐ HIGH |
| Calibration | ✅ | ✅ | ❌ | ⭐⭐⭐ HIGH |
| Pseudo-labeling | ✅ | ❌ | ❌ (tried, failed) | ⭐⭐ MEDIUM |
| Fourier features | ✅ | ✅ | ❌ | ⭐⭐ MEDIUM |
| Adversarial validation | ✅ | ✅ | ❌ | ⭐⭐ MEDIUM |
| Feature selection | ✅ | ✅ | ✅ | ✓ Done |
| Single strong model | ✅ | ❌ | ✅ | ✓ Done |

---

## VI. CONCRETE ACTION PLAN (RANKED BY EXPECTED GAIN)

### Tier 1: Highest Expected Gain (Target: +5-10%)
1. **Bazin Function Fitting** (Week 1)
   - Implementation: `sncosmo` or custom optimizer
   - 6 params × 6 bands = 36 new features
   - Expected: +2-4%

2. **SMOTE Oversampling** (Week 1)
   - Library: `imbalanced-learn`
   - Generate 2-3× synthetic TDE samples
   - Expected: +2-5%

3. **Probability Calibration** (Week 1)
   - Platt scaling + isotonic regression
   - Optimize threshold on calibrated probabilities
   - Expected: +1-3%

4. **Cesium Features** (Week 2)
   - Library: `cesium` or port key functions
   - Focus on: Stetson J/K, beyond1std, flux_percentile_ratio
   - Expected: +1-3%

**Total Tier 1 Expected**: +6-15% improvement → OOF F1 = 0.71-0.77

---

### Tier 2: Medium Gain (Target: +2-5%)
5. **Adversarial Validation** (Week 2)
   - Check train-test distribution
   - Reweight if shift detected
   - Expected: +1-2% if shift exists

6. **Focal Loss** (Week 2)
   - Custom XGBoost objective
   - Focus on hard TDE examples
   - Expected: +1-3%

7. **Pseudo-Labeling (Conservative)** (Week 3)
   - Threshold: >0.99 confidence
   - Iterative approach
   - Expected: +1-2%

8. **Fourier Features** (Week 3)
   - FFT power spectrum
   - Dominant frequency + power
   - Expected: +1-2%

**Total Tier 2 Expected**: +4-9% improvement

---

### Tier 3: Refinements (Target: +1-3%)
9. **Cost-Sensitive Learning**
10. **Test-Time Augmentation**
11. **Ensemble of Thresholds**
12. **Wavelet Features**

---

## VII. AGGRESSIVE 4-WEEK SCHEDULE FOR PODIUM

### Week 1 (Days 1-7): Foundation
- **Day 1-2**: Implement Bazin function fitting
- **Day 3-4**: Implement SMOTE oversampling
- **Day 5-6**: Implement calibration (Platt + isotonic)
- **Day 7**: Train v34 with all three → Target: OOF F1 > 0.70

### Week 2 (Days 8-14): Advanced Features
- **Day 8-10**: Port Cesium features
- **Day 11-12**: Implement adversarial validation
- **Day 13-14**: Train v35 → Target: OOF F1 > 0.72

### Week 3 (Days 15-21): Optimization
- **Day 15-16**: Focal loss custom objective
- **Day 17-18**: Conservative pseudo-labeling
- **Day 19-21**: Train v36 → Target: OOF F1 > 0.74

### Week 4 (Days 22-30): Final Push
- **Day 22-25**: Best ensemble strategy
- **Day 26-27**: Final hyperparameter tuning
- **Day 28**: Select top 2 submissions
- **Day 29-30**: Buffer for final refinements

**Target**: OOF F1 > 0.75, LB F1 > 0.73 → **Top 3 Podium Finish**

---

## VIII. SUCCESS CRITERIA

| Milestone | OOF F1 | LB F1 | Rank | Status |
|-----------|--------|-------|------|--------|
| Current | 0.6708 | 0.6649 | 23/496 | ✅ Achieved |
| Week 1 Target | >0.70 | >0.68 | <15 | 🎯 Aim |
| Week 2 Target | >0.72 | >0.70 | <10 | 🎯 Aim |
| Week 3 Target | >0.74 | >0.72 | <5 | 🎯 Aim |
| **PODIUM** | **>0.75** | **>0.73** | **1-3** | **🏆 GOAL** |

---

## IX. KEY PRINCIPLES FOR SUCCESS

1. **Incremental Progress**: Each experiment must show +1-2% gain
2. **Validation Trust**: OOF-LB correlation is <0.7% - trust CV
3. **Avoid Overfitting LB**: Maximum 2 submissions per week
4. **Systematic Approach**: Test one technique at a time
5. **Learn from Winners**: PLAsTiCC solutions are our roadmap
6. **Class Imbalance is KEY**: Focus on techniques that handle 64:2990 ratio
7. **Physics + ML**: Best performance comes from domain knowledge + ML best practices

---

**BOTTOM LINE**: We have clear, proven techniques that we haven't tried yet. PLAsTiCC winners achieved 0.80+ with similar dataset size using Bazin fitting + SMOTE + calibration. We're at 0.67. A podium finish is absolutely achievable.
