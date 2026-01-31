# MALLORN Astrophysics Competition - Experiment Summary

**Competition**: MALLORN Astronomical Classification Challenge
**Goal**: Classify nuclear transients into TDEs vs non-TDEs
**Metric**: F1 Score
**Best Result**: v92d with LB F1 = 0.6986

---

## Best Submissions (Kept in /submissions/)

| Rank | Version | LB F1 | OOF F1 | Description |
|------|---------|-------|--------|-------------|
| 1 | **v92d_baseline_adv** | **0.6986** | 0.6688 | XGBoost with adversarial validation weights |
| 2 | v34a_bazin | 0.6907 | 0.6667 | XGBoost with Bazin fit features |
| 3 | v79_set_c_extended_xgb | 0.6891 | 0.6834 | XGBoost with 70 physics features |
| 4 | v105b_interactions | 0.6862 | 0.6832 | XGBoost with cross-feature interactions |
| 5 | v106b_alpha04 | 0.6851 | 0.6435 | MixUp augmentation (alpha=0.4) |
| 6 | v104_seed_ensemble | 0.6811 | 0.6866 | Seed ensemble (10 seeds averaged) |

---

## Key Findings

### What Worked
1. **Adversarial validation weighting (v92d)** - Best approach, addresses train/test distribution shift
2. **Bazin function fitting (v34a)** - Physical model for lightcurve shape
3. **Physics-based feature engineering** - GP features, color evolution, temperature estimation
4. **Feature selection** - Fewer, higher-quality features generalized better

### What Didn't Work (Higher OOF = Worse LB)
1. **More features** - Adding features consistently hurt LB despite improving OOF
2. **Ensembling** - Combined models overfit more than single models
3. **Deep learning** - LSTM/GRU/Transformer all failed (best F1 ~0.50)
4. **Knowledge distillation** - v108b had best OOF (0.6925) but worst LB (0.6325)
5. **Pseudo-labeling** - Test leakage risk, didn't generalize

### Critical Pattern
**Higher OOF F1 often means WORSE LB F1** - The train/test distribution shift is severe.

---

## Experiment Versions (v81-v108)

### v81: Stronger Regularization
- **Goal**: Reduce overfitting with higher regularization
- **Method**: Increased reg_alpha/reg_lambda, reduced tree depth
- **Result**: Did not improve LB

### v82: Threshold Calibration
- **Goal**: Optimize decision threshold
- **Method**: Tested thresholds from 0.05 to 0.50
- **Result**: Optimal threshold ~0.07-0.10 due to class imbalance

### v83: Feature Noise Injection
- **Goal**: Regularize by adding noise to features during training
- **Method**: Gaussian noise with various sigma values
- **Result**: Did not improve generalization

### v84: Combined Best Approaches
- **Goal**: Combine successful techniques
- **Method**: Mixed best features + regularization + adversarial weights
- **Result**: Marginal improvement

### v85: Feature Removal Analysis
- **Goal**: Identify harmful features
- **Method**: Systematic ablation study
- **Result**: Some features were hurting LB

### v86: High OOF Investigation
- **Goal**: Understand why high OOF doesn't transfer
- **Method**: Analyzed feature importance shifts
- **Result**: Confirmed distribution shift issue

### v87: Structure Function + Adversarial
- **Goal**: Add AGN variability characterization
- **Method**: Structure function features with adversarial weighting
- **Result**: OOF improved but LB dropped

### v88: Optuna + Adversarial
- **Goal**: Full hyperparameter optimization with adversarial weights
- **Method**: Optuna with 100 trials
- **Result**: Best OOF for XGBoost, LB similar to v92d

### v89: SMOTE/ADASYN Oversampling
- **Goal**: Handle class imbalance through oversampling
- **Method**: SMOTE and ADASYN on minority class
- **Result**: Hurt generalization

### v90: Focal Loss
- **Goal**: Better handle class imbalance
- **Method**: Custom focal loss implementation
- **Result**: Mixed results, not better than baseline

### v91: Focal + Weighted Sampling
- **Goal**: Combine focal loss with sample weights
- **Method**: Focal loss + class weights
- **Result**: Similar to v90

### v92: Focal + Adversarial (BEST)
- **Goal**: Combine focal loss intuition with adversarial weights
- **Method**: XGBoost with adversarial sample weights
- **Result**: **v92d achieved LB 0.6986 (BEST)**
- **Key**: Adversarial weights were the winning factor

### v93: Easy Ensemble
- **Goal**: Bagging with undersampling
- **Method**: Multiple weak learners on balanced subsets
- **Result**: Did not improve over v92d

### v94-v97: Pseudo-Labeling Variations
- **Goal**: Semi-supervised learning with test predictions
- **Methods**: Hard labels, soft labels, various ratios
- **Results**: All versions had LB < 0.66, potential test leakage issues

### v98: AutoGluon
- **Goal**: Automated ML for best model selection
- **Method**: AutoGluon with presets
- **Result**: OOF 0.59, much worse than manual tuning

### v99: CatBoost
- **Goal**: Alternative gradient boosting
- **Method**: CatBoost with Optuna tuning
- **Result**: OOF 0.6553, higher recall but lower precision

### v100: LightGBM
- **Goal**: Alternative gradient boosting
- **Method**: LightGBM with Optuna tuning
- **Result**: LB 0.6436, worse than XGBoost

### v101: Multi-Algorithm Ensemble
- **Goal**: Combine XGBoost + CatBoost + LightGBM
- **Methods**: Average, weighted, rank averaging
- **Result**: v101a LB 0.6784 - ensembling hurt

### v102: Label Smoothing
- **Goal**: Regularization through soft labels
- **Method**: Replace 0/1 with epsilon/1-epsilon
- **Result**: OOF 0.6540, regularization too strong

### v103: CV Pseudo-Labels
- **Goal**: Self-training with OOF predictions
- **Method**: Blend hard labels with teacher OOF predictions
- **Result**: LB 0.6448, didn't generalize

### v104: Seed Ensemble
- **Goal**: Reduce variance by averaging multiple seeds
- **Method**: Train same model with 10 different random seeds
- **Result**: OOF 0.6866, LB 0.6811 - helped stability

### v105: Cross-Feature Interactions
- **Goal**: Capture physics relationships
- **Method**: 49 interaction features (color×temp, flux ratios, z-corrections)
- **Result**: OOF 0.6832, LB 0.6862 - good generalization

### v106: MixUp Augmentation
- **Goal**: Data augmentation for regularization
- **Method**: Linear interpolation of samples and labels
- **Result**: OOF 0.6435 (worse), but LB 0.6851 (surprisingly good!)
- **Insight**: Lower OOF sometimes means better generalization

### v107: Platt Scaling
- **Goal**: Better probability calibration
- **Method**: Logistic regression on OOF predictions
- **Result**: Minimal improvement (+0.20% for v105)

### v108: Knowledge Distillation
- **Goal**: Transfer knowledge from teacher to student
- **Method**: Train student on teacher's soft predictions
- **Results**:
  - v108b: OOF 0.6925 (BEST OOF!) → LB 0.6325 (WORST!)
  - v108e: OOF 0.6903 → LB 0.6472
- **Insight**: Perfect example of OOF-LB inverse relationship

---

## Feature Engineering Summary

### Successful Features
- Multi-band Gaussian Process fits (length scales, amplitudes)
- Color evolution (g-r, r-i at peak and post-peak)
- Bazin function parameters
- Adversarial validation weights
- Redshift and extinction corrections

### Unsuccessful Features
- Cesium time-series features (added noise)
- PLAsTiCC augmentation (domain shift)
- Structure functions (overfit)
- Too many physics features (overfit)

---

## Model Performance Summary

| Algorithm | Best OOF | Best LB | Notes |
|-----------|----------|---------|-------|
| XGBoost | 0.6925 | 0.6986 | Best overall, use adversarial weights |
| LightGBM | 0.6886 | 0.6714 | Good but less stable than XGBoost |
| CatBoost | 0.6553 | ~0.63 | Higher recall, lower precision |
| LSTM/GRU | 0.50 | N/A | Deep learning failed |
| Transformer | 0.50 | N/A | Deep learning failed |
| AutoGluon | 0.59 | N/A | Worse than manual tuning |

---

## Lessons Learned

1. **Trust adversarial validation** - Identifies train/test shift
2. **Simpler is better** - Fewer features, simpler models generalize better
3. **OOF is not reliable** - High OOF often means overfitting
4. **XGBoost wins** - More stable than LightGBM/CatBoost for this problem
5. **Deep learning fails** - Not enough data, too much class imbalance
6. **Ensembling hurts** - Combines overfitting from multiple models
7. **Physics features help** - But only a curated subset

---

## Files in This Archive

### scripts/
- train_v81_stronger_regularization.py
- train_v82_threshold_calibration.py
- train_v83_feature_noise.py
- train_v84_combined_best.py
- train_v85_feature_removal.py
- train_v86_high_oof.py
- train_v87_sf_adv.py
- train_v88_optuna_adv.py
- train_v89_smote_adasyn.py
- train_v90_focal_loss.py
- train_v91_focal_weighted.py
- train_v92_focal_adversarial.py
- train_v93_easy_ensemble.py
- train_v94_pseudo_labeling.py
- train_v95_pseudo_fixed.py
- train_v96_pseudo_ratio.py
- train_v97_soft_pseudo.py
- train_v98_autogluon.py
- train_v99_catboost.py
- train_v100_lightgbm.py
- train_v101_ensemble.py
- train_v102_label_smoothing.py
- train_v103_cv_pseudo.py
- train_v104_seed_ensemble.py
- train_v105_interactions.py
- train_v106_mixup.py
- train_v107_platt_scaling.py
- train_v108_knowledge_distillation.py
- error_analysis.py
- adversarial_validation.py

### submissions_archive/
- 237 submission files from various experiments

---

## Recommendations for Future Work

1. **Stick with v92d approach** - Adversarial weighting is the key
2. **Focus on feature quality, not quantity** - Remove noisy features
3. **Test on private LB proxy** - Create validation set mimicking test distribution
4. **Avoid high-OOF traps** - Lower OOF with MixUp gave better LB
5. **Consider model simplicity** - Deeper trees and more features = overfitting

---

*Generated: January 18, 2026*
