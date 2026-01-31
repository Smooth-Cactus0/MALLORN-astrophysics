# CLAUDE.md - MALLORN Astronomical Classification Challenge

## Competition Result

**Final Ranking**: 4th / 894 (Top 0.5%)

**Goal**: Binary classification - identify **TDEs** (Tidal Disruption Events) vs non-TDEs (Supernovae + AGN)

| Aspect | Details |
|--------|---------|
| **Metric** | F1 Score |
| **Prize** | EUR 1,000 |
| **Final Ranking** | 4th / 894 |
| **Winning Model** | v92d XGBoost + Adversarial Validation |
| **Private LB F1** | 0.6684 |
| **Public LB F1** | 0.6986 |

## Solution Summary

The winning approach was a **single XGBoost model** with **adversarial validation sample weights** and **222 physics-informed features**. No ensembling, no Optuna tuning, no multi-seed averaging. See [SOLUTION.md](SOLUTION.md) for details.

---

## Current Best Models

### Model 1: v92d XGBoost + Adversarial Weights (BEST LB: 0.6986)

**Script**: `non_successful_tests/scripts/train_v92_focal_adversarial.py`

**Architecture**: XGBoost with adversarial sample weighting
```python
# Uses v34a features + adversarial weights
# Adversarial weights: down-weight training samples that look different from test
# Loaded from: data/processed/adversarial_validation.pkl
```

**Key Innovation**:
- **Adversarial validation** identifies train samples that differ from test distribution
- Down-weights "outlier" training samples to focus on generalizable patterns
- Uses same v34a feature set (224 features)

**Key Metrics**:
- LB F1: **0.6986** (BEST)

### Model 2: v34a XGBoost (2nd Best LB: 0.6907)

**Script**: `scripts/train_v34a_bazin.py`

**Architecture**:
```python
xgb_params = {
    'objective': 'binary:logistic',
    'max_depth': 5,
    'learning_rate': 0.025,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'scale_pos_weight': 19.56,  # Class imbalance ratio
    'tree_method': 'hist',
}
```

**Features (224 total)**:
- 120 selected base features (from importance ranking)
- TDE physics features (decay rates, power law fits)
- Multi-band GP features (2D Gaussian Process)
- Bazin parametric fits (rise/fall timescales)

**Key Metrics**:
- OOF F1: 0.6667
- LB F1: **0.6907**
- Optimal threshold: ~0.07

---

## Feature Sets & Cached Data

All features are pre-computed and cached in `data/processed/`:

| Cache File | Features | Description |
|------------|----------|-------------|
| `features_v4_cache.pkl` | ~200 | Base statistical + shape + color features |
| `tde_physics_cache.pkl` | ~20 | TDE-specific physics (decay rates, power law) |
| `multiband_gp_cache.pkl` | ~15 | 2D Gaussian Process (time + wavelength) |
| `bazin_features_cache.pkl` | ~50 | Bazin function fits per band |
| `v34a_artifacts.pkl` | - | OOF predictions, feature importance, threshold |

### Top 10 Most Important Features (from v34a)

| Rank | Feature | Category |
|------|---------|----------|
| 1 | `gp_gr_color_50d` | Color evolution |
| 2 | `r_skew` | Light curve shape |
| 3 | `gp2d_wave_scale` | GP wavelength scale |
| 4 | `r_bazin_B` | Bazin amplitude |
| 5 | `r_bazin_tau_rise` | Rise timescale |
| 6 | `r_bazin_tau_fall` | Decline timescale |
| 7 | `gp_gr_color_20d` | Early color |
| 8 | `gp_ri_color_50d` | r-i color |
| 9 | `g_skew` | g-band asymmetry |
| 10 | `g_bazin_B` | g-band amplitude |

---

## LightGBM Development (In Progress)

### Critical Insight: OOF vs LB Paradox

**Higher OOF F1 often means WORSE LB F1** (overfitting):
- v34a: OOF 0.6667 → LB **0.6907** (best)
- v77: OOF 0.6886 → LB 0.6714 (overfit)
- v80a: OOF 0.7118 → LB 0.6666 (severe overfit)

**Strategy**: Use heavy regularization to force LightGBM to generalize.

### LightGBM Experiments

| Version | OOF F1 | LB F1 | Strategy |
|---------|--------|-------|----------|
| v77 | 0.6886 | 0.6714 | Optuna (overfits) |
| v110 | 0.6609 | TBD | Heavy regularization |
| v111 | 0.6608 | TBD | DART boosting |
| v112 | 0.6914 | TBD | Optuna constrained search |

**v112 Best Parameters** (Optuna found):
```python
lgb_params = {
    'boosting_type': 'dart',
    'num_leaves': 15,
    'max_depth': 5,
    'learning_rate': 0.029,
    'n_estimators': 655,
    'feature_fraction': 0.301,  # Very aggressive
    'bagging_fraction': 0.576,
    'reg_alpha': 2.41,
    'reg_lambda': 5.44,
    'drop_rate': 0.271,  # DART dropout
    'skip_drop': 0.540,
}
```

---

## Data Structure

**Location**: `data/raw/`

```
data/raw/
├── train_log.csv          # 3,054 objects (148 TDE, 2906 non-TDE)
├── test_log.csv           # 7,124 objects
├── sample_submission.csv
└── split_01/ to split_20/ # Lightcurve data
```

**Class Imbalance**: 148 TDE vs 2906 non-TDE (ratio: 19.56)

**LSST Bands**: u, g, r, i, z, y

---

## Key Commands

```bash
# Train best XGBoost model
python scripts/train_v34a_bazin.py

# Train LightGBM variants
python scripts/train_v110_lgbm_regularized.py
python scripts/train_v111_lgbm_dart.py
python scripts/train_v112_lgbm_optuna_reg.py

# Create ensemble (after LB scores known)
python scripts/train_v113_xgb_lgb_ensemble.py
```

---

## Key Learnings

### What Works
- **Bazin parametric fits** - Captures rise/fall physics
- **GP color features** - g-r, r-i colors at 20d, 50d post-peak
- **Heavy regularization** - Prevents overfitting to training quirks
- **Optimal threshold ~0.07** - Not 0.5 due to class imbalance
- **Feature selection** - 70-120 features optimal, more = worse

### What Doesn't Work
- **Adding more features** - Always hurts LB
- **Data augmentation** - Domain shift issues
- **Naive ensembling** - v78 ensemble worse than solo models
- **Deep learning** - LSTM 0.12 F1, ATAT 0.50 F1
- **Cesium features** - Added noise, not signal

### Physics Insight
TDEs maintain hot blackbody temperatures (~2-4×10⁴ K) due to continuous accretion heating. SNe cool as they expand. AGN fluctuate stochastically. This makes **color evolution** the key discriminator.

---

## Next Steps

1. **Submit v110, v111, v112 to Kaggle** - Get LB scores
2. **Identify best LightGBM** - Whichever has highest LB (not OOF!)
3. **Create ensemble** - Weighted average of v34a XGB + best LGB
4. **Feature engineering** - If time permits, explore new physics features

---

## File Structure

```
MALLORN astrophysics/
├── CLAUDE.md                    # This file
├── data/
│   ├── raw/                     # Kaggle data
│   └── processed/               # Feature caches
├── scripts/
│   ├── train_v34a_bazin.py      # Best XGBoost
│   ├── train_v110_lgbm_*.py     # LightGBM experiments
│   └── train_v113_*.py          # Ensemble
├── src/
│   ├── features/                # Feature extraction modules
│   ├── models/                  # Model definitions
│   └── utils/data_loader.py     # Data loading utilities
└── submissions/                 # CSV files for Kaggle
```

---

## Quick Start

```bash
cd "C:\Users\alexy\Documents\Claude_projects\Kaggle competition\MALLORN astrophysics"
```

**Priority tasks**:
1. Submit pending LightGBM models to Kaggle
2. Compare LB scores to identify best generalizing model
3. Build ensemble with XGBoost + best LightGBM
