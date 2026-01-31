# MALLORN Model Analysis - Post-Submission Analysis (January 28, 2026)

## LB Results Summary

| Model | Type | OOF F1 | LB F1 | Delta | Fold Std | Composite Score |
|-------|------|--------|-------|-------|----------|-----------------|
| **v115a_baseline_adv** | XGBoost | 0.6731 | **0.6894** | +0.0163 | **0.0359** | **0.851** |
| **v115c_extended_research** | XGBoost | 0.6645 | **0.6840** | +0.0195 | 0.0421 | **0.839** |
| **v114d_minimal_research** | LightGBM | 0.6607 | **0.6797** | +0.0190 | 0.0546 | 0.567 |
| v115b_minimal_research | XGBoost | 0.6551 | 0.6682 | +0.0131 | 0.0506 | 0.222 |
| v114a_best_research | LightGBM | 0.6354 | 0.6542 | +0.0188 | 0.0488 | 0.363 |

**Reference Models (previously submitted):**
| Model | OOF F1 | LB F1 | Delta |
|-------|--------|-------|-------|
| v92d XGBoost+Adv | ~0.67 | **0.6986** | +0.03 |
| v34a XGBoost | 0.6667 | 0.6907 | +0.024 |
| v113b nuclear | 0.6328 | 0.6717 | +0.039 |
| v113f all_research | 0.6376 | 0.672 | +0.034 |

---

## Rankings Analysis

### Rank by LB F1 (Public Leaderboard):
1. v115a_baseline_adv: **0.6894**
2. v115c_extended_research: 0.6840
3. v114d_minimal_research: 0.6797
4. v115b_minimal_research: 0.6682
5. v114a_best_research: 0.6542

### Rank by Fold Stability (lower std = more reliable for private LB):
1. v115a_baseline_adv: std=**0.0359** ← Most stable!
2. v115c_extended_research: std=0.0421
3. v114a_best_research: std=0.0488
4. v115b_minimal_research: std=0.0506
5. v114d_minimal_research: std=0.0546

### Rank by Delta (LB - OOF, higher = better generalization signal):
1. v115c_extended_research: delta=**+0.0195** ← Best generalization!
2. v114d_minimal_research: delta=+0.0190
3. v114a_best_research: delta=+0.0188
4. v115a_baseline_adv: delta=+0.0163
5. v115b_minimal_research: delta=+0.0131

---

## Key Insights for Private LB

### Risk Assessment

**Low Risk (recommended for final submission):**
- **v115a_baseline_adv**: Best LB + lowest fold std = most reliable
- Fold scores: [0.633, 0.667, 0.733, 0.687, 0.719] - consistent performance

**Medium Risk (good for ensemble diversity):**
- **v115c_extended_research**: Good LB + highest delta = potential for private LB improvement
- **v114d_minimal_research**: Best LightGBM, different algorithm = ensemble diversity

**Higher Risk (avoid as solo submission):**
- v114a, v115b: Lower LB and/or lower generalization signals

### The OOF-LB Paradox Continues

All models show positive Delta (LB > OOF), confirming:
- Our CV strategy is conservative (good for avoiding overfitting)
- Models that underfit slightly on CV tend to generalize better
- v92d (best LB 0.6986) has Delta +0.03 - even higher than our new models

### What This Means for Ensembling

**Diversity is excellent:**
- XGBoost (v115a): adversarial weights, no research features
- LightGBM (v114d): minimal research features, no adversarial weights
- Different algorithms + different features = low correlation = good ensemble potential

---

## Fold-Level Analysis

### v115a_baseline_adv (XGBoost) - MOST STABLE
```
Fold F1s: [0.6333, 0.6667, 0.7333, 0.6866, 0.7188]
Range: 0.100 (0.633 to 0.733)
```

### v115c_extended_research (XGBoost) - BEST GENERALIZER
```
Fold F1s: [0.6230, 0.6667, 0.7368, 0.6579, 0.7213]
Range: 0.114 (0.623 to 0.737)
```

### v114d_minimal_research (LightGBM) - BEST LIGHTGBM
```
Fold F1s: [0.6032, 0.6984, 0.7692, 0.6575, 0.6667]
Range: 0.166 (0.603 to 0.769) - Higher variance, but highest fold max!
```

---

## Strategic Recommendations

### 1. Models to Use in Final Ensemble:

| Priority | Model | Reason |
|----------|-------|--------|
| 1 | v92d XGBoost+Adv (LB 0.6986) | Best known LB score |
| 2 | v115a_baseline_adv (LB 0.6894) | Best new model, most stable |
| 3 | v114d_minimal_research (LB 0.6797) | Best LightGBM, algorithm diversity |
| 4 | v115c_extended_research (LB 0.6840) | Highest delta, good generalizer |

### 2. Optuna Tuning Priority:

**Tune v114d_minimal_research (LightGBM):**
- Has highest fold variance (std=0.0546) - room for improvement
- LightGBM can match XGBoost with better tuning
- Focus on regularization to reduce variance

**Do NOT over-tune v115a:**
- Already very stable (std=0.0359)
- Risk of overfitting with more tuning
- Use as-is in ensemble

### 3. Ensemble Strategy:

**Recommended: Weighted Average**
- v92d: 40% (highest LB)
- v115a: 30% (stable, second-best new LB)
- v114d: 20% (LightGBM diversity)
- v115c: 10% (high delta generalizer)

**Alternative: Rank Average**
- More robust to outliers
- Good for diverse models

---

## Next Steps

1. [ ] Run Optuna on v114d (LightGBM) - aim to reduce fold std
2. [ ] Load v92d predictions for ensemble
3. [ ] Create weighted ensemble combining v92d + v115a + v114d
4. [ ] Submit ensemble, compare to v92d solo
5. [ ] Final submission: best between ensemble and v92d solo

---

## Feature Summary by Model

### v115a_baseline_adv (XGBoost) - BEST OVERALL
- **Features:** v34a baseline (221 features)
- **Removed:** all_rise_time, all_asymmetry (adversarial discriminative)
- **Special:** Adversarial sample weights applied
- **Threshold:** 0.3890

### v115c_extended_research (XGBoost) - BEST DELTA
- **Features:** v34a + 11 research features
- Minimal set + nuclear_position_score, mhps_10d, mhps_30d, g_r_color_peak_to_late, r_i_color_peak_to_late
- **Threshold:** 0.3984

### v114d_minimal_research (LightGBM) - BEST LIGHTGBM
- **Features:** v34a + 6 minimal research features:
  - nuclear_concentration, nuclear_smoothness
  - g_r_color_at_peak, r_i_color_at_peak
  - mhps_10_100_ratio, mhps_30_100_ratio
- **Threshold:** 0.3795

---

## Composite Score Methodology

**Formula:** `0.4*normalized_LB + 0.3*(1-normalized_std) + 0.3*normalized_delta`

Rationale:
- 40% weight on LB: Current performance matters most
- 30% weight on stability: Low fold variance = reliable on unseen data
- 30% weight on delta: High delta = model generalizes beyond training

This balances between:
- Public LB performance (what we can see)
- Private LB resilience (fold stability)
- Generalization potential (OOF-LB gap)
