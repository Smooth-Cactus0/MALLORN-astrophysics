# Feature Engineering Roadmap to LB 0.70
**Target**: Increase from LB 0.6907 (v34a) → LB 0.70+ through systematic feature engineering

Based on research of:
- [PLAsTiCC Kaggle Competition Winners](https://iopscience.iop.org/article/10.3847/1538-4365/accd6a) (2018-2019)
- [2025 TDE Photometric Classification Research](https://arxiv.org/abs/2509.25902)
- [LightGBM Time Series Best Practices 2025](https://sciendo.com/article/10.2478/picbe-2025-0118)

---

## Current State Analysis

| Model | Features | OOF F1 | LB F1 | Status |
|-------|----------|--------|-------|--------|
| v34a (XGBoost) | 224 Bazin | 0.6667 | **0.6907** | ✅ BEST |
| v39b (XGBoost+Adv) | 224 Bazin | 0.6688 | 0.6855 | ❌ Failed (-0.75%) |
| v43 (XGBoost+Cesium) | 304 | 0.6667 | 0.6733 | ❌ Failed (-2.5%) |
| v46 (LightGBM) | 224 Bazin | 0.6120 | 0.6000 | ❌ Failed (-13%) |

**Key Finding**: Adding generic features (Cesium, Fourier, categorical) made things worse. We need **TDE-specific** features.

---

## Research Insights

### PLAsTiCC 1st Place (Kyle Boone - "avocado")

**Winning Strategy**:
- **Single LightGBM model** (not ensemble)
- **GP-based features**: Color at maximum light, time to decline to 20% of max
- **Data augmentation**: Time shifting, random observation removal, S/N degradation by redshift
- **Achieved**: 0.468 log-loss, 0.957 AUC for SN Ia classification

**Key Quote**: "Boone extracted the color of the object at maximum light and computed the time to decline to 20% of the maximum light of the transient, which resulted in greater accuracy of classification"

### 2025 TDE Classification Research

**Most Discriminative Features**:
1. **Post-peak colors** (g-r, r-i at +20d, +50d, +100d)
2. **GP hyperparameters** (characteristic timescales, length scales)
3. **Color evolution** (minimal for TDEs vs rapid for SNe)
4. **Rise/fade time ratios** (longer for TDEs)
5. **Bluer color** (hot blackbody from accretion disk)

**Performance**: XGBoost with GP features achieved 95% precision, 72% recall for TDEs

### LightGBM Time Series Best Practices (2025)

1. **last_k_summary method**: Statistical descriptors of recent observations
2. **Lagged variables**: Flux(t-1), Flux(t-2), Flux(t-3)
3. **Rolling window statistics**: Mean, std, slope over windows [5, 10, 20, 50 days]
4. **First-order differencing**: d(Flux)/dt patterns
5. **Native categorical handling**: Encode redshift bins, observation patterns

---

## Roadmap: 6 Prioritized Techniques

### ⭐⭐⭐⭐⭐ Tier 1: Proven TDE Discriminators (Highest Priority)

#### 1. Enhanced Post-Peak Color Features (v47)
**Based on**: 2025 TDE classifier paper
**What**: Extract colors at MORE time points post-peak
- Current: g-r, r-i at 20d, 50d (4 features)
- **Add**: 0d (peak), 10d, 30d, 75d, 100d, 150d post-peak (10 points × 2 colors = 20 features)
- **Add**: u-g, i-z colors (expand beyond g-r, r-i)
- **Add**: Color dispersion (std of color over 0-150d window)

**Expected Gain**: +2-3% (colors are #1 discriminator per research)
**Implementation**: 2-3 hours

#### 2. Time-to-Decline Features (v48)
**Based on**: PLAsTiCC 1st place (Kyle Boone)
**What**: Time to decline to X% of peak flux
- Time to decline to: 80%, 60%, 40%, 20%, 10% of peak flux
- Per band (u, g, r, i, z, y) = 5 thresholds × 6 bands = 30 features
- Ratio of decline times (e.g., t_80% / t_20%)

**Expected Gain**: +1-2% (critical feature for Boone's winner)
**Implementation**: 3-4 hours

#### 3. Color Evolution Rate Features (v49)
**Based on**: 2025 TDE research ("minimal color evolution")
**What**: Slope and acceleration of color changes
- d(g-r)/dt over [0-50d], [50-100d], [100-150d] post-peak
- Second derivative d²(color)/dt² (curvature)
- Color evolution consistency (std of slopes)

**Expected Gain**: +1-2% (TDEs have flat color evolution)
**Implementation**: 2-3 hours

---

### ⭐⭐⭐⭐ Tier 2: LightGBM Time Series Optimizations

#### 4. Rolling Window Statistics (v50)
**Based on**: 2025 LightGBM time series best practices
**What**: Statistical summaries over temporal windows
- Windows: [5, 10, 20, 50, 100 days] around peak
- Statistics: mean, std, skew, slope, min, max
- Per band = 5 windows × 6 stats × 6 bands = 180 features

**Expected Gain**: +0.5-1% (captures temporal patterns)
**Implementation**: 4-5 hours

#### 5. Lagged Flux Features (v51)
**Based on**: LightGBM time series methods
**What**: Temporal dependencies in flux measurements
- Lagged fluxes: Flux(t-1d), Flux(t-2d), Flux(t-5d), Flux(t-10d)
- Flux differences: Flux(t) - Flux(t-1d)
- Per band = 4 lags × 6 bands = 24 features

**Expected Gain**: +0.5-1%
**Implementation**: 3-4 hours

---

### ⭐⭐⭐ Tier 3: Data Augmentation (Following Boone's Strategy)

#### 6. GP-based Data Augmentation (v52)
**Based on**: PLAsTiCC 1st place
**What**: Augment training data using fitted GPs
- **Time shifting**: Shift lightcurves by ±20 days
- **Random observation removal**: Drop 10-30% of observations randomly
- **S/N degradation**: Degrade low-z objects to simulate high-z
- Expand 3,043 → 12,000+ training samples

**Expected Gain**: +1-2% (Boone's key technique)
**Implementation**: 6-8 hours (complex)
**Risk**: May overfit if not done carefully

---

## Implementation Order

### Week 1: Quick Wins (Tier 1)
1. **Day 1-2**: v47 - Enhanced post-peak colors (20 features)
   - Extract colors at 10 time points post-peak
   - Test on XGBoost v34a architecture

2. **Day 3-4**: v48 - Time-to-decline features (30 features)
   - Implement decline time calculations
   - Critical feature from PLAsTiCC winner

3. **Day 5-6**: v49 - Color evolution rates (15 features)
   - Compute color slopes and curvature
   - TDEs have flat evolution vs SNe

**Milestone**: If v47+v48+v49 reach LB 0.70, STOP here and tune hyperparameters

### Week 2: Advanced Features (Tier 2)
4. **Day 1-3**: v50 - Rolling window statistics (180 features)
   - Temporal statistical summaries
   - May be redundant with existing features - test carefully

5. **Day 4-5**: v51 - Lagged flux features (24 features)
   - Temporal dependencies
   - Quick to implement

### Week 3: Data Augmentation (Tier 3)
6. **Day 1-5**: v52 - GP-based augmentation
   - Most complex, highest risk
   - Only if Tier 1+2 don't reach 0.70

---

## Success Criteria

**Target**: LB F1 ≥ 0.70 (requires +1.35% improvement)

**Go/No-Go Decision Points**:
- After v47-v49 (Tier 1): If LB < 0.695, continue to Tier 2
- After v50-v51 (Tier 2): If LB < 0.695, proceed to Tier 3 augmentation
- After v52 (Tier 3): If still < 0.70, consider ensemble strategies

**Kill Criteria** (Stop if):
- OOF F1 decreases by >2% (features hurt)
- LB F1 decreases vs previous best
- Overfitting detected (OOF increases, LB decreases)

---

## Expected Timeline

| Technique | Implementation | Testing | Total | Expected LB |
|-----------|---------------|---------|-------|-------------|
| v47 (Colors) | 2-3h | 1h | 3-4h | 0.695-0.705 |
| v48 (Decline) | 3-4h | 1h | 4-5h | 0.700-0.710 |
| v49 (Color evo) | 2-3h | 1h | 3-4h | 0.705-0.715 |
| **Tier 1 Total** | **7-10h** | **3h** | **10-13h** | **0.70-0.715** |

If Tier 1 reaches 0.70+: STOP and tune hyperparameters
If not: Continue to Tier 2 (additional 7-9 hours)

---

## Technical Implementation Notes

### For All Techniques

```python
# Common pattern:
def extract_feature_at_timepoint(times, fluxes, peak_time, offset_days):
    """Extract feature X days after peak"""
    target_time = peak_time + offset_days
    window = (times >= target_time - 5) & (times <= target_time + 5)
    if np.sum(window) < 2:
        return np.nan
    return compute_feature(times[window], fluxes[window])
```

### v47: Enhanced Colors

```python
# Current: 4 color features (g-r, r-i at 20d, 50d)
# New: 40 color features (u-g, g-r, r-i, i-z at 10 time points)

time_points = [0, 10, 20, 30, 50, 75, 100, 150]  # days post-peak
color_pairs = [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z')]

for t in time_points:
    for (band1, band2) in color_pairs:
        color = get_flux(band1, peak_time + t) - get_flux(band2, peak_time + t)
        features[f'{band1}{band2}_color_{t}d'] = color
```

### v48: Time-to-Decline

```python
decline_thresholds = [0.8, 0.6, 0.4, 0.2, 0.1]  # % of peak flux

for band in ['u', 'g', 'r', 'i', 'z', 'y']:
    peak_flux = get_peak_flux(band)
    peak_time = get_peak_time(band)

    for thresh in decline_thresholds:
        target_flux = peak_flux * thresh
        time_to_decline = find_time_when_flux_reaches(target_flux, after=peak_time)
        features[f'{band}_decline_to_{int(thresh*100)}pct'] = time_to_decline - peak_time
```

### v49: Color Evolution Rate

```python
time_windows = [(0, 50), (50, 100), (100, 150)]

for (t_start, t_end) in time_windows:
    colors = []
    times = []
    for t in range(t_start, t_end, 5):
        color = get_color('g', 'r', peak_time + t)
        if not np.isnan(color):
            colors.append(color)
            times.append(t)

    if len(colors) >= 3:
        slope = np.polyfit(times, colors, 1)[0]
        features[f'gr_slope_{t_start}_{t_end}d'] = slope
```

---

## Why This Will Work

1. **Proven techniques**: All Tier 1 features are from published winners/research
2. **TDE-specific**: Targeting actual physics differences (color evolution, decline rates)
3. **Incremental approach**: Test each feature set independently
4. **Low risk**: If feature hurts, don't use it (unlike model changes)
5. **Historical precedent**: Boone won PLAsTiCC with similar approach

---

## References

- [PLAsTiCC Results Paper](https://iopscience.iop.org/article/10.3847/1538-4365/accd6a) (2023)
- [2025 TDE Photometric Classifier](https://arxiv.org/abs/2509.25902) (October 2025)
- [LightGBM Time Series Methods](https://sciendo.com/article/10.2478/picbe-2025-0118) (2025)
- [avocado: Boone's PLAsTiCC Winner](https://arxiv.org/abs/1907.04690) (2019)
- [LightGBM Feature Engineering Guide](https://www.numberanalytics.com/blog/lightgbm-feature-engineering) (2025)

---

**Next Action**: Implement v47 (Enhanced Post-Peak Colors) - highest priority, highest expected gain (+2-3%)
