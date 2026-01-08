"""
MALLORN v61a: Enhanced Two-Stage Classifier with Quick-Win Improvements

Building on v60a (LB 0.69), adding:
1. AGN probability as Stage 2 feature (let model learn to down-weight borderline AGN)
2. Rise/decline time ratio (TDEs more symmetric, SNe decline faster)
3. 30-day color change rate (SNe redden faster than TDEs)
4. Power law decay features (capture t^-5/3 TDE signature)

v60a baseline: OOF F1=0.6815, LB=0.69
Target: OOF F1 > 0.70
"""

import sys
import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_baseline_features():
    """Load all cached features from v34a/v60."""
    cache_dir = Path(__file__).parent.parent / 'data' / 'processed'

    # Base features
    with open(cache_dir / 'features_cache.pkl', 'rb') as f:
        base = pickle.load(f)
    train_features = base['train_features'].copy()
    test_features = base['test_features'].copy()

    # Bazin features
    with open(cache_dir / 'bazin_features_cache.pkl', 'rb') as f:
        bazin = pickle.load(f)
    train_features = train_features.merge(bazin['train'], on='object_id', how='left')
    test_features = test_features.merge(bazin['test'], on='object_id', how='left')

    # TDE physics features
    with open(cache_dir / 'tde_physics_cache.pkl', 'rb') as f:
        tde = pickle.load(f)
    train_features = train_features.merge(tde['train'], on='object_id', how='left')
    test_features = test_features.merge(tde['test'], on='object_id', how='left')

    # Multi-band GP features
    with open(cache_dir / 'multiband_gp_cache.pkl', 'rb') as f:
        gp = pickle.load(f)
    train_features = train_features.merge(gp['train'], on='object_id', how='left')
    test_features = test_features.merge(gp['test'], on='object_id', how='left')

    return train_features, test_features


def load_powerlaw_features(train_ids, test_ids):
    """Load power law decay features."""
    cache_dir = Path(__file__).parent.parent / 'data' / 'processed'

    with open(cache_dir / 'powerlaw_features.pkl', 'rb') as f:
        powerlaw_df = pickle.load(f)

    # The cache is a single DataFrame, split by object_id
    if isinstance(powerlaw_df, dict) and 'object_id' in powerlaw_df:
        # Convert dict to DataFrame
        powerlaw_df = pd.DataFrame(powerlaw_df)

    train_pl = powerlaw_df[powerlaw_df['object_id'].isin(train_ids)].copy()
    test_pl = powerlaw_df[powerlaw_df['object_id'].isin(test_ids)].copy()

    return train_pl, test_pl


def extract_rise_decline_features(train_lc, test_lc, train_ids, test_ids):
    """
    Extract rise time / decline time ratio features.

    TDEs tend to be more symmetric (ratio ~ 1)
    SNe typically decline faster than they rise (ratio < 1)
    """
    print("   Extracting rise/decline ratio features...", flush=True)

    def compute_rise_decline(lc_df, object_ids):
        results = []

        for i, obj_id in enumerate(object_ids):
            if (i + 1) % 500 == 0:
                print(f"      {i+1}/{len(object_ids)}", flush=True)

            obj_lc = lc_df[lc_df['object_id'] == obj_id]

            row = {'object_id': obj_id}

            # Compute per band
            for band in ['g', 'r', 'i']:
                band_lc = obj_lc[obj_lc['Filter'] == band].sort_values('Time (MJD)')

                if len(band_lc) < 5:
                    row[f'rise_decline_ratio_{band}'] = np.nan
                    row[f'rise_time_{band}'] = np.nan
                    row[f'decline_time_{band}'] = np.nan
                    continue

                times = band_lc['Time (MJD)'].values
                flux = band_lc['Flux'].values

                # Find peak
                peak_idx = np.argmax(flux)
                peak_time = times[peak_idx]
                peak_flux = flux[peak_idx]

                # Half-max threshold
                half_max = peak_flux / 2

                # Rise time: from first crossing half-max to peak
                pre_peak = np.where((times < peak_time) & (flux >= half_max))[0]
                if len(pre_peak) > 0:
                    rise_start = times[pre_peak[0]]
                    rise_time = peak_time - rise_start
                else:
                    rise_time = peak_time - times[0]  # Use first observation

                # Decline time: from peak to last crossing half-max
                post_peak = np.where((times > peak_time) & (flux >= half_max))[0]
                if len(post_peak) > 0:
                    decline_end = times[post_peak[-1]]
                    decline_time = decline_end - peak_time
                else:
                    decline_time = times[-1] - peak_time  # Use last observation

                # Avoid division by zero
                if decline_time > 0:
                    ratio = rise_time / decline_time
                else:
                    ratio = np.nan

                row[f'rise_decline_ratio_{band}'] = ratio
                row[f'rise_time_{band}'] = rise_time
                row[f'decline_time_{band}'] = decline_time

            # Average across bands
            ratios = [row.get(f'rise_decline_ratio_{b}', np.nan) for b in ['g', 'r', 'i']]
            row['rise_decline_ratio_mean'] = np.nanmean(ratios)

            results.append(row)

        return pd.DataFrame(results)

    train_rd = compute_rise_decline(train_lc, train_ids)
    test_rd = compute_rise_decline(test_lc, test_ids)

    return train_rd, test_rd


def extract_color_change_features(train_lc, test_lc, train_meta, test_meta, train_ids, test_ids):
    """
    Extract 30-day color change rate.

    SNe redden faster than TDEs after peak.
    TDEs maintain blue colors longer due to continuous disk heating.
    """
    print("   Extracting 30-day color change features...", flush=True)

    def compute_color_change(lc_df, meta_df, object_ids):
        results = []

        for i, obj_id in enumerate(object_ids):
            if (i + 1) % 500 == 0:
                print(f"      {i+1}/{len(object_ids)}", flush=True)

            obj_lc = lc_df[lc_df['object_id'] == obj_id]

            row = {'object_id': obj_id}

            # Get g and r band data
            g_lc = obj_lc[obj_lc['Filter'] == 'g'].sort_values('Time (MJD)')
            r_lc = obj_lc[obj_lc['Filter'] == 'r'].sort_values('Time (MJD)')

            if len(g_lc) < 3 or len(r_lc) < 3:
                row['color_change_gr_30d'] = np.nan
                row['color_at_peak_gr'] = np.nan
                row['color_at_30d_gr'] = np.nan
                results.append(row)
                continue

            # Find peak time (use r-band as reference)
            r_times = r_lc['Time (MJD)'].values
            r_flux = r_lc['Flux'].values
            peak_idx = np.argmax(r_flux)
            peak_time = r_times[peak_idx]

            # Interpolate colors at peak and peak+30d
            def get_flux_at_time(times, flux, target_time, window=5):
                """Get flux near target time."""
                mask = np.abs(times - target_time) <= window
                if mask.sum() == 0:
                    return np.nan
                weights = 1.0 / (np.abs(times[mask] - target_time) + 0.1)
                return np.average(flux[mask], weights=weights)

            g_times = g_lc['Time (MJD)'].values
            g_flux = g_lc['Flux'].values

            # At peak
            g_peak = get_flux_at_time(g_times, g_flux, peak_time)
            r_peak = get_flux_at_time(r_times, r_flux, peak_time)

            # At 30 days post-peak
            g_30d = get_flux_at_time(g_times, g_flux, peak_time + 30)
            r_30d = get_flux_at_time(r_times, r_flux, peak_time + 30)

            # Compute colors (using flux ratio as proxy for color)
            if g_peak > 0 and r_peak > 0:
                color_peak = -2.5 * np.log10(g_peak / r_peak)  # g-r color
            else:
                color_peak = np.nan

            if g_30d > 0 and r_30d > 0:
                color_30d = -2.5 * np.log10(g_30d / r_30d)
            else:
                color_30d = np.nan

            # Color change rate (mag per 30 days)
            if not np.isnan(color_peak) and not np.isnan(color_30d):
                color_change = color_30d - color_peak  # Positive = reddening
            else:
                color_change = np.nan

            row['color_change_gr_30d'] = color_change
            row['color_at_peak_gr'] = color_peak
            row['color_at_30d_gr'] = color_30d

            # Also compute r-i color change
            i_lc = obj_lc[obj_lc['Filter'] == 'i'].sort_values('Time (MJD)')
            if len(i_lc) >= 3:
                i_times = i_lc['Time (MJD)'].values
                i_flux = i_lc['Flux'].values

                i_peak = get_flux_at_time(i_times, i_flux, peak_time)
                i_30d = get_flux_at_time(i_times, i_flux, peak_time + 30)

                if r_peak > 0 and i_peak > 0:
                    ri_peak = -2.5 * np.log10(r_peak / i_peak)
                else:
                    ri_peak = np.nan

                if r_30d > 0 and i_30d > 0:
                    ri_30d = -2.5 * np.log10(r_30d / i_30d)
                else:
                    ri_30d = np.nan

                if not np.isnan(ri_peak) and not np.isnan(ri_30d):
                    row['color_change_ri_30d'] = ri_30d - ri_peak
                else:
                    row['color_change_ri_30d'] = np.nan
            else:
                row['color_change_ri_30d'] = np.nan

            results.append(row)

        return pd.DataFrame(results)

    train_cc = compute_color_change(train_lc, train_meta, train_ids)
    test_cc = compute_color_change(test_lc, test_meta, test_ids)

    return train_cc, test_cc


def main():
    print("=" * 80)
    print("MALLORN v61a: Enhanced Two-Stage with Quick-Win Improvements")
    print("=" * 80)

    base_path = Path(__file__).parent.parent
    data_dir = base_path / 'data' / 'raw'

    # ==========================================================================
    # 1. LOAD DATA
    # ==========================================================================
    print("\n1. Loading data...", flush=True)

    train_meta = pd.read_csv(data_dir / 'train_log.csv')
    test_meta = pd.read_csv(data_dir / 'test_log.csv')

    print(f"   Training: {len(train_meta)}")
    print(f"   - TDEs: {(train_meta['target'] == 1).sum()}")
    print(f"   - AGN: {(train_meta['SpecType'] == 'AGN').sum()}")

    # ==========================================================================
    # 2. LOAD BASELINE FEATURES
    # ==========================================================================
    print("\n2. Loading baseline features...", flush=True)

    train_features, test_features = load_baseline_features()
    print(f"   Baseline features: {len([c for c in train_features.columns if c != 'object_id'])}")

    # Get object IDs for splitting
    train_ids = train_meta['object_id'].tolist()
    test_ids = test_meta['object_id'].tolist()

    # ==========================================================================
    # 3. LOAD POWER LAW FEATURES
    # ==========================================================================
    print("\n3. Loading power law features...", flush=True)

    try:
        train_pl, test_pl = load_powerlaw_features(train_ids, test_ids)
        train_features = train_features.merge(train_pl, on='object_id', how='left')
        test_features = test_features.merge(test_pl, on='object_id', how='left')
        print(f"   Added {len(train_pl.columns) - 1} power law features")
    except Exception as e:
        print(f"   [WARN] Could not load power law features: {e}")

    # ==========================================================================
    # 4. EXTRACT NEW FEATURES (Rise/Decline, Color Change)
    # ==========================================================================
    print("\n4. Loading lightcurves for new features...", flush=True)

    from utils.data_loader import load_all_data
    data = load_all_data()
    train_lc = data['train_lc']
    test_lc = data['test_lc']

    # Check if we have cached features
    cache_path = base_path / 'data' / 'processed' / 'v61a_features_cache.pkl'

    if cache_path.exists():
        print("   Loading cached v61a features...", flush=True)
        with open(cache_path, 'rb') as f:
            v61a_cache = pickle.load(f)
        train_rd = v61a_cache['train_rd']
        test_rd = v61a_cache['test_rd']
        train_cc = v61a_cache['train_cc']
        test_cc = v61a_cache['test_cc']
    else:
        print("\n   Computing new features (this will be cached)...", flush=True)

        # Rise/decline ratio
        train_rd, test_rd = extract_rise_decline_features(
            train_lc, test_lc, train_ids, test_ids
        )

        # Color change rate
        train_cc, test_cc = extract_color_change_features(
            train_lc, test_lc, train_meta, test_meta, train_ids, test_ids
        )

        # Cache
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'train_rd': train_rd,
                'test_rd': test_rd,
                'train_cc': train_cc,
                'test_cc': test_cc
            }, f)
        print(f"   Cached to {cache_path}")

    # Merge new features
    train_features = train_features.merge(train_rd, on='object_id', how='left')
    test_features = test_features.merge(test_rd, on='object_id', how='left')
    train_features = train_features.merge(train_cc, on='object_id', how='left')
    test_features = test_features.merge(test_cc, on='object_id', how='left')

    # Analyze new features by class
    print("\n   New feature analysis by class:", flush=True)
    temp_df = train_features.merge(train_meta[['object_id', 'SpecType', 'target']], on='object_id')

    for feat in ['rise_decline_ratio_mean', 'color_change_gr_30d']:
        if feat in temp_df.columns:
            print(f"\n   {feat}:")
            for spec in ['TDE', 'AGN', 'SN Ia', 'SN II']:
                mask = temp_df['SpecType'] == spec
                if mask.sum() > 0:
                    val = temp_df.loc[mask, feat].mean()
                    print(f"      {spec}: {val:.3f}")

    # ==========================================================================
    # 5. PREPARE FEATURE MATRIX
    # ==========================================================================
    feature_cols = [c for c in train_features.columns if c != 'object_id']
    X_train_full = train_features[feature_cols].fillna(train_features[feature_cols].median())
    X_test_full = test_features[feature_cols].fillna(X_train_full.median())

    print(f"\n   Total features: {len(feature_cols)}")

    # ==========================================================================
    # STAGE 1: AGN CLASSIFIER
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: AGN vs Rest Classifier")
    print("=" * 80)

    y_agn = (train_meta['SpecType'] == 'AGN').astype(int).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    agn_oof_probs = np.zeros(len(X_train_full))
    agn_test_probs = np.zeros(len(X_test_full))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y_agn)):
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42+fold, n_jobs=-1, verbosity=0
        )
        model.fit(X_train_full.iloc[tr_idx], y_agn[tr_idx])
        agn_oof_probs[val_idx] = model.predict_proba(X_train_full.iloc[val_idx])[:, 1]
        agn_test_probs += model.predict_proba(X_test_full)[:, 1] / 5

    print(f"   AGN classifier accuracy: {((agn_oof_probs > 0.5) == y_agn).mean():.4f}")

    # Analyze TDE loss at different thresholds
    print("\n   TDE loss analysis:")
    tde_mask = train_meta['target'] == 1
    n_tdes = tde_mask.sum()

    for thresh in [0.95, 0.97, 0.99]:
        tdes_lost = ((agn_oof_probs >= thresh) & tde_mask).sum()
        print(f"      Threshold {thresh}: {tdes_lost} TDEs lost ({100*tdes_lost/n_tdes:.1f}%)")

    agn_threshold = 0.99  # Cautious threshold

    # ==========================================================================
    # STAGE 2: TDE vs REST (with AGN_prob as feature!)
    # ==========================================================================
    print("\n" + "=" * 80)
    print(f"STAGE 2: TDE vs Rest (AGN_prob >= {agn_threshold} filtered)")
    print("=" * 80)

    # Add AGN probability as a feature for Stage 2
    X_train_with_agn = X_train_full.copy()
    X_train_with_agn['agn_prob'] = agn_oof_probs

    X_test_with_agn = X_test_full.copy()
    X_test_with_agn['agn_prob'] = agn_test_probs

    # Filter training set
    keep_mask_train = agn_oof_probs < agn_threshold
    X_train_filtered = X_train_with_agn[keep_mask_train].reset_index(drop=True)
    y_train_filtered = train_meta.loc[keep_mask_train, 'target'].values
    meta_filtered = train_meta[keep_mask_train].reset_index(drop=True)

    print(f"\n   After filtering: {len(X_train_filtered)} samples")
    print(f"   TDEs remaining: {y_train_filtered.sum()} / {n_tdes}")
    print(f"   Features (with agn_prob): {X_train_filtered.shape[1]}")

    # Train TDE classifier
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tde_oof_probs = np.zeros(len(X_train_filtered))
    tde_test_probs = np.zeros(len(X_test_with_agn))
    feature_importance = np.zeros(X_train_filtered.shape[1])

    for fold, (tr_idx, val_idx) in enumerate(skf2.split(X_train_filtered, y_train_filtered)):
        pos_weight = len(y_train_filtered[y_train_filtered==0]) / max(1, len(y_train_filtered[y_train_filtered==1]))

        model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pos_weight,
            random_state=42+fold, n_jobs=-1, verbosity=0
        )

        model.fit(X_train_filtered.iloc[tr_idx], y_train_filtered[tr_idx],
                  eval_set=[(X_train_filtered.iloc[val_idx], y_train_filtered[val_idx])],
                  verbose=False)

        tde_oof_probs[val_idx] = model.predict_proba(X_train_filtered.iloc[val_idx])[:, 1]
        tde_test_probs += model.predict_proba(X_test_with_agn)[:, 1] / 5
        feature_importance += model.feature_importances_ / 5

        f1 = max([f1_score(y_train_filtered[val_idx],
                          (tde_oof_probs[val_idx] >= t).astype(int))
                  for t in np.arange(0.05, 0.5, 0.01)])
        print(f"   Fold {fold+1}: F1={f1:.4f}")

    # Find best threshold
    best_f1_filtered, best_thresh = 0, 0.1
    for t in np.arange(0.05, 0.5, 0.01):
        preds = (tde_oof_probs >= t).astype(int)
        f1 = f1_score(y_train_filtered, preds)
        if f1 > best_f1_filtered:
            best_f1_filtered, best_thresh = f1, t

    print(f"\n   Stage 2 OOF F1: {best_f1_filtered:.4f} @ thresh={best_thresh:.2f}")

    # ==========================================================================
    # FULL TRAINING SET EVALUATION
    # ==========================================================================
    print("\n" + "=" * 80)
    print("FULL TRAINING SET EVALUATION")
    print("=" * 80)

    # Map predictions back to full set
    tde_oof_full = np.zeros(len(X_train_full))
    keep_indices = np.where(keep_mask_train)[0]
    for i, idx in enumerate(keep_indices):
        tde_oof_full[idx] = tde_oof_probs[i]

    # Filtered-out samples get probability 0
    tde_oof_full[~keep_mask_train] = 0.0

    y_train_full = train_meta['target'].values

    # Find best threshold on full set
    best_f1_full, best_thresh_full = 0, 0.1
    for t in np.arange(0.05, 0.5, 0.01):
        preds = (tde_oof_full >= t).astype(int)
        f1 = f1_score(y_train_full, preds)
        if f1 > best_f1_full:
            best_f1_full, best_thresh_full = f1, t

    print(f"\n   Full OOF F1: {best_f1_full:.4f} @ thresh={best_thresh_full:.2f}")

    # Confusion matrix
    final_preds = (tde_oof_full >= best_thresh_full).astype(int)
    cm = confusion_matrix(y_train_full, final_preds)
    print(f"\n   Confusion Matrix:")
    print(f"   TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    recall = cm[1,1] / (cm[1,1] + cm[1,0])
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    print(f"   Recall: {recall:.1%}, Precision: {precision:.1%}")

    # ==========================================================================
    # TOP FEATURE IMPORTANCE (including new features)
    # ==========================================================================
    print("\n" + "-" * 60)
    print("TOP FEATURES (Stage 2)")
    print("-" * 60)

    feat_cols = X_train_filtered.columns.tolist()
    imp_df = pd.DataFrame({'feature': feat_cols, 'importance': feature_importance})
    imp_df = imp_df.sort_values('importance', ascending=False)

    print(f"\n   {'Rank':<6} {'Feature':<40} {'Importance':>12}")
    print("   " + "-" * 60)
    for i, (_, row) in enumerate(imp_df.head(20).iterrows()):
        marker = " *NEW*" if row['feature'] in ['agn_prob', 'rise_decline_ratio_mean',
                                                  'color_change_gr_30d', 'color_change_ri_30d'] else ""
        print(f"   {i+1:<6} {row['feature']:<40} {row['importance']:>12.4f}{marker}")

    # ==========================================================================
    # COMPARISON
    # ==========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"\n   v34a baseline:     OOF F1=0.6667, LB=0.6907")
    print(f"   v60a hard filter:  OOF F1=0.6815, LB=0.69")
    print(f"   v61a enhanced:     OOF F1={best_f1_full:.4f}")

    improvement = 100 * (best_f1_full - 0.6815) / 0.6815
    print(f"\n   Improvement over v60a: {improvement:+.2f}%")

    # ==========================================================================
    # SUBMISSIONS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUBMISSIONS")
    print("=" * 80)

    sub_dir = base_path / 'submissions'

    # Apply same filtering to test set
    keep_mask_test = agn_test_probs < agn_threshold
    test_preds = np.zeros(len(X_test_full))
    test_preds[keep_mask_test] = (tde_test_probs[keep_mask_test] >= best_thresh).astype(int)

    sub = pd.DataFrame({
        'object_id': test_meta['object_id'],
        'target': test_preds.astype(int)
    })
    sub.to_csv(sub_dir / 'submission_v61a_enhanced_two_stage.csv', index=False)
    print(f"   v61a: {sub['target'].sum()} TDEs predicted")
    print(f"   Saved: submission_v61a_enhanced_two_stage.csv")

    # Also save with alternative threshold (slightly lower for higher recall)
    test_preds_b = np.zeros(len(X_test_full))
    test_preds_b[keep_mask_test] = (tde_test_probs[keep_mask_test] >= best_thresh * 0.8).astype(int)

    sub_b = pd.DataFrame({
        'object_id': test_meta['object_id'],
        'target': test_preds_b.astype(int)
    })
    sub_b.to_csv(sub_dir / 'submission_v61a_high_recall.csv', index=False)
    print(f"   v61a-hr: {sub_b['target'].sum()} TDEs predicted (high recall)")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
