"""
MALLORN v60: Two-Stage Classifier with Cautious AGN Filtering

Strategy:
1. Train AGN vs Rest classifier
2. Analyze TDE loss at different AGN probability thresholds
3. Only filter objects with VERY HIGH AGN probability (minimize TDE loss)
4. Train TDE vs (SN + remaining AGN) classifier on filtered set
5. Combine predictions

Key insight: Better to let some AGN through than to lose TDEs!
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')


def load_features():
    """Load cached features."""
    cache_dir = Path(__file__).parent.parent / 'data' / 'processed'

    with open(cache_dir / 'features_cache.pkl', 'rb') as f:
        base = pickle.load(f)
    train_features = base['train_features'].copy()
    test_features = base['test_features'].copy()

    with open(cache_dir / 'bazin_features_cache.pkl', 'rb') as f:
        bazin = pickle.load(f)
    train_features = train_features.merge(bazin['train'], on='object_id', how='left')
    test_features = test_features.merge(bazin['test'], on='object_id', how='left')

    with open(cache_dir / 'tde_physics_cache.pkl', 'rb') as f:
        tde = pickle.load(f)
    train_features = train_features.merge(tde['train'], on='object_id', how='left')
    test_features = test_features.merge(tde['test'], on='object_id', how='left')

    with open(cache_dir / 'multiband_gp_cache.pkl', 'rb') as f:
        gp = pickle.load(f)
    train_features = train_features.merge(gp['train'], on='object_id', how='left')
    test_features = test_features.merge(gp['test'], on='object_id', how='left')

    return train_features, test_features


def main():
    print("=" * 80)
    print("MALLORN v60: Two-Stage Classifier with Cautious AGN Filtering")
    print("=" * 80)

    data_dir = Path(__file__).parent.parent / 'data' / 'raw'

    # Load metadata
    print("\n1. Loading data...")
    train_meta = pd.read_csv(data_dir / 'train_log.csv')
    test_meta = pd.read_csv(data_dir / 'test_log.csv')

    print(f"   Training: {len(train_meta)}")
    print(f"   - TDEs: {(train_meta['target'] == 1).sum()}")
    print(f"   - AGN: {(train_meta['SpecType'] == 'AGN').sum()}")
    print(f"   - SNe: {train_meta['SpecType'].str.contains('SN').sum()}")

    # Load features
    print("\n2. Loading features...")
    train_features, test_features = load_features()

    feature_cols = [c for c in train_features.columns if c != 'object_id']
    X_train_full = train_features[feature_cols].fillna(train_features[feature_cols].median())
    X_test_full = test_features[feature_cols].fillna(X_train_full.median())

    print(f"   Features: {len(feature_cols)}")

    # =========================================================================
    # STAGE 1: AGN vs Rest Classifier
    # =========================================================================
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

    agn_accuracy = ((agn_oof_probs > 0.5) == y_agn).mean()
    print(f"\n   AGN classifier OOF accuracy: {agn_accuracy:.4f}")

    # =========================================================================
    # Analyze TDE loss at different thresholds
    # =========================================================================
    print("\n3. Analyzing TDE loss at different AGN thresholds:")
    print(f"   {'Threshold':>10} {'AGN filtered':>15} {'TDEs lost':>12} {'TDE loss %':>12}")
    print("   " + "-" * 52)

    tde_mask = train_meta['target'] == 1
    n_tdes = tde_mask.sum()

    threshold_results = []

    for thresh in [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]:
        is_agn_pred = agn_oof_probs >= thresh
        agn_filtered = is_agn_pred.sum()
        tdes_lost = (is_agn_pred & tde_mask).sum()
        tde_loss_pct = 100 * tdes_lost / n_tdes

        threshold_results.append({
            'threshold': thresh,
            'agn_filtered': agn_filtered,
            'tdes_lost': tdes_lost,
            'tde_loss_pct': tde_loss_pct
        })

        marker = " <-- SAFE" if tde_loss_pct < 5 else ""
        print(f"   {thresh:>10.2f} {agn_filtered:>15} {tdes_lost:>12} {tde_loss_pct:>11.1f}%{marker}")

    # Choose threshold with <5% TDE loss
    safe_thresholds = [r for r in threshold_results if r['tde_loss_pct'] < 5]
    if safe_thresholds:
        # Choose lowest threshold that's still safe (filters most AGN)
        best_thresh_info = min(safe_thresholds, key=lambda x: x['threshold'])
        agn_threshold = best_thresh_info['threshold']
    else:
        agn_threshold = 0.99  # Ultra conservative

    print(f"\n   Selected threshold: {agn_threshold} (TDE loss: {best_thresh_info['tde_loss_pct']:.1f}%)")

    # =========================================================================
    # STAGE 2: TDE vs Rest on filtered set
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"STAGE 2: TDE vs Rest (after filtering AGN prob >= {agn_threshold})")
    print("=" * 80)

    # Filter training set
    keep_mask_train = agn_oof_probs < agn_threshold
    X_train_filtered = X_train_full[keep_mask_train].reset_index(drop=True)
    y_train_filtered = train_meta.loc[keep_mask_train, 'target'].values
    train_meta_filtered = train_meta[keep_mask_train].reset_index(drop=True)

    print(f"\n   After filtering:")
    print(f"   - Training samples: {len(X_train_filtered)} (was {len(X_train_full)})")
    print(f"   - TDEs remaining: {y_train_filtered.sum()} / {n_tdes}")
    print(f"   - AGN remaining: {(train_meta_filtered['SpecType'] == 'AGN').sum()}")

    # Train TDE classifier on filtered set
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tde_oof_probs = np.zeros(len(X_train_filtered))

    # For full training set, we need to track which indices
    tde_oof_probs_full = np.zeros(len(X_train_full))
    tde_test_probs = np.zeros(len(X_test_full))

    for fold, (tr_idx, val_idx) in enumerate(skf2.split(X_train_filtered, y_train_filtered)):
        # Higher class weight for TDEs since they're rare
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
        tde_test_probs += model.predict_proba(X_test_full)[:, 1] / 5

        # Also predict on filtered-out training samples for comparison

        f1 = max([f1_score(y_train_filtered[val_idx],
                          (tde_oof_probs[val_idx] >= t).astype(int))
                  for t in np.arange(0.05, 0.5, 0.01)])
        print(f"   Fold {fold+1}: F1={f1:.4f}")

    # Map filtered predictions back to full training set
    keep_indices = np.where(keep_mask_train)[0]
    for i, idx in enumerate(keep_indices):
        tde_oof_probs_full[idx] = tde_oof_probs[i]

    # For filtered-out samples (high AGN prob), set TDE prob to 0
    tde_oof_probs_full[~keep_mask_train] = 0.0

    # =========================================================================
    # Combine predictions and evaluate
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMBINED RESULTS")
    print("=" * 80)

    # Find best threshold on filtered OOF predictions
    best_f1_filtered, best_thresh_filtered = 0, 0.1
    for t in np.arange(0.05, 0.5, 0.01):
        preds = (tde_oof_probs >= t).astype(int)
        f1 = f1_score(y_train_filtered, preds)
        if f1 > best_f1_filtered:
            best_f1_filtered, best_thresh_filtered = f1, t

    print(f"\n   Stage 2 (filtered) OOF F1: {best_f1_filtered:.4f} @ thresh={best_thresh_filtered:.2f}")

    # Evaluate on FULL training set
    y_train_full = train_meta['target'].values

    best_f1_full, best_thresh_full = 0, 0.1
    for t in np.arange(0.05, 0.5, 0.01):
        preds = (tde_oof_probs_full >= t).astype(int)
        f1 = f1_score(y_train_full, preds)
        if f1 > best_f1_full:
            best_f1_full, best_thresh_full = f1, t

    print(f"   Full training set OOF F1: {best_f1_full:.4f} @ thresh={best_thresh_full:.2f}")

    # Confusion matrix on full set
    final_preds = (tde_oof_probs_full >= best_thresh_full).astype(int)
    cm = confusion_matrix(y_train_full, final_preds)
    print(f"\n   Confusion Matrix (full training set):")
    print(f"   TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    # Compare with baseline
    print(f"\n   Comparison:")
    print(f"   - v34a baseline: OOF F1=0.6667, LB F1=0.6907")
    print(f"   - v60 two-stage: OOF F1={best_f1_full:.4f}")

    # =========================================================================
    # Alternative: Soft combination (use AGN prob as weight)
    # =========================================================================
    print("\n" + "=" * 80)
    print("ALTERNATIVE: Soft Combination")
    print("=" * 80)

    # TDE_prob_final = TDE_prob * (1 - AGN_prob)
    # This downweights TDE probability for high-AGN objects

    soft_probs_full = np.zeros(len(X_train_full))

    # For objects not filtered, use weighted combination
    for i in range(len(X_train_full)):
        if keep_mask_train[i]:
            # Use filtered TDE probability, weighted by (1 - AGN_prob)
            soft_probs_full[i] = tde_oof_probs_full[i] * (1 - agn_oof_probs[i])
        else:
            # Filtered out as AGN, very low TDE probability
            soft_probs_full[i] = 0.0

    # Also try: just multiply TDE prob by (1 - AGN prob) for all
    soft_probs_v2 = np.zeros(len(X_train_full))

    # Train TDE classifier on ALL data (not filtered)
    tde_oof_all = np.zeros(len(X_train_full))
    tde_test_all = np.zeros(len(X_test_full))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        pos_weight = len(y_train_full[y_train_full==0]) / max(1, len(y_train_full[y_train_full==1]))

        model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            scale_pos_weight=pos_weight,
            random_state=42+fold, n_jobs=-1, verbosity=0
        )
        model.fit(X_train_full.iloc[tr_idx], y_train_full[tr_idx])
        tde_oof_all[val_idx] = model.predict_proba(X_train_full.iloc[val_idx])[:, 1]
        tde_test_all += model.predict_proba(X_test_full)[:, 1] / 5

    # Soft combination: TDE_prob * (1 - AGN_prob)
    soft_probs_v2 = tde_oof_all * (1 - agn_oof_probs)
    soft_test_probs = tde_test_all * (1 - agn_test_probs)

    best_f1_soft, best_thresh_soft = 0, 0.1
    for t in np.arange(0.01, 0.3, 0.005):
        preds = (soft_probs_v2 >= t).astype(int)
        f1 = f1_score(y_train_full, preds)
        if f1 > best_f1_soft:
            best_f1_soft, best_thresh_soft = f1, t

    print(f"\n   Soft combination OOF F1: {best_f1_soft:.4f} @ thresh={best_thresh_soft:.3f}")

    # =========================================================================
    # Create submissions
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUBMISSIONS")
    print("=" * 80)

    sub_dir = Path(__file__).parent.parent / 'submissions'

    # Version A: Hard filtering
    keep_mask_test = agn_test_probs < agn_threshold
    test_preds_hard = np.zeros(len(X_test_full))
    test_preds_hard[keep_mask_test] = (tde_test_probs[keep_mask_test] >= best_thresh_filtered)

    sub_a = pd.DataFrame({
        'object_id': test_meta['object_id'],
        'target': test_preds_hard.astype(int)
    })
    sub_a.to_csv(sub_dir / 'submission_v60a_hard_filter.csv', index=False)
    print(f"   v60a (hard filter): {sub_a['target'].sum()} TDEs")

    # Version B: Soft combination
    test_preds_soft = (soft_test_probs >= best_thresh_soft).astype(int)

    sub_b = pd.DataFrame({
        'object_id': test_meta['object_id'],
        'target': test_preds_soft
    })
    sub_b.to_csv(sub_dir / 'submission_v60b_soft_combine.csv', index=False)
    print(f"   v60b (soft combine): {sub_b['target'].sum()} TDEs")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n   v34a baseline:      OOF F1=0.6667, LB=0.6907")
    print(f"   v60a hard filter:   OOF F1={best_f1_full:.4f}")
    print(f"   v60b soft combine:  OOF F1={best_f1_soft:.4f}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
