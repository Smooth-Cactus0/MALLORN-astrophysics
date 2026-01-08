"""
MALLORN: Deep Analysis of Two-Stage Classifier Errors

Goal: Understand exactly where we're making mistakes to target improvements

Analysis:
1. Stage 1 (AGN classifier):
   - Which TDEs look like AGN? (dangerous false positives)
   - Which AGN slip through? (at different thresholds)
   - Feature comparison: TDEs misclassified as AGN vs correctly classified TDEs

2. Stage 2 (TDE vs SN):
   - Confusion matrix by SpecType (SN Ia, SN II, SLSN, etc.)
   - Which SNe look like TDEs? (false positives)
   - Which TDEs look like SNe? (false negatives)
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def load_features():
    """Load cached features."""
    cache_dir = Path(__file__).parent.parent / 'data' / 'processed'

    with open(cache_dir / 'features_cache.pkl', 'rb') as f:
        base = pickle.load(f)
    train_features = base['train_features'].copy()

    with open(cache_dir / 'bazin_features_cache.pkl', 'rb') as f:
        bazin = pickle.load(f)
    train_features = train_features.merge(bazin['train'], on='object_id', how='left')

    with open(cache_dir / 'tde_physics_cache.pkl', 'rb') as f:
        tde = pickle.load(f)
    train_features = train_features.merge(tde['train'], on='object_id', how='left')

    with open(cache_dir / 'multiband_gp_cache.pkl', 'rb') as f:
        gp = pickle.load(f)
    train_features = train_features.merge(gp['train'], on='object_id', how='left')

    return train_features


def main():
    print("=" * 80)
    print("MALLORN: Deep Error Analysis for Two-Stage Classifier")
    print("=" * 80)

    data_dir = Path(__file__).parent.parent / 'data' / 'raw'

    # Load data
    train_meta = pd.read_csv(data_dir / 'train_log.csv')
    train_features = load_features()

    feature_cols = [c for c in train_features.columns if c != 'object_id']
    X = train_features[feature_cols].fillna(train_features[feature_cols].median())

    print(f"\nDataset composition:")
    print(train_meta['SpecType'].value_counts().to_string())

    # =========================================================================
    # STAGE 1: AGN Classifier Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: AGN CLASSIFIER ANALYSIS")
    print("=" * 80)

    y_agn = (train_meta['SpecType'] == 'AGN').astype(int).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    agn_probs = np.zeros(len(X))
    feature_importance = np.zeros(len(feature_cols))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_agn)):
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42+fold, n_jobs=-1, verbosity=0
        )
        model.fit(X.iloc[tr_idx], y_agn[tr_idx])
        agn_probs[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        feature_importance += model.feature_importances_ / 5

    train_meta['agn_prob'] = agn_probs

    # Confusion matrix at different thresholds
    print("\n1.1 Confusion by SpecType at AGN_prob >= 0.99:")
    high_agn_mask = agn_probs >= 0.99

    print(f"\n   Objects filtered as AGN (prob >= 0.99):")
    filtered = train_meta[high_agn_mask]
    print(f"   {filtered['SpecType'].value_counts().to_string()}")

    print(f"\n   Objects KEPT (prob < 0.99):")
    kept = train_meta[~high_agn_mask]
    print(f"   {kept['SpecType'].value_counts().to_string()}")

    # TDEs misclassified as AGN
    print("\n" + "-" * 60)
    print("1.2 TDEs MISCLASSIFIED AS AGN (Critical Errors!)")
    print("-" * 60)

    tde_mask = train_meta['target'] == 1
    tde_as_agn = train_meta[tde_mask & high_agn_mask]

    print(f"\n   TDEs with AGN_prob >= 0.99: {len(tde_as_agn)} / {tde_mask.sum()}")

    if len(tde_as_agn) > 0:
        print(f"\n   These TDEs look like AGN:")
        for _, row in tde_as_agn.iterrows():
            print(f"      {row['object_id']}: AGN_prob = {row['agn_prob']:.3f}")

        # Compare features of misclassified TDEs vs correctly classified TDEs
        tde_correct = train_meta[tde_mask & ~high_agn_mask]

        print(f"\n   Feature comparison (misclassified vs correct TDEs):")

        # Get top AGN-discriminative features
        imp_df = pd.DataFrame({'feature': feature_cols, 'importance': feature_importance})
        imp_df = imp_df.sort_values('importance', ascending=False)
        top_features = imp_df.head(15)['feature'].tolist()

        print(f"   {'Feature':<30} {'Misclassified':>15} {'Correct TDEs':>15} {'Diff':>10}")
        print("   " + "-" * 72)

        for feat in top_features:
            if feat in X.columns:
                misc_mean = X.loc[tde_as_agn.index, feat].mean()
                corr_mean = X.loc[tde_correct.index, feat].mean()
                diff = misc_mean - corr_mean
                print(f"   {feat:<30} {misc_mean:>15.3f} {corr_mean:>15.3f} {diff:>+10.3f}")

    # AGN that slip through
    print("\n" + "-" * 60)
    print("1.3 AGN THAT SLIP THROUGH (at different thresholds)")
    print("-" * 60)

    agn_mask = train_meta['SpecType'] == 'AGN'

    for thresh in [0.95, 0.97, 0.99]:
        slipped = train_meta[agn_mask & (agn_probs < thresh)]
        print(f"\n   AGN with prob < {thresh}: {len(slipped)} / {agn_mask.sum()}")

        if thresh == 0.99 and len(slipped) > 0:
            # Analyze these borderline AGN
            print(f"\n   Low-confidence AGN features (prob < 0.99):")
            high_conf_agn = train_meta[agn_mask & (agn_probs >= 0.99)]

            print(f"   {'Feature':<30} {'Low-conf AGN':>15} {'High-conf AGN':>15}")
            print("   " + "-" * 62)

            for feat in top_features[:10]:
                if feat in X.columns:
                    low_mean = X.loc[slipped.index, feat].mean()
                    high_mean = X.loc[high_conf_agn.index, feat].mean()
                    print(f"   {feat:<30} {low_mean:>15.3f} {high_mean:>15.3f}")

    # Top features for AGN classification
    print("\n" + "-" * 60)
    print("1.4 TOP FEATURES FOR AGN CLASSIFICATION")
    print("-" * 60)

    print(f"\n   {'Rank':<6} {'Feature':<40} {'Importance':>12}")
    print("   " + "-" * 60)
    for i, (_, row) in enumerate(imp_df.head(20).iterrows()):
        print(f"   {i+1:<6} {row['feature']:<40} {row['importance']:>12.4f}")

    # =========================================================================
    # STAGE 2: TDE vs Rest Analysis (on filtered set)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 2: TDE vs REST ANALYSIS (after AGN filtering)")
    print("=" * 80)

    # Filter to prob < 0.99
    keep_mask = agn_probs < 0.99
    X_filtered = X[keep_mask].reset_index(drop=True)
    meta_filtered = train_meta[keep_mask].reset_index(drop=True)
    y_tde = meta_filtered['target'].values

    print(f"\n   Filtered dataset: {len(X_filtered)} samples")
    print(f"   Composition:")
    print(f"   {meta_filtered['SpecType'].value_counts().to_string()}")

    # Train TDE classifier
    tde_probs = np.zeros(len(X_filtered))
    tde_importance = np.zeros(len(feature_cols))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_filtered, y_tde)):
        pos_weight = len(y_tde[y_tde==0]) / max(1, len(y_tde[y_tde==1]))

        model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            scale_pos_weight=pos_weight,
            random_state=42+fold, n_jobs=-1, verbosity=0
        )
        model.fit(X_filtered.iloc[tr_idx], y_tde[tr_idx])
        tde_probs[val_idx] = model.predict_proba(X_filtered.iloc[val_idx])[:, 1]
        tde_importance += model.feature_importances_ / 5

    meta_filtered['tde_prob'] = tde_probs

    # Find best threshold
    best_thresh = 0.32  # From v60

    preds = (tde_probs >= best_thresh).astype(int)

    print("\n" + "-" * 60)
    print("2.1 CONFUSION MATRIX BY SPECTYPE")
    print("-" * 60)

    print(f"\n   Predictions at threshold = {best_thresh}:")
    print(f"\n   {'SpecType':<12} {'Total':>8} {'Pred TDE':>10} {'Pred non-TDE':>14} {'Accuracy':>10}")
    print("   " + "-" * 56)

    for spec in meta_filtered['SpecType'].unique():
        spec_mask = meta_filtered['SpecType'] == spec
        total = spec_mask.sum()
        pred_tde = (preds[spec_mask] == 1).sum()
        pred_non = (preds[spec_mask] == 0).sum()

        # Accuracy depends on true class
        if spec == 'TDE':
            acc = pred_tde / total  # Should predict TDE
        else:
            acc = pred_non / total  # Should predict non-TDE

        print(f"   {spec:<12} {total:>8} {pred_tde:>10} {pred_non:>14} {acc:>10.1%}")

    # False Positives (non-TDE predicted as TDE)
    print("\n" + "-" * 60)
    print("2.2 FALSE POSITIVES (non-TDE predicted as TDE)")
    print("-" * 60)

    fp_mask = (preds == 1) & (y_tde == 0)
    fp_data = meta_filtered[fp_mask]

    print(f"\n   False positives by SpecType:")
    print(f"   {fp_data['SpecType'].value_counts().to_string()}")

    # Analyze FP features
    if len(fp_data) > 0:
        print(f"\n   Feature comparison (FP vs True TDEs):")
        true_tde = meta_filtered[meta_filtered['target'] == 1]

        tde_imp_df = pd.DataFrame({'feature': feature_cols, 'importance': tde_importance})
        tde_imp_df = tde_imp_df.sort_values('importance', ascending=False)
        top_tde_features = tde_imp_df.head(15)['feature'].tolist()

        print(f"   {'Feature':<30} {'False Pos':>12} {'True TDEs':>12} {'Diff':>10}")
        print("   " + "-" * 66)

        for feat in top_tde_features:
            if feat in X_filtered.columns:
                fp_mean = X_filtered.loc[fp_data.index, feat].mean()
                tde_mean = X_filtered.loc[true_tde.index, feat].mean()
                diff = fp_mean - tde_mean
                print(f"   {feat:<30} {fp_mean:>12.3f} {tde_mean:>12.3f} {diff:>+10.3f}")

    # False Negatives (TDE predicted as non-TDE)
    print("\n" + "-" * 60)
    print("2.3 FALSE NEGATIVES (TDE predicted as non-TDE)")
    print("-" * 60)

    fn_mask = (preds == 0) & (y_tde == 1)
    fn_data = meta_filtered[fn_mask]

    print(f"\n   Missed TDEs: {len(fn_data)} / {y_tde.sum()}")

    if len(fn_data) > 0:
        print(f"\n   Missed TDEs:")
        for _, row in fn_data.iterrows():
            print(f"      {row['object_id']}: TDE_prob = {row['tde_prob']:.3f}")

        # Compare missed TDEs to caught TDEs
        caught_tde = meta_filtered[(preds == 1) & (y_tde == 1)]

        print(f"\n   Feature comparison (Missed vs Caught TDEs):")
        print(f"   {'Feature':<30} {'Missed TDEs':>12} {'Caught TDEs':>12} {'Diff':>10}")
        print("   " + "-" * 66)

        for feat in top_tde_features:
            if feat in X_filtered.columns:
                miss_mean = X_filtered.loc[fn_data.index, feat].mean()
                catch_mean = X_filtered.loc[caught_tde.index, feat].mean()
                diff = miss_mean - catch_mean
                print(f"   {feat:<30} {miss_mean:>12.3f} {catch_mean:>12.3f} {diff:>+10.3f}")

    # Top features for TDE classification
    print("\n" + "-" * 60)
    print("2.4 TOP FEATURES FOR TDE CLASSIFICATION (Stage 2)")
    print("-" * 60)

    print(f"\n   {'Rank':<6} {'Feature':<40} {'Importance':>12}")
    print("   " + "-" * 60)
    for i, (_, row) in enumerate(tde_imp_df.head(20).iterrows()):
        print(f"   {i+1:<6} {row['feature']:<40} {row['importance']:>12.4f}")

    # =========================================================================
    # IMPROVEMENT RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("=" * 80)

    print("""
    STAGE 1 (AGN Classifier):
    -------------------------
    Goal: Reduce TDE loss while maintaining high AGN filtering

    Current issues:
    - {} TDEs get misclassified as AGN at threshold 0.99
    - These TDEs likely have AGN-like features (stochastic behavior?)

    Potential improvements:
    1. Add features that distinguish TDE from AGN specifically
       - TDEs have coherent rise/fall, AGN have random variability
       - Color evolution: TDEs maintain blue colors, AGN fluctuate
    2. Use different model architecture (ensemble of specialists?)
    3. Add TDE-specific physics features that AGN can't mimic

    STAGE 2 (TDE vs SN):
    --------------------
    Goal: Better separate TDEs from SNe after AGN removal

    Current false positives by type:
    {}

    Potential improvements:
    1. Features targeting specific SN types that mimic TDEs
    2. Temperature evolution (SNe cool, TDEs stay hot)
    3. Timescale features (Bazin tau parameters)
    4. Host galaxy features if available
    """.format(
        len(tde_as_agn),
        fp_data['SpecType'].value_counts().to_string() if len(fp_data) > 0 else "None"
    ))


if __name__ == '__main__':
    main()
