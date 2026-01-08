"""
MALLORN v62: Multi-Class Probabilities + Rank Ensemble

NEW APPROACH to close the 0.044 gap to the leader (0.735 vs our 0.6907).

Strategy:
1. Train multi-class classifier (TDE vs SN-types vs AGN) using SpecType
2. Use class probabilities as features for the final binary classifier
3. Combine with best existing features (Bazin)
4. Rank-average with our top models (v34a, v60a)

Key insight: The problem is fundamentally 3-class, not binary.
By learning the full class structure, we may capture more signal.

SpecType distribution in training:
- AGN: 1786 (58.5%)
- SN II: 531 (17.4%)
- SN Ia: 435 (14.2%)
- TDE: 64 (2.1%)
- SLSN-I: 70 (2.3%)
- SN IIn: 103 (3.4%)
- SN Ibc: 65 (2.1%)

Target: LB F1 > 0.71 (from 0.6907)
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v62: Multi-Class Probabilities + Rank Ensemble", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_lc = data['train_lc']
test_lc = data['test_lc']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y_binary = train_meta['target'].values

# Get spectral types for multi-class
spec_types = train_meta['SpecType'].values
print(f"\nSpecType distribution:", flush=True)
for st in np.unique(spec_types):
    count = np.sum(spec_types == st)
    pct = 100 * count / len(spec_types)
    print(f"   {st}: {count} ({pct:.1f}%)", flush=True)

# Create simplified class labels (7 classes -> 4 classes)
# Group SNe subtypes to reduce class imbalance
def simplify_spectype(st):
    if st == 'TDE':
        return 'TDE'
    elif st == 'AGN':
        return 'AGN'
    elif st in ['SN Ia']:
        return 'SN_Ia'  # Thermonuclear
    else:
        return 'SN_CC'  # Core-collapse (II, IIn, Ibc, SLSN)

train_meta['simple_class'] = train_meta['SpecType'].apply(simplify_spectype)
y_multiclass = train_meta['simple_class'].values

print(f"\nSimplified classes:", flush=True)
for sc in np.unique(y_multiclass):
    count = np.sum(y_multiclass == sc)
    pct = 100 * count / len(y_multiclass)
    print(f"   {sc}: {count} ({pct:.1f}%)", flush=True)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_multiclass)
n_classes = len(le.classes_)
print(f"\nEncoded classes: {list(le.classes_)}", flush=True)

# ====================
# 2. LOAD v34a BAZIN FEATURES
# ====================
print("\n2. Loading v34a Bazin features...", flush=True)

try:
    with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
        v34a = pickle.load(f)

    X_train = v34a['X_train']
    X_test = v34a['X_test']
    feature_names = v34a['feature_names']
    v34a_oof = v34a['oof_preds']
    v34a_test = v34a['test_preds']

    print(f"   v34a features: {len(feature_names)}", flush=True)
    print(f"   v34a OOF F1: {v34a['oof_f1']:.4f}", flush=True)
except:
    print("   v34a artifacts not found, loading features manually...", flush=True)

    # Load base features
    cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
    train_base = cached['train_features']
    test_base = cached['test_features']

    selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
    importance_df = selection['importance_df']
    high_corr_df = selection['high_corr_df']

    corr_to_drop = set()
    for _, row in high_corr_df.iterrows():
        if row['feature_1'] not in corr_to_drop:
            corr_to_drop.add(row['feature_2'])
    clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
    selected_120 = clean_features.head(120)['feature'].tolist()

    # GP2D features
    with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
        gp2d_data = pickle.load(f)
    train_gp2d = gp2d_data['train']
    test_gp2d = gp2d_data['test']
    gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

    # Combine
    train_combined = train_base.merge(train_gp2d, on='object_id', how='left')
    test_combined = test_base.merge(test_gp2d, on='object_id', how='left')

    feature_names = [c for c in selected_120 if c in train_combined.columns] + gp2d_cols
    feature_names = list(dict.fromkeys(feature_names))

    train_combined = train_combined.set_index('object_id').loc[train_ids].reset_index()
    test_combined = test_combined.set_index('object_id').loc[test_ids].reset_index()

    X_train = train_combined[feature_names].values.astype(np.float32)
    X_test = test_combined[feature_names].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    v34a_oof = None
    v34a_test = None

    print(f"   Features loaded: {len(feature_names)}", flush=True)

# ====================
# 3. TRAIN MULTI-CLASS MODEL
# ====================
print("\n3. Training 4-class model (TDE/AGN/SN_Ia/SN_CC)...", flush=True)

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store multi-class probabilities
oof_multiclass = np.zeros((len(X_train), n_classes))
test_multiclass = np.zeros((len(X_test), n_classes))

xgb_params_mc = {
    'objective': 'multi:softprob',
    'num_class': n_classes,
    'eval_metric': 'mlogloss',
    'max_depth': 5,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.3,
    'reg_lambda': 1.5,
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_encoded), 1):
    print(f"   Fold {fold}/{n_folds}...", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)

    model_mc = xgb.train(
        xgb_params_mc,
        dtrain,
        num_boost_round=400,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    oof_multiclass[val_idx] = model_mc.predict(dval)
    test_multiclass += model_mc.predict(dtest) / n_folds

# Get class probabilities as features
tde_idx = list(le.classes_).index('TDE')
agn_idx = list(le.classes_).index('AGN')
sn_ia_idx = list(le.classes_).index('SN_Ia')
sn_cc_idx = list(le.classes_).index('SN_CC')

print(f"\n   Multi-class OOF probabilities extracted", flush=True)
print(f"   TDE class index: {tde_idx}", flush=True)

# Check multi-class performance for TDE detection
mc_tde_probs = oof_multiclass[:, tde_idx]
best_mc_f1 = 0
best_mc_thresh = 0.5
for t in np.linspace(0.01, 0.5, 100):
    preds = (mc_tde_probs > t).astype(int)
    f1 = f1_score(y_binary, preds)
    if f1 > best_mc_f1:
        best_mc_f1 = f1
        best_mc_thresh = t

print(f"   Multi-class TDE detection: OOF F1 = {best_mc_f1:.4f} @ thresh={best_mc_thresh:.3f}", flush=True)

# ====================
# 4. CREATE ENHANCED FEATURES
# ====================
print("\n4. Creating enhanced feature set...", flush=True)

# Add multi-class probabilities as features
train_enhanced = X_train.copy()
test_enhanced = X_test.copy()

# Add class probabilities
train_mc_features = np.column_stack([
    oof_multiclass[:, tde_idx],      # P(TDE)
    oof_multiclass[:, agn_idx],      # P(AGN)
    oof_multiclass[:, sn_ia_idx],    # P(SN_Ia)
    oof_multiclass[:, sn_cc_idx],    # P(SN_CC)
    oof_multiclass[:, tde_idx] / (oof_multiclass[:, agn_idx] + 0.001),  # TDE/AGN ratio
    oof_multiclass[:, tde_idx] / (oof_multiclass[:, sn_ia_idx] + 0.001),  # TDE/SN_Ia ratio
])

test_mc_features = np.column_stack([
    test_multiclass[:, tde_idx],
    test_multiclass[:, agn_idx],
    test_multiclass[:, sn_ia_idx],
    test_multiclass[:, sn_cc_idx],
    test_multiclass[:, tde_idx] / (test_multiclass[:, agn_idx] + 0.001),
    test_multiclass[:, tde_idx] / (test_multiclass[:, sn_ia_idx] + 0.001),
])

train_enhanced = np.column_stack([train_enhanced, train_mc_features])
test_enhanced = np.column_stack([test_enhanced, test_mc_features])

mc_feature_names = ['mc_prob_tde', 'mc_prob_agn', 'mc_prob_sn_ia', 'mc_prob_sn_cc',
                    'mc_ratio_tde_agn', 'mc_ratio_tde_sn_ia']
enhanced_feature_names = feature_names + mc_feature_names

print(f"   Enhanced features: {len(enhanced_feature_names)}", flush=True)
print(f"   ({len(feature_names)} base + {len(mc_feature_names)} multi-class)", flush=True)

# ====================
# 5. TRAIN FINAL BINARY CLASSIFIER
# ====================
print("\n5. Training final binary classifier with enhanced features...", flush=True)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros((len(X_test), n_folds))
feature_importance = np.zeros(len(enhanced_feature_names))

n_neg, n_pos = np.sum(y_binary == 0), np.sum(y_binary == 1)
scale_pos_weight = n_neg / n_pos

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.025,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.2,
    'reg_lambda': 1.5,
    'scale_pos_weight': scale_pos_weight,
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

for fold, (train_idx, val_idx) in enumerate(skf.split(train_enhanced, y_binary), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = train_enhanced[train_idx], train_enhanced[val_idx]
    y_tr, y_val = y_binary[train_idx], y_binary[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=enhanced_feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=enhanced_feature_names)
    dtest = xgb.DMatrix(test_enhanced, feature_names=enhanced_feature_names)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    oof_preds[val_idx] = model.predict(dval)
    test_preds[:, fold-1] = model.predict(dtest)

    importance = model.get_score(importance_type='gain')
    for feat, gain in importance.items():
        if feat in enhanced_feature_names:
            idx = enhanced_feature_names.index(feat)
            feature_importance[idx] += gain

    # Fold performance
    best_f1 = 0
    best_thresh = 0.5
    for t in np.linspace(0.05, 0.5, 50):
        preds = (oof_preds[val_idx] > t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    print(f"      Fold F1: {best_f1:.4f} @ threshold={best_thresh:.3f}", flush=True)

# ====================
# 6. EVALUATE
# ====================
print("\n" + "=" * 80, flush=True)
print("CROSS-VALIDATION RESULTS", flush=True)
print("=" * 80, flush=True)

best_f1 = 0
best_thresh = 0.5
for t in np.linspace(0.05, 0.5, 100):
    preds = (oof_preds > t).astype(int)
    f1 = f1_score(y_binary, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n   v62 OOF F1: {best_f1:.4f} @ threshold={best_thresh:.3f}", flush=True)

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y_binary == 1))
fp = np.sum((final_preds == 1) & (y_binary == 0))
fn = np.sum((final_preds == 0) & (y_binary == 1))
tn = np.sum((final_preds == 0) & (y_binary == 0))

print(f"   Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}", flush=True)
print(f"   Recall: {tp/(tp+fn):.4f}", flush=True)

# Compare to v34a
print(f"\n   Comparison:", flush=True)
print(f"   v34a: OOF F1 = 0.6667, LB F1 = 0.6907", flush=True)
print(f"   v62:  OOF F1 = {best_f1:.4f}", flush=True)
if best_f1 > 0.6667:
    print(f"   Improvement: +{100*(best_f1 - 0.6667)/0.6667:.2f}%", flush=True)
else:
    print(f"   Difference: {100*(best_f1 - 0.6667)/0.6667:.2f}%", flush=True)

# ====================
# 7. FEATURE IMPORTANCE
# ====================
print("\n" + "=" * 80, flush=True)
print("TOP 20 FEATURES", flush=True)
print("=" * 80, flush=True)

feat_imp_df = pd.DataFrame({
    'feature': enhanced_feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

for i, row in feat_imp_df.head(20).iterrows():
    print(f"   {row['feature']}: {row['importance']:.2f}", flush=True)

# Check if multi-class features are in top 20
mc_in_top20 = feat_imp_df.head(20)[feat_imp_df.head(20)['feature'].str.startswith('mc_')]
if len(mc_in_top20) > 0:
    print(f"\n   Multi-class features in top 20: {len(mc_in_top20)}", flush=True)
    for _, row in mc_in_top20.iterrows():
        print(f"      {row['feature']}: rank {feat_imp_df[feat_imp_df['feature'] == row['feature']].index[0] + 1}", flush=True)

# ====================
# 8. RANK ENSEMBLE WITH v34a
# ====================
print("\n" + "=" * 80, flush=True)
print("RANK ENSEMBLE", flush=True)
print("=" * 80, flush=True)

if v34a_oof is not None:
    # Rank averaging
    from scipy.stats import rankdata

    rank_v62 = rankdata(oof_preds) / len(oof_preds)
    rank_v34a = rankdata(v34a_oof) / len(v34a_oof)

    # Try different weights
    best_ensemble_f1 = 0
    best_weight = 0.5
    best_ensemble_thresh = 0.5

    for w in np.linspace(0.0, 1.0, 21):
        rank_avg = w * rank_v62 + (1 - w) * rank_v34a

        for t in np.linspace(0.3, 0.7, 50):
            preds = (rank_avg > t).astype(int)
            f1 = f1_score(y_binary, preds)
            if f1 > best_ensemble_f1:
                best_ensemble_f1 = f1
                best_weight = w
                best_ensemble_thresh = t

    print(f"\n   Rank ensemble (v62 weight={best_weight:.2f}, v34a weight={1-best_weight:.2f}):", flush=True)
    print(f"   OOF F1: {best_ensemble_f1:.4f} @ threshold={best_ensemble_thresh:.3f}", flush=True)

    if best_ensemble_f1 > max(best_f1, 0.6667):
        print(f"   IMPROVEMENT over best single model!", flush=True)
else:
    print("   v34a OOF predictions not available for ensemble", flush=True)

# ====================
# 9. GENERATE SUBMISSION
# ====================
print("\n" + "=" * 80, flush=True)
print("GENERATING SUBMISSION", flush=True)
print("=" * 80, flush=True)

test_avg = test_preds.mean(axis=1)
test_binary = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_binary
})

submission_path = base_path / 'submissions' / 'submission_v62_multiclass.csv'
submission.to_csv(submission_path, index=False)

print(f"   Submission saved: {submission_path}", flush=True)
print(f"   Predicted TDEs: {test_binary.sum()}", flush=True)

# Save artifacts
artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'oof_f1': best_f1,
    'threshold': best_thresh,
    'feature_names': enhanced_feature_names,
    'oof_multiclass': oof_multiclass,
    'test_multiclass': test_multiclass,
    'label_encoder': le
}

with open(base_path / 'data/processed/v62_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v62 Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"v34a baseline: OOF F1 = 0.6667, LB F1 = 0.6907", flush=True)
print(f"Leader: LB F1 = 0.735 (gap: {0.735 - 0.6907:.4f})", flush=True)
print("=" * 80, flush=True)
