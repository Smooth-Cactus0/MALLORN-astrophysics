"""
MALLORN v34b_conservative: Bazin + Conservative SMOTE (Techniques #1 + #2, refined)

Combining:
- Technique #1: Bazin parametric features (52 features, 21.5% importance in v34b)
- Technique #2: Conservative SMOTE synthetic minority oversampling

SMOTE Strategy (Conservative):
- Current: 148 TDEs vs 2895 non-TDEs (4.9% positive class)
- Target: 10% minority class (less aggressive than v34b's 25%)
- v34b used 25% → 579 TDEs per fold (80% synthetic) - too aggressive
- v34b_conservative: 10% → ~257 TDEs per fold (55% synthetic) - more balanced
- Apply in feature space with median imputation
- Use per-fold to avoid data leakage

Hypothesis: v34b's high variance (F1: 0.60-0.78) due to over-smoothing
Expected: More stable fold performance, OOF F1 > 0.67
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

# Install imbalanced-learn if needed
try:
    from imblearn.over_sampling import SMOTE, ADASYN
except ImportError:
    print("Installing imbalanced-learn...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])
    from imblearn.over_sampling import SMOTE, ADASYN

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v34b_conservative: Bazin + Conservative SMOTE", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD v21 FEATURES + BAZIN
# ====================
print("\n1. Loading features (v21 + Bazin)...", flush=True)

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_lc = data['train_lc']
test_lc = data['test_lc']

train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y = train_meta['target'].values

# Load v21 features
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

tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
train_tde = tde_cached['train']
test_tde = tde_cached['test']
tde_cols = [c for c in train_tde.columns if c != 'object_id']

train_base = train_base.merge(train_tde, on='object_id', how='left')
test_base = test_base.merge(test_tde, on='object_id', how='left')

with open(base_path / 'data/processed/multiband_gp_cache.pkl', 'rb') as f:
    gp2d_data = pickle.load(f)
train_gp2d = gp2d_data['train']
test_gp2d = gp2d_data['test']
gp2d_cols = [c for c in train_gp2d.columns if c != 'object_id']

train_v21 = train_base[['object_id'] + selected_120].copy()
train_v21 = train_v21.merge(train_tde, on='object_id', how='left')
train_v21 = train_v21.merge(train_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

test_v21 = test_base[['object_id'] + selected_120].copy()
test_v21 = test_v21.merge(test_tde, on='object_id', how='left')
test_v21 = test_v21.merge(test_gp2d[['object_id'] + gp2d_cols], on='object_id', how='left')

# Extract Bazin features
print("\n2. Extracting Bazin features...", flush=True)
from features.bazin_fitting import extract_bazin_features

print("   Training set...", flush=True)
train_bazin = extract_bazin_features(train_lc, train_ids)
print("   Test set...", flush=True)
test_bazin = extract_bazin_features(test_lc, test_ids)

# Combine
train_combined = train_v21.merge(train_bazin, on='object_id', how='left')
test_combined = test_v21.merge(test_bazin, on='object_id', how='left')

X_train = train_combined.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values
feature_names = [c for c in train_combined.columns if c != 'object_id']

print(f"\n   Total features: {len(feature_names)}", flush=True)
print(f"   Training shape: {X_train.shape}", flush=True)

# ====================
# 3. SMOTE CONFIGURATION
# ====================
print("\n3. Configuring SMOTE oversampling...", flush=True)

# Current class distribution
n_tdes = np.sum(y == 1)
n_non_tdes = np.sum(y == 0)
print(f"   Original: {n_tdes} TDEs, {n_non_tdes} non-TDEs ({100*n_tdes/(n_tdes+n_non_tdes):.1f}% positive)", flush=True)

# SMOTE strategy: conservative oversampling to 10% minority class
# From 4.9% → 10% means ~2× increase in TDEs (less aggressive than v34b)
sampling_strategy = 0.10  # Target 10% TDEs (conservative to avoid over-smoothing)
print(f"   Target sampling: 10% TDEs (conservative approach)", flush=True)

# Use SMOTE with careful k_neighbors selection
# k=5 is default, but we only have 64 TDEs, so use k=3 to be safe
smote = SMOTE(
    sampling_strategy=sampling_strategy,
    k_neighbors=3,  # Use 3 nearest neighbors (conservative for 64 samples)
    random_state=42
)

print(f"   SMOTE config: k_neighbors=3, sampling_strategy={sampling_strategy}", flush=True)

# ====================
# 4. TRAIN WITH SMOTE (PER-FOLD)
# ====================
print("\n4. Training XGBoost + SMOTE with 5-fold CV...", flush=True)
print("   (Applying SMOTE per-fold to avoid data leakage)", flush=True)

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
    'scale_pos_weight': 1.0,  # SMOTE handles class balance, set to 1
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros((len(X_test), n_folds))
feature_importance = np.zeros(len(feature_names))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Impute NaNs before SMOTE (SMOTE can't handle NaNs)
    # Fit imputer on training fold only to avoid data leakage
    imputer = SimpleImputer(strategy='median')
    X_tr_imputed = imputer.fit_transform(X_tr)

    # Apply SMOTE to imputed training fold
    print(f"      Before SMOTE: {np.sum(y_tr==1)} TDEs, {np.sum(y_tr==0)} non-TDEs", flush=True)
    X_tr_smote, y_tr_smote = smote.fit_resample(X_tr_imputed, y_tr)
    print(f"      After SMOTE:  {np.sum(y_tr_smote==1)} TDEs, {np.sum(y_tr_smote==0)} non-TDEs", flush=True)

    # Train on SMOTE-augmented data
    # Note: XGBoost can handle NaNs in validation, so we keep X_val as-is
    dtrain = xgb.DMatrix(X_tr_smote, label=y_tr_smote, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)

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
        if feat in feature_names:
            idx = feature_names.index(feat)
            feature_importance[idx] += gain

    best_f1 = 0
    best_thresh = 0.5
    for t in np.linspace(0.05, 0.5, 50):
        preds_binary = (oof_preds[val_idx] > t).astype(int)
        f1 = f1_score(y_val, preds_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"      Best threshold: {best_thresh:.3f}, F1: {best_f1:.4f}", flush=True)

print("\n" + "=" * 80, flush=True)
print("CROSS-VALIDATION RESULTS", flush=True)
print("=" * 80, flush=True)

best_f1 = 0
best_thresh = 0.5
for t in np.linspace(0.05, 0.5, 100):
    preds_binary = (oof_preds > t).astype(int)
    f1 = f1_score(y, preds_binary)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}", flush=True)

final_preds = (oof_preds > best_thresh).astype(int)
tp = np.sum((final_preds == 1) & (y == 1))
fp = np.sum((final_preds == 1) & (y == 0))
fn = np.sum((final_preds == 0) & (y == 1))
tn = np.sum((final_preds == 0) & (y == 0))

print(f"   Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}", flush=True)
print(f"   Recall: {tp/(tp+fn):.4f}", flush=True)

# ====================
# 5. FEATURE IMPORTANCE
# ====================
print("\n5. Top 30 Features by Importance:", flush=True)

feature_importance = feature_importance / n_folds
importance_df_result = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df_result.head(30).to_string(index=False), flush=True)

# Highlight Bazin features
bazin_cols = [c for c in feature_names if 'bazin' in c]
bazin_importance = importance_df_result[importance_df_result['feature'].isin(bazin_cols)]
if len(bazin_importance) > 0:
    print(f"\n   Bazin features account for {100*bazin_importance['importance'].sum()/importance_df_result['importance'].sum():.1f}% of model importance", flush=True)

# ====================
# 6. CREATE SUBMISSION
# ====================
print("\n6. Creating submission...", flush=True)

test_avg = test_preds.mean(axis=1)
test_final = (test_avg > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v34b_conservative.csv'
submission.to_csv(submission_path, index=False)

print(f"   Submission saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_final.sum()} / {len(test_final)}", flush=True)

# Save artifacts
artifacts = {
    'oof_preds': oof_preds,
    'test_preds': test_avg,
    'feature_importance': importance_df_result,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'feature_names': feature_names
}

with open(base_path / 'data/processed/v34b_conservative_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v34b_conservative (Bazin + Conservative SMOTE) Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"Baseline v21: OOF F1 = 0.6708", flush=True)
print(f"v34a (Bazin only): OOF F1 = 0.6667", flush=True)
print(f"v34b (aggressive SMOTE 25%): OOF F1 = 0.6585", flush=True)
change_vs_v21 = (best_f1 - 0.6708) * 100 / 0.6708
change_vs_v34a = (best_f1 - 0.6667) * 100 / 0.6667
change_vs_v34b = (best_f1 - 0.6585) * 100 / 0.6585
print(f"Change vs v21: {change_vs_v21:+.2f}% ({best_f1 - 0.6708:+.4f})", flush=True)
print(f"Change vs v34a: {change_vs_v34a:+.2f}% ({best_f1 - 0.6667:+.4f})", flush=True)
print(f"Change vs v34b: {change_vs_v34b:+.2f}% ({best_f1 - 0.6585:+.4f})", flush=True)

if best_f1 > 0.70:
    print("SUCCESS: Achieved Week 1 target (OOF > 0.70)!", flush=True)
elif best_f1 > 0.6708:
    print("IMPROVEMENT: Beat v21 baseline!", flush=True)
else:
    print("Need to analyze and adjust approach", flush=True)

print("=" * 80, flush=True)
