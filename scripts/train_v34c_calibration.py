"""
MALLORN v34c: Probability Calibration (Technique #3)

Building on v34a (Bazin features, OOF F1=0.6667):
- Technique #3: Probability Calibration via Platt Scaling and Isotonic Regression

Why Calibration:
- v34a threshold is fragile (optimized per-fold)
- Model outputs poorly calibrated probabilities
- PLAsTiCC 1st & 2nd place both used calibration

Methods:
1. Platt Scaling: Sigmoid transformation (parametric)
2. Isotonic Regression: Non-parametric monotonic fit
3. Beta Calibration: For skewed distributions (if needed)

Expected gain: +1-3% (stabilizes threshold, improves precision)
Target: OOF F1 > 0.68
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v34c: Probability Calibration (Technique #3/8)", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD v34a ARTIFACTS (BASELINE)
# ====================
print("\n1. Loading v34a artifacts (Bazin features)...", flush=True)

with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

oof_preds_raw = v34a['oof_preds']  # Raw XGBoost probabilities
feature_names = v34a['feature_names']

print(f"   v34a OOF F1: {v34a['oof_f1']:.4f} @ threshold={v34a['best_threshold']:.2f}", flush=True)

# ====================
# 2. LOAD DATA FOR RECALIBRATION
# ====================
print("\n2. Loading data and features...", flush=True)

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
print("   Extracting Bazin features...", flush=True)
from features.bazin_fitting import extract_bazin_features

print("      Training set...", flush=True)
train_bazin = extract_bazin_features(train_lc, train_ids)
print("      Test set...", flush=True)
test_bazin = extract_bazin_features(test_lc, test_ids)

# Combine
train_combined = train_v21.merge(train_bazin, on='object_id', how='left')
test_combined = test_v21.merge(test_bazin, on='object_id', how='left')

X_train = train_combined.drop(columns=['object_id']).values
X_test = test_combined.drop(columns=['object_id']).values

print(f"   Features: {len(feature_names)}, Training shape: {X_train.shape}", flush=True)

# ====================
# 3. APPLY CALIBRATION METHODS
# ====================
print("\n3. Applying calibration methods...", flush=True)

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
    'scale_pos_weight': len(y[y==0]) / len(y[y==1]),
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Storage for different calibration methods
oof_raw = np.zeros(len(X_train))
oof_platt = np.zeros(len(X_train))
oof_isotonic = np.zeros(len(X_train))

test_raw = np.zeros((len(X_test), n_folds))
test_platt = np.zeros((len(X_test), n_folds))
test_isotonic = np.zeros((len(X_test), n_folds))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    print(f"\n   Fold {fold}/{n_folds}:", flush=True)

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Train base XGBoost model
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
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

    # Get raw predictions
    raw_pred_val = model.predict(dval)
    raw_pred_test = model.predict(dtest)

    oof_raw[val_idx] = raw_pred_val
    test_raw[:, fold-1] = raw_pred_test

    # Method 1: Platt Scaling (sigmoid fit)
    # Use a held-out calibration set within training fold
    cal_size = int(0.2 * len(train_idx))
    cal_idx = train_idx[-cal_size:]
    train_sub_idx = train_idx[:-cal_size]

    X_cal = X_train[cal_idx]
    y_cal = y[cal_idx]

    dcal = xgb.DMatrix(X_cal, feature_names=feature_names)
    raw_pred_cal = model.predict(dcal)

    # Fit Platt scaler (logistic regression on raw predictions)
    platt_scaler = LogisticRegression()
    platt_scaler.fit(raw_pred_cal.reshape(-1, 1), y_cal)

    platt_pred_val = platt_scaler.predict_proba(raw_pred_val.reshape(-1, 1))[:, 1]
    platt_pred_test = platt_scaler.predict_proba(raw_pred_test.reshape(-1, 1))[:, 1]

    oof_platt[val_idx] = platt_pred_val
    test_platt[:, fold-1] = platt_pred_test

    # Method 2: Isotonic Regression (non-parametric)
    iso_scaler = IsotonicRegression(out_of_bounds='clip')
    iso_scaler.fit(raw_pred_cal, y_cal)

    iso_pred_val = iso_scaler.predict(raw_pred_val)
    iso_pred_test = iso_scaler.predict(raw_pred_test)

    oof_isotonic[val_idx] = iso_pred_val
    test_isotonic[:, fold-1] = iso_pred_test

    # Evaluate fold performance
    best_f1_raw, thresh_raw = 0, 0.5
    best_f1_platt, thresh_platt = 0, 0.5
    best_f1_iso, thresh_iso = 0, 0.5

    for t in np.linspace(0.05, 0.5, 50):
        f1_raw = f1_score(y_val, (raw_pred_val > t).astype(int))
        f1_platt = f1_score(y_val, (platt_pred_val > t).astype(int))
        f1_iso = f1_score(y_val, (iso_pred_val > t).astype(int))

        if f1_raw > best_f1_raw:
            best_f1_raw, thresh_raw = f1_raw, t
        if f1_platt > best_f1_platt:
            best_f1_platt, thresh_platt = f1_platt, t
        if f1_iso > best_f1_iso:
            best_f1_iso, thresh_iso = f1_iso, t

    print(f"      Raw:      F1={best_f1_raw:.4f} @ {thresh_raw:.3f}", flush=True)
    print(f"      Platt:    F1={best_f1_platt:.4f} @ {thresh_platt:.3f}", flush=True)
    print(f"      Isotonic: F1={best_f1_iso:.4f} @ {thresh_iso:.3f}", flush=True)

# ====================
# 4. EVALUATE CALIBRATION METHODS
# ====================
print("\n" + "=" * 80, flush=True)
print("CALIBRATION RESULTS", flush=True)
print("=" * 80, flush=True)

def evaluate_method(oof_preds, method_name):
    best_f1 = 0
    best_thresh = 0.5
    for t in np.linspace(0.05, 0.5, 100):
        preds_binary = (oof_preds > t).astype(int)
        f1 = f1_score(y, preds_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    final_preds = (oof_preds > best_thresh).astype(int)
    tp = np.sum((final_preds == 1) & (y == 1))
    fp = np.sum((final_preds == 1) & (y == 0))
    fn = np.sum((final_preds == 0) & (y == 1))

    brier = brier_score_loss(y, oof_preds)

    print(f"\n{method_name}:", flush=True)
    print(f"   OOF F1: {best_f1:.4f} @ threshold={best_thresh:.2f}", flush=True)
    print(f"   Brier score: {brier:.4f} (lower is better calibration)", flush=True)
    print(f"   Confusion: TP={tp}, FP={fp}, FN={fn}", flush=True)
    print(f"   Precision: {tp/(tp+fp):.4f}, Recall: {tp/(tp+fn):.4f}", flush=True)

    return best_f1, best_thresh, brier

f1_raw, thresh_raw, brier_raw = evaluate_method(oof_raw, "Raw (Uncalibrated)")
f1_platt, thresh_platt, brier_platt = evaluate_method(oof_platt, "Platt Scaling")
f1_iso, thresh_iso, brier_iso = evaluate_method(oof_isotonic, "Isotonic Regression")

# ====================
# 5. SELECT BEST METHOD
# ====================
print("\n" + "=" * 80, flush=True)
print("COMPARISON", flush=True)
print("=" * 80, flush=True)

print(f"v34a (baseline):        OOF F1 = 0.6667", flush=True)
print(f"v34c Raw:               OOF F1 = {f1_raw:.4f} ({(f1_raw-0.6667)*100/0.6667:+.2f}%)", flush=True)
print(f"v34c Platt Scaling:     OOF F1 = {f1_platt:.4f} ({(f1_platt-0.6667)*100/0.6667:+.2f}%)", flush=True)
print(f"v34c Isotonic:          OOF F1 = {f1_iso:.4f} ({(f1_iso-0.6667)*100/0.6667:+.2f}%)", flush=True)

# Select best method
methods = {
    'raw': (f1_raw, thresh_raw, oof_raw, test_raw.mean(axis=1)),
    'platt': (f1_platt, thresh_platt, oof_platt, test_platt.mean(axis=1)),
    'isotonic': (f1_iso, thresh_iso, oof_isotonic, test_isotonic.mean(axis=1))
}

best_method = max(methods.items(), key=lambda x: x[1][0])
method_name, (best_f1, best_thresh, oof_best, test_best) = best_method

print(f"\nBest method: {method_name.upper()} (F1={best_f1:.4f})", flush=True)

# ====================
# 6. CREATE SUBMISSION
# ====================
print("\n6. Creating submission...", flush=True)

test_final = (test_best > best_thresh).astype(int)

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / f'submissions/submission_v34c_{method_name}.csv'
submission.to_csv(submission_path, index=False)

print(f"   Submission saved: {submission_path.name}", flush=True)
print(f"   Predicted TDEs: {test_final.sum()} / {len(test_final)}", flush=True)

# Save artifacts
artifacts = {
    'oof_preds': oof_best,
    'test_preds': test_best,
    'best_threshold': best_thresh,
    'oof_f1': best_f1,
    'method': method_name,
    'all_methods': {
        'raw': {'f1': f1_raw, 'thresh': thresh_raw, 'brier': brier_raw},
        'platt': {'f1': f1_platt, 'thresh': thresh_platt, 'brier': brier_platt},
        'isotonic': {'f1': f1_iso, 'thresh': thresh_iso, 'brier': brier_iso}
    }
}

with open(base_path / 'data/processed/v34c_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v34c (Calibration) Complete: OOF F1 = {best_f1:.4f}", flush=True)
print(f"v34a (Bazin): OOF F1 = 0.6667", flush=True)
change = (best_f1 - 0.6667) * 100 / 0.6667
print(f"Change vs v34a: {change:+.2f}% ({best_f1 - 0.6667:+.4f})", flush=True)

if best_f1 > 0.68:
    print("SUCCESS: Calibration improved performance!", flush=True)
elif best_f1 > 0.6667:
    print("IMPROVEMENT: Beat v34a baseline!", flush=True)
else:
    print("Calibration did not improve - continue with other techniques", flush=True)

print("=" * 80, flush=True)
