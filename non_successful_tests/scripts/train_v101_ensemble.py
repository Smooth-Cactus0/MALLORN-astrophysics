"""
MALLORN v101: Diverse Algorithm Ensemble

Combines three different gradient boosting algorithms:
1. XGBoost (v92d) - OOF 0.6688, LB 0.6986 (BEST)
2. CatBoost (v99e) - OOF 0.6553, recall 77.7%
3. LightGBM (v100a) - OOF 0.6608, recall 76.4%

Key insight from competition:
- Lower OOF F1 often correlates with BETTER LB
- v34a: OOF 0.6667 → LB 0.6907 (best)
- v92d: OOF 0.6688 → LB 0.6986 (new best)

Strategy: Ensemble diverse algorithms hoping their different error patterns
complement each other.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix
from scipy.stats import rankdata
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
base_path = Path(__file__).parent.parent

print("=" * 80)
print("MALLORN v101: Diverse Algorithm Ensemble")
print("=" * 80)

# ====================
# 1. LOAD DATA
# ====================
print("\n1. Loading data...")

from utils.data_loader import load_all_data
data = load_all_data()

train_meta = data['train_meta']
test_meta = data['test_meta']
train_ids = train_meta['object_id'].tolist()
test_ids = test_meta['object_id'].tolist()
y_train = train_meta['target'].values

print(f"   Training: {len(train_ids)} ({np.sum(y_train)} TDE)")

# ====================
# 2. LOAD MODEL PREDICTIONS
# ====================
print("\n2. Loading model predictions...")

# v92d XGBoost (our best LB model)
with open(base_path / 'data/processed/v92_focal_adv_artifacts.pkl', 'rb') as f:
    v92_results = pickle.load(f)
xgb_oof = v92_results['v92d_baseline_adv']['oof_preds']
xgb_test = v92_results['v92d_baseline_adv']['test_preds']
xgb_thresh = v92_results['v92d_baseline_adv']['threshold']
print(f"   v92d XGBoost: OOF F1={v92_results['v92d_baseline_adv']['oof_f1']:.4f}")

# v99e CatBoost
with open(base_path / 'data/processed/v99_catboost_artifacts.pkl', 'rb') as f:
    v99_results = pickle.load(f)
cat_oof = v99_results['v99e_optuna']['oof_preds']
cat_test = v99_results['v99e_optuna']['test_preds']
cat_thresh = v99_results['v99e_optuna']['threshold']
print(f"   v99e CatBoost: OOF F1={v99_results['v99e_optuna']['oof_f1']:.4f}")

# v100a LightGBM
with open(base_path / 'data/processed/v100_lightgbm_artifacts.pkl', 'rb') as f:
    v100_results = pickle.load(f)
lgb_oof = v100_results['v100a_optuna']['oof_preds']
lgb_test = v100_results['v100a_optuna']['test_preds']
lgb_thresh = v100_results['v100a_optuna']['threshold']
print(f"   v100a LightGBM: OOF F1={v100_results['v100a_optuna']['oof_f1']:.4f}")

# ====================
# 3. ENSEMBLE STRATEGIES
# ====================
print("\n3. Testing ensemble strategies...")

results = {}

# Helper function for OOF F1
def evaluate_oof(preds, y, name):
    best_f1 = 0
    best_thresh = 0.3
    for t in np.linspace(0.05, 0.5, 200):
        f1 = f1_score(y, (preds > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    preds_binary = (preds > best_thresh).astype(int)
    cm = confusion_matrix(y, preds_binary)
    tn, fp, fn, tp = cm.ravel()

    return {
        'oof_f1': best_f1,
        'threshold': best_thresh,
        'oof_preds': preds,
        'confusion': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'recall': tp / (tp + fn),
        'precision': tp / (tp + fp),
    }

# Strategy 1: Simple Average
print("\n   v101a_avg: Simple Average...")
avg_oof = (xgb_oof + cat_oof + lgb_oof) / 3
avg_test = (xgb_test + cat_test + lgb_test) / 3
res = evaluate_oof(avg_oof, y_train, 'avg')
res['test_preds'] = avg_test
results['v101a_avg'] = res
print(f"      OOF F1: {res['oof_f1']:.4f}, Recall: {res['recall']:.1%}, Prec: {res['precision']:.1%}")

# Strategy 2: Weighted by OOF F1
print("\n   v101b_weighted: Weighted by OOF F1...")
xgb_f1 = v92_results['v92d_baseline_adv']['oof_f1']
cat_f1 = v99_results['v99e_optuna']['oof_f1']
lgb_f1 = v100_results['v100a_optuna']['oof_f1']
total_f1 = xgb_f1 + cat_f1 + lgb_f1
w_xgb, w_cat, w_lgb = xgb_f1/total_f1, cat_f1/total_f1, lgb_f1/total_f1
print(f"      Weights: XGB={w_xgb:.3f}, Cat={w_cat:.3f}, LGB={w_lgb:.3f}")

weighted_oof = w_xgb * xgb_oof + w_cat * cat_oof + w_lgb * lgb_oof
weighted_test = w_xgb * xgb_test + w_cat * cat_test + w_lgb * lgb_test
res = evaluate_oof(weighted_oof, y_train, 'weighted')
res['test_preds'] = weighted_test
results['v101b_weighted'] = res
print(f"      OOF F1: {res['oof_f1']:.4f}, Recall: {res['recall']:.1%}, Prec: {res['precision']:.1%}")

# Strategy 3: Rank Average
print("\n   v101c_rank: Rank Averaging...")
rank_xgb = rankdata(xgb_oof) / len(xgb_oof)
rank_cat = rankdata(cat_oof) / len(cat_oof)
rank_lgb = rankdata(lgb_oof) / len(lgb_oof)
rank_oof = (rank_xgb + rank_cat + rank_lgb) / 3

rank_xgb_test = rankdata(xgb_test) / len(xgb_test)
rank_cat_test = rankdata(cat_test) / len(cat_test)
rank_lgb_test = rankdata(lgb_test) / len(lgb_test)
rank_test = (rank_xgb_test + rank_cat_test + rank_lgb_test) / 3

res = evaluate_oof(rank_oof, y_train, 'rank')
res['test_preds'] = rank_test
results['v101c_rank'] = res
print(f"      OOF F1: {res['oof_f1']:.4f}, Recall: {res['recall']:.1%}, Prec: {res['precision']:.1%}")

# Strategy 4: XGBoost-heavy (since it has best LB)
print("\n   v101d_xgb_heavy: XGBoost-heavy (0.6/0.2/0.2)...")
heavy_oof = 0.6 * xgb_oof + 0.2 * cat_oof + 0.2 * lgb_oof
heavy_test = 0.6 * xgb_test + 0.2 * cat_test + 0.2 * lgb_test
res = evaluate_oof(heavy_oof, y_train, 'xgb_heavy')
res['test_preds'] = heavy_test
results['v101d_xgb_heavy'] = res
print(f"      OOF F1: {res['oof_f1']:.4f}, Recall: {res['recall']:.1%}, Prec: {res['precision']:.1%}")

# Strategy 5: XGBoost + CatBoost only (higher recall blend)
print("\n   v101e_xgb_cat: XGBoost + CatBoost (0.5/0.5)...")
xc_oof = 0.5 * xgb_oof + 0.5 * cat_oof
xc_test = 0.5 * xgb_test + 0.5 * cat_test
res = evaluate_oof(xc_oof, y_train, 'xgb_cat')
res['test_preds'] = xc_test
results['v101e_xgb_cat'] = res
print(f"      OOF F1: {res['oof_f1']:.4f}, Recall: {res['recall']:.1%}, Prec: {res['precision']:.1%}")

# Strategy 6: Geometric Mean
print("\n   v101f_geom: Geometric Mean...")
geom_oof = np.power(xgb_oof * cat_oof * lgb_oof, 1/3)
geom_test = np.power(xgb_test * cat_test * lgb_test, 1/3)
res = evaluate_oof(geom_oof, y_train, 'geom')
res['test_preds'] = geom_test
results['v101f_geom'] = res
print(f"      OOF F1: {res['oof_f1']:.4f}, Recall: {res['recall']:.1%}, Prec: {res['precision']:.1%}")

# Strategy 7: Max voting (conservative - all must agree)
print("\n   v101g_max: Max Probability...")
max_oof = np.maximum(np.maximum(xgb_oof, cat_oof), lgb_oof)
max_test = np.maximum(np.maximum(xgb_test, cat_test), lgb_test)
res = evaluate_oof(max_oof, y_train, 'max')
res['test_preds'] = max_test
results['v101g_max'] = res
print(f"      OOF F1: {res['oof_f1']:.4f}, Recall: {res['recall']:.1%}, Prec: {res['precision']:.1%}")

# Strategy 8: Min probability (conservative)
print("\n   v101h_min: Min Probability (conservative)...")
min_oof = np.minimum(np.minimum(xgb_oof, cat_oof), lgb_oof)
min_test = np.minimum(np.minimum(xgb_test, cat_test), lgb_test)
res = evaluate_oof(min_oof, y_train, 'min')
res['test_preds'] = min_test
results['v101h_min'] = res
print(f"      OOF F1: {res['oof_f1']:.4f}, Recall: {res['recall']:.1%}, Prec: {res['precision']:.1%}")

# ====================
# 4. RESULTS
# ====================
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n{'Variant':<20} {'OOF F1':<10} {'Recall':<10} {'Prec':<10} {'FN':<6} {'FP':<6}")
print("-" * 65)
print(f"{'v92d (LB=0.6986)':<20} {'0.6688':<10} {'69.6%':<10} {'64.4%':<10} {'45':<6} {'57':<6}")
print("-" * 65)

sorted_results = sorted(results.items(), key=lambda x: -x[1]['oof_f1'])

for name, res in sorted_results:
    recall_str = f"{100*res['recall']:.1f}%"
    prec_str = f"{100*res['precision']:.1f}%"
    print(f"{name:<20} {res['oof_f1']:<10.4f} {recall_str:<10} {prec_str:<10} {res['confusion']['fn']:<6} {res['confusion']['fp']:<6}")

# ====================
# 5. SUBMISSIONS
# ====================
print("\n" + "=" * 80)
print("SUBMISSIONS")
print("=" * 80)

for name, res in sorted_results:
    test_binary = (res['test_preds'] > res['threshold']).astype(int)

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': test_binary
    })

    filename = f"submission_{name}.csv"
    submission.to_csv(base_path / f'submissions/{filename}', index=False)

    print(f"   {filename}: OOF={res['oof_f1']:.4f}, TDEs={test_binary.sum()}")

with open(base_path / 'data/processed/v101_ensemble_artifacts.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("v101 Ensemble Complete")
print("=" * 80)
print("\nRecommendation: Submit v101d_xgb_heavy or v101h_min first")
print("(Lower OOF F1 has historically led to better LB scores)")
