"""
MALLORN v38a: Rank Averaging Ensemble (v34a Bazin + v37a TDE Physics)

Combining two complementary parametric models using rank averaging:
- v34a (Bazin): SN-optimized exponential model (LB F1=0.6907)
- v37a (TDE Physics): GR-based power law model (LB F1=0.6809)

Rank Averaging Strategy:
1. Convert probabilities to ranks (highest prob = rank 1)
2. Average ranks: rank_ensemble = (rank_v34a + rank_v37a) / 2
3. Select top N objects by averaged rank

Why rank averaging works:
- Robust to probability calibration differences
- Used by PLAsTiCC 1st place winner
- Treats each model's ordering as equally valuable
- No hyperparameters to tune

Expected gain: +0.5-2% over v34a (0.6907)
Target: LB F1 > 0.70
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import f1_score
import warnings

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v38a: Rank Averaging Ensemble (v34a + v37a)", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD ARTIFACTS
# ====================
print("\n1. Loading model artifacts...", flush=True)

# Load v34a (Bazin)
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

print(f"   v34a (Bazin): OOF F1={v34a['oof_f1']:.4f}, LB F1=0.6907", flush=True)
print(f"   - {len(v34a['feature_names'])} features", flush=True)
print(f"   - Threshold: {v34a['best_threshold']:.2f}", flush=True)

# Load v37a (TDE Physics)
with open(base_path / 'data/processed/v37a_artifacts.pkl', 'rb') as f:
    v37a = pickle.load(f)

print(f"   v37a (TDE Physics): OOF F1={v37a['oof_f1']:.4f}, LB F1=0.6809", flush=True)
print(f"   - {len(v37a['feature_names'])} features", flush=True)
print(f"   - Threshold: {v37a['best_threshold']:.2f}", flush=True)

# ====================
# 2. RANK AVERAGING ON OOF
# ====================
print("\n2. Applying rank averaging to OOF predictions...", flush=True)

# Get OOF predictions
oof_v34a = v34a['oof_preds']
oof_v37a = v37a['oof_preds']

print(f"   OOF predictions shape: {oof_v34a.shape}", flush=True)

# Convert to ranks (higher probability = lower rank number, so negate)
# rankdata gives rank 1 to smallest value, so we use method='average' and negate
rank_v34a_oof = rankdata(-oof_v34a, method='average')
rank_v37a_oof = rankdata(-oof_v37a, method='average')

print(f"   v34a rank range: [{rank_v34a_oof.min():.0f}, {rank_v34a_oof.max():.0f}]", flush=True)
print(f"   v37a rank range: [{rank_v37a_oof.min():.0f}, {rank_v37a_oof.max():.0f}]", flush=True)

# Average ranks
rank_avg_oof = (rank_v34a_oof + rank_v37a_oof) / 2.0

print(f"   Averaged rank range: [{rank_avg_oof.min():.0f}, {rank_avg_oof.max():.0f}]", flush=True)

# ====================
# 3. OPTIMIZE THRESHOLD ON OOF
# ====================
print("\n3. Optimizing threshold on averaged ranks...", flush=True)

# Load true labels
sys.path.insert(0, str(base_path / 'src'))
from utils.data_loader import load_all_data
data = load_all_data()
y = data['train_meta']['target'].values

print(f"   True labels: {np.sum(y==1)} TDEs, {np.sum(y==0)} non-TDEs", flush=True)

# Find optimal number of top-ranked samples to predict as TDE
best_f1_oof = 0
best_n_tde = 0
best_rank_thresh = 0

# Try different cutoffs
n_samples = len(rank_avg_oof)
for n_tde in range(50, 500, 5):
    # Predict top n_tde ranked samples as TDE
    rank_threshold = np.partition(rank_avg_oof, n_tde)[n_tde]
    preds_binary = (rank_avg_oof <= rank_threshold).astype(int)

    f1 = f1_score(y, preds_binary)

    if f1 > best_f1_oof:
        best_f1_oof = f1
        best_n_tde = n_tde
        best_rank_thresh = rank_threshold

print(f"   Best OOF F1: {best_f1_oof:.4f}", flush=True)
print(f"   Optimal strategy: Predict top {best_n_tde} ranked as TDEs", flush=True)
print(f"   (Rank threshold: {best_rank_thresh:.1f})", flush=True)

# Get final OOF predictions
oof_ensemble = (rank_avg_oof <= best_rank_thresh).astype(int)

tp = np.sum((oof_ensemble == 1) & (y == 1))
fp = np.sum((oof_ensemble == 1) & (y == 0))
fn = np.sum((oof_ensemble == 0) & (y == 1))
tn = np.sum((oof_ensemble == 0) & (y == 0))

print(f"   Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}", flush=True)
print(f"   Precision: {tp/(tp+fp):.4f}", flush=True)
print(f"   Recall: {tp/(tp+fn):.4f}", flush=True)

# ====================
# 4. COMPARE TO INDIVIDUAL MODELS
# ====================
print("\n4. Comparison with individual models:", flush=True)

print(f"   v34a OOF F1:      {v34a['oof_f1']:.4f}", flush=True)
print(f"   v37a OOF F1:      {v37a['oof_f1']:.4f}", flush=True)
print(f"   v38a Ensemble:    {best_f1_oof:.4f}", flush=True)

change_vs_v34a = (best_f1_oof - v34a['oof_f1']) * 100 / v34a['oof_f1']
change_vs_v37a = (best_f1_oof - v37a['oof_f1']) * 100 / v37a['oof_f1']

print(f"   Change vs v34a:   {change_vs_v34a:+.2f}%", flush=True)
print(f"   Change vs v37a:   {change_vs_v37a:+.2f}%", flush=True)

if best_f1_oof > v34a['oof_f1']:
    print(f"   SUCCESS: Ensemble beats v34a by {best_f1_oof - v34a['oof_f1']:.4f}!", flush=True)
elif best_f1_oof > v37a['oof_f1']:
    print(f"   SUCCESS: Ensemble beats v37a by {best_f1_oof - v37a['oof_f1']:.4f}!", flush=True)
else:
    print(f"   WARNING: Ensemble did not improve - models may be highly correlated", flush=True)

# ====================
# 5. APPLY TO TEST SET
# ====================
print("\n5. Applying rank averaging to test predictions...", flush=True)

# Get test predictions
test_v34a = v34a['test_preds']
test_v37a = v37a['test_preds']

print(f"   Test predictions shape: {test_v34a.shape}", flush=True)

# Convert to ranks
rank_v34a_test = rankdata(-test_v34a, method='average')
rank_v37a_test = rankdata(-test_v37a, method='average')

# Average ranks
rank_avg_test = (rank_v34a_test + rank_v37a_test) / 2.0

print(f"   Test rank range: [{rank_avg_test.min():.0f}, {rank_avg_test.max():.0f}]", flush=True)

# Predict top N as TDEs (using optimal N from OOF)
rank_threshold_test = np.partition(rank_avg_test, best_n_tde)[best_n_tde]
test_final = (rank_avg_test <= rank_threshold_test).astype(int)

print(f"   Predicted TDEs: {test_final.sum()} / {len(test_final)}", flush=True)
print(f"   v34a predicted: {(test_v34a > v34a['best_threshold']).sum()}", flush=True)
print(f"   v37a predicted: {(test_v37a > v37a['best_threshold']).sum()}", flush=True)

# ====================
# 6. CREATE SUBMISSION
# ====================
print("\n6. Creating submission...", flush=True)

test_ids = data['test_meta']['object_id'].tolist()

submission = pd.DataFrame({
    'object_id': test_ids,
    'target': test_final
})

submission_path = base_path / 'submissions/submission_v38a_ensemble_rank.csv'
submission.to_csv(submission_path, index=False)

print(f"   Submission saved: {submission_path.name}", flush=True)

# ====================
# 7. ANALYZE AGREEMENT
# ====================
print("\n7. Model agreement analysis:", flush=True)

# On OOF
oof_v34a_binary = (oof_v34a > v34a['best_threshold']).astype(int)
oof_v37a_binary = (oof_v37a > v37a['best_threshold']).astype(int)

agreement = np.sum(oof_v34a_binary == oof_v37a_binary) / len(oof_v34a_binary)
print(f"   OOF agreement: {100*agreement:.1f}%", flush=True)

both_predict_tde = np.sum((oof_v34a_binary == 1) & (oof_v37a_binary == 1))
only_v34a_tde = np.sum((oof_v34a_binary == 1) & (oof_v37a_binary == 0))
only_v37a_tde = np.sum((oof_v34a_binary == 0) & (oof_v37a_binary == 1))
both_predict_non_tde = np.sum((oof_v34a_binary == 0) & (oof_v37a_binary == 0))

print(f"   Both predict TDE: {both_predict_tde}", flush=True)
print(f"   Only v34a predicts TDE: {only_v34a_tde}", flush=True)
print(f"   Only v37a predicts TDE: {only_v37a_tde}", flush=True)
print(f"   Both predict non-TDE: {both_predict_non_tde}", flush=True)

# When they disagree, who's right?
disagree_mask = oof_v34a_binary != oof_v37a_binary
if np.sum(disagree_mask) > 0:
    v34a_correct_on_disagree = np.sum((oof_v34a_binary[disagree_mask] == y[disagree_mask]))
    v37a_correct_on_disagree = np.sum((oof_v37a_binary[disagree_mask] == y[disagree_mask]))
    total_disagree = np.sum(disagree_mask)

    print(f"\n   When models disagree ({total_disagree} cases):", flush=True)
    print(f"   v34a correct: {v34a_correct_on_disagree} ({100*v34a_correct_on_disagree/total_disagree:.1f}%)", flush=True)
    print(f"   v37a correct: {v37a_correct_on_disagree} ({100*v37a_correct_on_disagree/total_disagree:.1f}%)", flush=True)

# Correlation of probabilities
from scipy.stats import pearsonr, spearmanr
pearson_corr, _ = pearsonr(oof_v34a, oof_v37a)
spearman_corr, _ = spearmanr(oof_v34a, oof_v37a)

print(f"\n   Probability correlation:", flush=True)
print(f"   Pearson: {pearson_corr:.3f}", flush=True)
print(f"   Spearman: {spearman_corr:.3f}", flush=True)

if spearman_corr < 0.85:
    print(f"   GOOD: Low correlation - models are complementary!", flush=True)
else:
    print(f"   WARNING: High correlation - models may be too similar", flush=True)

# Save ensemble artifacts
artifacts = {
    'oof_preds': rank_avg_oof,
    'test_preds': rank_avg_test,
    'best_n_tde': best_n_tde,
    'best_rank_thresh': best_rank_thresh,
    'oof_f1': best_f1_oof,
    'v34a_oof_f1': v34a['oof_f1'],
    'v37a_oof_f1': v37a['oof_f1']
}

with open(base_path / 'data/processed/v38a_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "=" * 80, flush=True)
print(f"MALLORN v38a (Rank Averaging Ensemble) Complete", flush=True)
print(f"OOF F1 = {best_f1_oof:.4f}", flush=True)
print(f"v34a (Bazin):       OOF={v34a['oof_f1']:.4f}, LB=0.6907", flush=True)
print(f"v37a (TDE Physics): OOF={v37a['oof_f1']:.4f}, LB=0.6809", flush=True)

if best_f1_oof > max(v34a['oof_f1'], v37a['oof_f1']):
    print(f"SUCCESS: Ensemble improves over both individual models!", flush=True)
    print(f"Expected LB: 0.69-0.71 (combining best of both approaches)", flush=True)
else:
    print(f"Ensemble did not beat both - may still help via diversity", flush=True)
    print(f"Expected LB: 0.685-0.695 (between individual models)", flush=True)

print("\nNEXT STEP: Submit v38a to leaderboard", flush=True)
print("=" * 80, flush=True)
