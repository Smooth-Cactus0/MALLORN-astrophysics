"""
MALLORN v54: Ensemble of Best Performing Models

Combining our top 3 LB performers:
- v34a: LB 0.6907 (XGBoost + Bazin features) - BEST
- v48:  LB 0.6748 (XGBoost + time-to-decline features)
- v50:  LB 0.6721 (XGBoost + GP augmentation all classes)

Ensemble strategies:
1. Simple averaging of probabilities
2. Weighted averaging (by LB performance)
3. Rank averaging
4. Voting ensemble (majority vote on binary predictions)

Target: Close the gap to 0.73 (current top of leaderboard)
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

sys.stdout.reconfigure(line_buffering=True)

base_path = Path(__file__).parent.parent

print("=" * 80, flush=True)
print("MALLORN v54: Ensemble of Best Performing Models", flush=True)
print("=" * 80, flush=True)

# ====================
# 1. LOAD MODEL PREDICTIONS
# ====================
print("\n1. Loading model predictions...", flush=True)

# Load artifacts
with open(base_path / 'data/processed/v34a_artifacts.pkl', 'rb') as f:
    v34a = pickle.load(f)

with open(base_path / 'data/processed/v48_artifacts.pkl', 'rb') as f:
    v48 = pickle.load(f)

with open(base_path / 'data/processed/v50_artifacts.pkl', 'rb') as f:
    v50 = pickle.load(f)

# Get test predictions (probabilities)
pred_v34a = v34a['test_preds']
pred_v48 = v48['test_preds']
pred_v50 = v50['test_preds']

# Get thresholds
thresh_v34a = v34a['best_threshold']
thresh_v48 = v48['best_threshold']
thresh_v50 = v50['best_threshold']

print(f"   v34a: LB=0.6907, threshold={thresh_v34a:.3f}, shape={pred_v34a.shape}", flush=True)
print(f"   v48:  LB=0.6748, threshold={thresh_v48:.3f}, shape={pred_v48.shape}", flush=True)
print(f"   v50:  LB=0.6721, threshold={thresh_v50:.3f}, shape={pred_v50.shape}", flush=True)

# Load test IDs
test_meta = pd.read_csv(base_path / 'data/raw/test_log.csv')
test_ids = test_meta['object_id'].tolist()
print(f"   Test objects: {len(test_ids)}", flush=True)

# ====================
# 2. ENSEMBLE STRATEGIES
# ====================
print("\n2. Creating ensemble predictions...", flush=True)

# LB scores for weighting
lb_scores = {
    'v34a': 0.6907,
    'v48': 0.6748,
    'v50': 0.6721
}

# Strategy 1: Simple Average
print("\n   Strategy 1: Simple Average", flush=True)
avg_preds = (pred_v34a + pred_v48 + pred_v50) / 3
print(f"      Mean prob: {avg_preds.mean():.4f}", flush=True)

# Strategy 2: Weighted Average (by LB score)
print("\n   Strategy 2: Weighted Average (by LB performance)", flush=True)
total_lb = sum(lb_scores.values())
w_v34a = lb_scores['v34a'] / total_lb
w_v48 = lb_scores['v48'] / total_lb
w_v50 = lb_scores['v50'] / total_lb
print(f"      Weights: v34a={w_v34a:.3f}, v48={w_v48:.3f}, v50={w_v50:.3f}", flush=True)

weighted_preds = w_v34a * pred_v34a + w_v48 * pred_v48 + w_v50 * pred_v50
print(f"      Mean prob: {weighted_preds.mean():.4f}", flush=True)

# Strategy 3: Rank Average
print("\n   Strategy 3: Rank Average", flush=True)
rank_v34a = rankdata(pred_v34a) / len(pred_v34a)
rank_v48 = rankdata(pred_v48) / len(pred_v48)
rank_v50 = rankdata(pred_v50) / len(pred_v50)
rank_avg = (rank_v34a + rank_v48 + rank_v50) / 3
print(f"      Mean rank: {rank_avg.mean():.4f}", flush=True)

# Strategy 4: Weighted Rank Average
print("\n   Strategy 4: Weighted Rank Average", flush=True)
weighted_rank = w_v34a * rank_v34a + w_v48 * rank_v48 + w_v50 * rank_v50
print(f"      Mean weighted rank: {weighted_rank.mean():.4f}", flush=True)

# Strategy 5: Voting (majority vote on binary)
print("\n   Strategy 5: Voting Ensemble", flush=True)
binary_v34a = (pred_v34a > thresh_v34a).astype(int)
binary_v48 = (pred_v48 > thresh_v48).astype(int)
binary_v50 = (pred_v50 > thresh_v50).astype(int)
vote_sum = binary_v34a + binary_v48 + binary_v50
voting_preds = (vote_sum >= 2).astype(int)  # At least 2 out of 3 agree
print(f"      TDEs by v34a: {binary_v34a.sum()}", flush=True)
print(f"      TDEs by v48: {binary_v48.sum()}", flush=True)
print(f"      TDEs by v50: {binary_v50.sum()}", flush=True)
print(f"      TDEs by voting (>=2/3): {voting_preds.sum()}", flush=True)

# Strategy 6: Best model + boost from others
print("\n   Strategy 6: v34a dominant (0.6 * v34a + 0.2 * v48 + 0.2 * v50)", flush=True)
dominant_preds = 0.6 * pred_v34a + 0.2 * pred_v48 + 0.2 * pred_v50
print(f"      Mean prob: {dominant_preds.mean():.4f}", flush=True)

# ====================
# 3. FIND OPTIMAL THRESHOLDS
# ====================
print("\n3. Finding optimal thresholds for each ensemble...", flush=True)

# Since we don't have ground truth for test, use average of individual thresholds
avg_thresh = (thresh_v34a + thresh_v48 + thresh_v50) / 3
print(f"   Average threshold from individual models: {avg_thresh:.3f}", flush=True)

# Also try the best performing model's threshold (v34a)
print(f"   Best model threshold (v34a): {thresh_v34a:.3f}", flush=True)

# ====================
# 4. CREATE SUBMISSIONS
# ====================
print("\n4. Creating submission files...", flush=True)

submissions = {}

# Simple Average with average threshold
submissions['v54a_simple_avg'] = {
    'preds': (avg_preds > avg_thresh).astype(int),
    'desc': 'Simple average, avg threshold'
}

# Simple Average with v34a threshold
submissions['v54b_simple_v34a_thresh'] = {
    'preds': (avg_preds > thresh_v34a).astype(int),
    'desc': 'Simple average, v34a threshold'
}

# Weighted Average with v34a threshold
submissions['v54c_weighted_avg'] = {
    'preds': (weighted_preds > thresh_v34a).astype(int),
    'desc': 'Weighted average (by LB), v34a threshold'
}

# Rank Average with percentile threshold
rank_thresh = 0.95  # Top 5% as TDEs
submissions['v54d_rank_avg'] = {
    'preds': (rank_avg > rank_thresh).astype(int),
    'desc': f'Rank average, top {100*(1-rank_thresh):.0f}% threshold'
}

# Voting
submissions['v54e_voting'] = {
    'preds': voting_preds,
    'desc': 'Voting (>=2/3 agree)'
}

# v34a dominant
submissions['v54f_v34a_dominant'] = {
    'preds': (dominant_preds > thresh_v34a).astype(int),
    'desc': 'v34a dominant (0.6/0.2/0.2), v34a threshold'
}

# Create submission files
for name, data in submissions.items():
    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': data['preds']
    })

    submission_path = base_path / f'submissions/submission_{name}.csv'
    submission.to_csv(submission_path, index=False)

    n_tdes = data['preds'].sum()
    print(f"   {name}: {n_tdes} TDEs ({100*n_tdes/len(test_ids):.1f}%) - {data['desc']}", flush=True)

# ====================
# 5. ANALYSIS
# ====================
print("\n" + "=" * 80, flush=True)
print("ENSEMBLE ANALYSIS", flush=True)
print("=" * 80, flush=True)

print("\nPrediction Agreement Analysis:", flush=True)
all_agree_tde = ((binary_v34a == 1) & (binary_v48 == 1) & (binary_v50 == 1)).sum()
all_agree_non_tde = ((binary_v34a == 0) & (binary_v48 == 0) & (binary_v50 == 0)).sum()
some_disagree = len(test_ids) - all_agree_tde - all_agree_non_tde

print(f"   All 3 models agree TDE: {all_agree_tde} ({100*all_agree_tde/len(test_ids):.1f}%)", flush=True)
print(f"   All 3 models agree non-TDE: {all_agree_non_tde} ({100*all_agree_non_tde/len(test_ids):.1f}%)", flush=True)
print(f"   Models disagree: {some_disagree} ({100*some_disagree/len(test_ids):.1f}%)", flush=True)

print("\nCorrelation between model predictions:", flush=True)
corr_34a_48 = np.corrcoef(pred_v34a, pred_v48)[0, 1]
corr_34a_50 = np.corrcoef(pred_v34a, pred_v50)[0, 1]
corr_48_50 = np.corrcoef(pred_v48, pred_v50)[0, 1]
print(f"   v34a vs v48: {corr_34a_48:.4f}", flush=True)
print(f"   v34a vs v50: {corr_34a_50:.4f}", flush=True)
print(f"   v48 vs v50: {corr_48_50:.4f}", flush=True)

print("\nRecommendation:", flush=True)
print("   High correlation suggests models make similar predictions.", flush=True)
print("   For diverse ensemble, consider adding a different model type.", flush=True)

print("\n" + "=" * 80, flush=True)
print("MALLORN v54 Ensemble Complete", flush=True)
print("=" * 80, flush=True)
print("\nSubmission files created:", flush=True)
for name in submissions.keys():
    print(f"   - submission_{name}.csv", flush=True)

print("\nRECOMMENDED: Start with v54e_voting or v54c_weighted_avg", flush=True)
print("These typically generalize better than simple averaging.", flush=True)
print("=" * 80, flush=True)
