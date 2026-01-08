"""
MALLORN v15: Ensemble of Best Models

Combines:
1. GBM v8 (XGB+LGB+CB): Best overall F1, solid precision/recall balance
2. NN v14: Feature-based neural network, different model family
3. LSTM v11: High recall (72%), can catch TDEs that GBM misses

Ensemble strategies tested:
1. Simple averaging
2. Weighted averaging (optimized weights)
3. Rank averaging
4. Stacking with meta-learner
5. Boosted recall: use LSTM to boost GBM predictions
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')


def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score."""
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


def optimize_weights(probs_list, y_true):
    """Find optimal weights using scipy optimization."""
    def neg_f1(weights):
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        combined = sum(w * p for w, p in zip(weights, probs_list))
        _, f1 = find_optimal_threshold(y_true, combined)
        return -f1

    n_models = len(probs_list)
    initial = [1.0 / n_models] * n_models
    bounds = [(0.01, 1.0)] * n_models

    result = minimize(neg_f1, initial, method='L-BFGS-B', bounds=bounds)
    weights = result.x / result.x.sum()

    return weights


def rank_average(probs_list):
    """Combine predictions using rank averaging."""
    ranks = [rankdata(p) / len(p) for p in probs_list]
    return np.mean(ranks, axis=0)


def main():
    print("=" * 60)
    print("MALLORN v15: Ensemble of Best Models")
    print("=" * 60)

    base_path = Path(__file__).parent.parent

    # 1. Load all model predictions
    print("\n1. Loading model predictions...")

    import sys
    sys.path.insert(0, str(base_path / 'src'))
    from utils.data_loader import load_all_data
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    data = load_all_data()
    y_true = data['train_meta']['target'].values

    # Load v8 features for regenerating OOF predictions
    cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
    train_features = cached['train_features']
    test_features = cached['test_features']

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

    train_combined = train_features[['object_id'] + selected_120].merge(train_tde, on='object_id', how='left')
    test_combined = test_features[['object_id'] + selected_120].merge(test_tde, on='object_id', how='left')
    train_combined = train_combined.merge(data['train_meta'][['object_id', 'target']], on='object_id')

    all_feature_cols = selected_120 + tde_cols
    X = train_combined[all_feature_cols].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X_test = test_combined[all_feature_cols].values
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    # GBM v8 - regenerate OOF predictions
    print("   Regenerating GBM v8 OOF predictions...")
    with open(base_path / 'data/processed/models_v8.pkl', 'rb') as f:
        v8 = pickle.load(f)

    gbm_oof = np.zeros(len(X))
    gbm_test_probs = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_true)):
        X_val = X[val_idx]

        # Get predictions from each model
        xgb_pred = v8['xgb_models'][fold].predict_proba(X_val)[:, 1]
        lgb_pred = v8['lgb_models'][fold].predict_proba(X_val)[:, 1]
        cat_pred = v8['cat_models'][fold].predict_proba(X_val)[:, 1]

        # Weighted ensemble
        weights = v8['best_weights']
        fold_pred = weights[0] * xgb_pred + weights[1] * lgb_pred + weights[2] * cat_pred
        gbm_oof[val_idx] = fold_pred

    # Test predictions (average across folds)
    for fold in range(5):
        xgb_test = v8['xgb_models'][fold].predict_proba(X_test)[:, 1]
        lgb_test = v8['lgb_models'][fold].predict_proba(X_test)[:, 1]
        cat_test = v8['cat_models'][fold].predict_proba(X_test)[:, 1]
        fold_test = weights[0] * xgb_test + weights[1] * lgb_test + weights[2] * cat_test
        gbm_test_probs.append(fold_test)

    test_gbm = np.mean(gbm_test_probs, axis=0)

    gbm_thresh, gbm_f1 = find_optimal_threshold(y_true, gbm_oof)
    print(f"   GBM v8: OOF F1 = {gbm_f1:.4f}, thresh = {gbm_thresh:.2f}")

    # NN v14
    with open(base_path / 'data/processed/models_v14.pkl', 'rb') as f:
        v14 = pickle.load(f)
    nn_oof = v14['oof_probs']
    nn_thresh = v14['best_thresh']
    print(f"   NN v14: OOF F1 = {v14['oof_f1']:.4f}, thresh = {nn_thresh:.2f}")

    # LSTM v11
    with open(base_path / 'data/processed/models_v11.pkl', 'rb') as f:
        v11 = pickle.load(f)
    lstm_oof = v11['oof_probs']
    lstm_thresh = v11['best_thresh']
    print(f"   LSTM v11: OOF F1 = {v11['oof_f1']:.4f}, thresh = {lstm_thresh:.2f}")

    print(f"\n   Total samples: {len(y_true)}, TDEs: {y_true.sum()}")

    # 2. Test individual model performances
    print("\n2. Individual model OOF performance:")

    models = {
        'GBM v8': gbm_oof,
        'NN v14': nn_oof,
        'LSTM v11': lstm_oof
    }

    for name, probs in models.items():
        thresh, f1 = find_optimal_threshold(y_true, probs)
        preds = (probs >= thresh).astype(int)
        prec = precision_score(y_true, preds)
        rec = recall_score(y_true, preds)
        print(f"   {name}: F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, Thresh={thresh:.2f}, Preds={preds.sum()}")

    # 3. Test ensemble strategies
    print("\n3. Testing ensemble strategies...")

    results = {}

    # Strategy 1: Simple average (all models)
    avg_all = (gbm_oof + nn_oof + lstm_oof) / 3
    thresh, f1 = find_optimal_threshold(y_true, avg_all)
    preds = (avg_all >= thresh).astype(int)
    results['Simple Avg (all)'] = {
        'f1': f1, 'probs': avg_all, 'thresh': thresh,
        'prec': precision_score(y_true, preds),
        'rec': recall_score(y_true, preds),
        'preds': preds.sum()
    }

    # Strategy 2: Simple average (GBM + NN only)
    avg_gbm_nn = (gbm_oof + nn_oof) / 2
    thresh, f1 = find_optimal_threshold(y_true, avg_gbm_nn)
    preds = (avg_gbm_nn >= thresh).astype(int)
    results['Simple Avg (GBM+NN)'] = {
        'f1': f1, 'probs': avg_gbm_nn, 'thresh': thresh,
        'prec': precision_score(y_true, preds),
        'rec': recall_score(y_true, preds),
        'preds': preds.sum()
    }

    # Strategy 3: Weighted average (optimized)
    print("   Optimizing ensemble weights...")
    weights = optimize_weights([gbm_oof, nn_oof, lstm_oof], y_true)
    weighted_all = weights[0] * gbm_oof + weights[1] * nn_oof + weights[2] * lstm_oof
    thresh, f1 = find_optimal_threshold(y_true, weighted_all)
    preds = (weighted_all >= thresh).astype(int)
    results['Weighted Avg'] = {
        'f1': f1, 'probs': weighted_all, 'thresh': thresh,
        'weights': weights,
        'prec': precision_score(y_true, preds),
        'rec': recall_score(y_true, preds),
        'preds': preds.sum()
    }
    print(f"   Optimized weights: GBM={weights[0]:.3f}, NN={weights[1]:.3f}, LSTM={weights[2]:.3f}")

    # Strategy 4: Rank averaging
    rank_avg = rank_average([gbm_oof, nn_oof, lstm_oof])
    thresh, f1 = find_optimal_threshold(y_true, rank_avg)
    preds = (rank_avg >= thresh).astype(int)
    results['Rank Avg'] = {
        'f1': f1, 'probs': rank_avg, 'thresh': thresh,
        'prec': precision_score(y_true, preds),
        'rec': recall_score(y_true, preds),
        'preds': preds.sum()
    }

    # Strategy 5: Stacking with logistic regression
    print("   Training stacking meta-learner...")
    stacking_features = np.column_stack([gbm_oof, nn_oof, lstm_oof])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stacking_oof = np.zeros(len(y_true))

    for train_idx, val_idx in skf.split(stacking_features, y_true):
        X_tr, X_val = stacking_features[train_idx], stacking_features[val_idx]
        y_tr = y_true[train_idx]

        meta = LogisticRegression(C=1.0, class_weight='balanced')
        meta.fit(X_tr, y_tr)
        stacking_oof[val_idx] = meta.predict_proba(X_val)[:, 1]

    thresh, f1 = find_optimal_threshold(y_true, stacking_oof)
    preds = (stacking_oof >= thresh).astype(int)
    results['Stacking (LR)'] = {
        'f1': f1, 'probs': stacking_oof, 'thresh': thresh,
        'prec': precision_score(y_true, preds),
        'rec': recall_score(y_true, preds),
        'preds': preds.sum()
    }

    # Strategy 6: LSTM-boosted GBM
    # Use LSTM's high recall to boost uncertain GBM predictions
    lstm_high = lstm_oof > 0.5  # LSTM thinks it's TDE
    gbm_uncertain = (gbm_oof > 0.1) & (gbm_oof < 0.6)  # GBM uncertain
    boost_mask = lstm_high & gbm_uncertain
    lstm_boosted = gbm_oof.copy()
    lstm_boosted[boost_mask] = gbm_oof[boost_mask] * 1.3  # Boost by 30%
    lstm_boosted = np.clip(lstm_boosted, 0, 1)

    thresh, f1 = find_optimal_threshold(y_true, lstm_boosted)
    preds = (lstm_boosted >= thresh).astype(int)
    results['LSTM-boosted GBM'] = {
        'f1': f1, 'probs': lstm_boosted, 'thresh': thresh,
        'prec': precision_score(y_true, preds),
        'rec': recall_score(y_true, preds),
        'preds': preds.sum()
    }

    # Strategy 7: Conservative ensemble (GBM primary, NN for confirmation)
    # Only predict TDE if both GBM and NN agree
    conservative = gbm_oof * nn_oof  # Product emphasizes agreement
    conservative = np.sqrt(conservative)  # Geometric mean
    thresh, f1 = find_optimal_threshold(y_true, conservative)
    preds = (conservative >= thresh).astype(int)
    results['Conservative (GBM*NN)'] = {
        'f1': f1, 'probs': conservative, 'thresh': thresh,
        'prec': precision_score(y_true, preds),
        'rec': recall_score(y_true, preds),
        'preds': preds.sum()
    }

    # 4. Print results
    print("\n4. Ensemble Results:")
    print("   " + "-" * 75)
    print(f"   {'Strategy':<25} | {'F1':>6} | {'Prec':>6} | {'Rec':>6} | {'Thresh':>6} | {'Preds':>5}")
    print("   " + "-" * 75)

    for name, r in results.items():
        print(f"   {name:<25} | {r['f1']:.4f} | {r['prec']:.4f} | {r['rec']:.4f} | {r['thresh']:.2f}   | {r['preds']:5d}")

    print("   " + "-" * 75)
    print(f"   {'GBM v8 (reference)':<25} | {gbm_f1:.4f} | ---    | ---    | {gbm_thresh:.2f}   | ---")

    # Find best strategy
    best_strategy = max(results.keys(), key=lambda x: results[x]['f1'])
    best_f1 = results[best_strategy]['f1']

    print(f"\n   Best ensemble: {best_strategy} (F1 = {best_f1:.4f})")

    # 5. Generate test predictions with best strategy
    print("\n5. Generating test predictions...")

    # test_gbm already generated above in section 1

    # For LSTM-boosted GBM, we need LSTM test predictions
    # Load LSTM test probs - need to regenerate from models
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(base_path / 'src'))
    from models.lightcurve_dataset import LightcurveDataset, collate_fn
    from models.lstm_classifier import LSTMClassifier

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_ids = data['test_meta']['object_id'].tolist()
    lstm_config = v11['config']

    test_dataset = LightcurveDataset(
        lightcurves=data['test_lc'],
        metadata=data['test_meta'],
        object_ids=test_ids,
        labels=None,
        max_length=lstm_config['max_length']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=lstm_config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Get LSTM test predictions
    test_lstm_all = []
    for fold_idx, state_dict in enumerate(v11['fold_models']):
        model = LSTMClassifier(
            hidden_dim=lstm_config['hidden_dim'],
            n_layers=lstm_config['n_layers'],
            dropout=lstm_config['dropout']
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        fold_probs = []
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(device)
                bands = batch['bands'].to(device)
                mask = batch['mask'].to(device)
                metadata = batch['metadata'].to(device)
                output = model(features, bands, mask, metadata)
                fold_probs.extend(output['probs'].cpu().numpy())

        test_lstm_all.append(np.array(fold_probs))

    test_lstm_probs = np.mean(test_lstm_all, axis=0)

    # Apply LSTM-boosted GBM strategy to test data
    lstm_high = test_lstm_probs > 0.5
    gbm_uncertain = (test_gbm > 0.1) & (test_gbm < 0.6)
    boost_mask = lstm_high & gbm_uncertain
    test_probs = test_gbm.copy()
    test_probs[boost_mask] = test_gbm[boost_mask] * 1.3
    test_probs = np.clip(test_probs, 0, 1)

    print(f"   Applied LSTM-boosted GBM strategy")
    print(f"   Boosted {boost_mask.sum()} predictions")

    test_thresh = results['LSTM-boosted GBM']['thresh']
    test_preds = (test_probs >= test_thresh).astype(int)

    # 6. Create submission
    print("\n6. Creating submission...")

    # Get test object IDs from GBM submission
    gbm_sub = pd.read_csv(base_path / 'submissions/submission_v8_tuned.csv')
    test_ids = gbm_sub['object_id'].values

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': test_preds
    })

    submission_path = base_path / 'submissions' / 'submission_v15_ensemble.csv'
    submission.to_csv(submission_path, index=False)
    print(f"   Saved to {submission_path}")
    print(f"   Predictions: {test_preds.sum()} TDEs / {len(test_preds)} total ({test_preds.sum()/len(test_preds)*100:.1f}%)")

    # 7. Save ensemble results
    ensemble_path = base_path / 'data/processed/ensemble_v15.pkl'
    with open(ensemble_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'best_strategy': best_strategy,
            'best_f1': best_f1,
            'weights': weights if 'weights' in results['Weighted Avg'] else None
        }, f)
    print(f"   Results saved to {ensemble_path}")

    # Summary
    print("\n" + "=" * 60)
    print("ENSEMBLE COMPLETE!")
    print("=" * 60)
    print(f"\nModel Performance Summary:")
    print(f"  GBM v8 alone:           OOF F1 = {gbm_f1:.4f}, LB = 0.6481")
    print(f"  Best ensemble:          OOF F1 = {best_f1:.4f} ({best_strategy})")
    print(f"  Improvement:            {(best_f1 - gbm_f1)/gbm_f1*100:+.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
