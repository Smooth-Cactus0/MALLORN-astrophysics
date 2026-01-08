"""
MALLORN v11-calibrated: Probability Calibration for RNN

The v11 RNN learns to identify TDEs well (72% recall) but predicts
too many positives because its probability outputs aren't calibrated
for the true 5% TDE distribution.

This script applies:
1. Platt Scaling (logistic regression on probabilities)
2. Isotonic Regression (non-parametric calibration)
3. Temperature Scaling (simple but effective)

We compare all three and use the best one.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_score, recall_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score."""
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


def platt_scaling(probs_train, y_train, probs_test):
    """
    Platt Scaling: Fit logistic regression on probabilities.
    Maps uncalibrated probabilities to calibrated ones.
    """
    # Reshape for sklearn
    X_train = probs_train.reshape(-1, 1)
    X_test = probs_test.reshape(-1, 1)

    # Fit logistic regression
    lr = LogisticRegression(C=1.0, solver='lbfgs')
    lr.fit(X_train, y_train)

    # Get calibrated probabilities
    calibrated = lr.predict_proba(X_test)[:, 1]
    return calibrated, lr


def isotonic_calibration(probs_train, y_train, probs_test):
    """
    Isotonic Regression: Non-parametric calibration.
    More flexible than Platt but can overfit.
    """
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(probs_train, y_train)
    calibrated = ir.predict(probs_test)
    return calibrated, ir


def temperature_scaling(probs_train, y_train, probs_test):
    """
    Temperature Scaling: Simple but effective.
    Finds optimal temperature T to divide logits by.
    """
    # Convert probabilities to logits
    eps = 1e-7
    logits_train = np.log(probs_train + eps) - np.log(1 - probs_train + eps)
    logits_test = np.log(probs_test + eps) - np.log(1 - probs_test + eps)

    # Find optimal temperature
    best_t, best_brier = 1.0, float('inf')
    for t in np.arange(0.1, 10.0, 0.1):
        scaled_probs = 1 / (1 + np.exp(-logits_train / t))
        brier = brier_score_loss(y_train, scaled_probs)
        if brier < best_brier:
            best_brier = brier
            best_t = t

    # Apply to test
    calibrated = 1 / (1 + np.exp(-logits_test / best_t))
    return calibrated, best_t


def main():
    print("=" * 60)
    print("MALLORN v11-calibrated: Probability Calibration")
    print("=" * 60)

    base_path = Path(__file__).parent.parent

    # 1. Load v11 results
    print("\n1. Loading v11 model outputs...")

    models_path = base_path / 'data/processed/models_v11.pkl'
    with open(models_path, 'rb') as f:
        v11_data = pickle.load(f)

    oof_probs = v11_data['oof_probs']
    original_thresh = v11_data['best_thresh']
    original_f1 = v11_data['oof_f1']

    # Load original labels
    from utils.data_loader import load_all_data
    data = load_all_data()
    y = data['train_meta']['target'].values

    print(f"   Original v11: OOF F1 = {original_f1:.4f} @ threshold = {original_thresh:.2f}")
    print(f"   Original predictions: {(oof_probs >= original_thresh).sum()} TDEs")

    # 2. Apply calibration methods using cross-validation
    print("\n2. Applying calibration methods (5-fold CV)...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    platt_calibrated = np.zeros_like(oof_probs)
    isotonic_calibrated = np.zeros_like(oof_probs)
    temp_calibrated = np.zeros_like(oof_probs)

    platt_models = []
    isotonic_models = []
    temp_values = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(oof_probs, y)):
        # Split
        probs_train = oof_probs[train_idx]
        probs_val = oof_probs[val_idx]
        y_train = y[train_idx]

        # Platt scaling
        platt_val, platt_model = platt_scaling(probs_train, y_train, probs_val)
        platt_calibrated[val_idx] = platt_val
        platt_models.append(platt_model)

        # Isotonic
        iso_val, iso_model = isotonic_calibration(probs_train, y_train, probs_val)
        isotonic_calibrated[val_idx] = iso_val
        isotonic_models.append(iso_model)

        # Temperature
        temp_val, temp = temperature_scaling(probs_train, y_train, probs_val)
        temp_calibrated[val_idx] = temp_val
        temp_values.append(temp)

    # 3. Evaluate all methods
    print("\n3. Evaluating calibration methods...")

    results = {}

    # Original
    orig_thresh, orig_f1 = find_optimal_threshold(y, oof_probs)
    orig_preds = (oof_probs >= orig_thresh).astype(int)
    results['Original'] = {
        'threshold': orig_thresh,
        'f1': orig_f1,
        'precision': precision_score(y, orig_preds),
        'recall': recall_score(y, orig_preds),
        'n_preds': orig_preds.sum(),
        'brier': brier_score_loss(y, oof_probs)
    }

    # Platt
    platt_thresh, platt_f1 = find_optimal_threshold(y, platt_calibrated)
    platt_preds = (platt_calibrated >= platt_thresh).astype(int)
    results['Platt'] = {
        'threshold': platt_thresh,
        'f1': platt_f1,
        'precision': precision_score(y, platt_preds),
        'recall': recall_score(y, platt_preds),
        'n_preds': platt_preds.sum(),
        'brier': brier_score_loss(y, platt_calibrated)
    }

    # Isotonic
    iso_thresh, iso_f1 = find_optimal_threshold(y, isotonic_calibrated)
    iso_preds = (isotonic_calibrated >= iso_thresh).astype(int)
    results['Isotonic'] = {
        'threshold': iso_thresh,
        'f1': iso_f1,
        'precision': precision_score(y, iso_preds),
        'recall': recall_score(y, iso_preds),
        'n_preds': iso_preds.sum(),
        'brier': brier_score_loss(y, isotonic_calibrated)
    }

    # Temperature
    temp_thresh, temp_f1 = find_optimal_threshold(y, temp_calibrated)
    temp_preds = (temp_calibrated >= temp_thresh).astype(int)
    results['Temperature'] = {
        'threshold': temp_thresh,
        'f1': temp_f1,
        'precision': precision_score(y, temp_preds),
        'recall': recall_score(y, temp_preds),
        'n_preds': temp_preds.sum(),
        'brier': brier_score_loss(y, temp_calibrated)
    }

    # Print comparison
    print("\n   Method       | F1     | Precision | Recall | Preds | Brier")
    print("   " + "-" * 60)
    for method, r in results.items():
        print(f"   {method:12} | {r['f1']:.4f} | {r['precision']:.4f}    | {r['recall']:.4f} | {r['n_preds']:5d} | {r['brier']:.4f}")

    # Find best method
    best_method = max(results.keys(), key=lambda x: results[x]['f1'])
    best_f1 = results[best_method]['f1']

    print(f"\n   Best method: {best_method} (F1 = {best_f1:.4f})")

    # 4. Generate test predictions with best calibration
    print("\n4. Generating calibrated test predictions...")

    # Load test probabilities from v11
    # We need to regenerate them using the saved models
    import torch
    from torch.utils.data import DataLoader
    from models.lightcurve_dataset import LightcurveDataset, collate_fn
    from models.lstm_classifier import LSTMClassifier

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = v11_data['config']

    test_ids = data['test_meta']['object_id'].tolist()

    test_dataset = LightcurveDataset(
        lightcurves=data['test_lc'],
        metadata=data['test_meta'],
        object_ids=test_ids,
        labels=None,
        max_length=config['max_length']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Get test predictions from all folds
    test_probs_all = []

    for fold_idx, state_dict in enumerate(v11_data['fold_models']):
        model = LSTMClassifier(
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            dropout=config['dropout']
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

        test_probs_all.append(np.array(fold_probs))

    test_probs_raw = np.mean(test_probs_all, axis=0)

    # Apply best calibration
    if best_method == 'Platt':
        # Average calibration across folds
        test_probs_calibrated = np.mean([
            m.predict_proba(test_probs_raw.reshape(-1, 1))[:, 1]
            for m in platt_models
        ], axis=0)
        best_thresh = platt_thresh
    elif best_method == 'Isotonic':
        test_probs_calibrated = np.mean([
            m.predict(test_probs_raw)
            for m in isotonic_models
        ], axis=0)
        best_thresh = iso_thresh
    elif best_method == 'Temperature':
        avg_temp = np.mean(temp_values)
        eps = 1e-7
        logits = np.log(test_probs_raw + eps) - np.log(1 - test_probs_raw + eps)
        test_probs_calibrated = 1 / (1 + np.exp(-logits / avg_temp))
        best_thresh = temp_thresh
    else:
        test_probs_calibrated = test_probs_raw
        best_thresh = orig_thresh

    test_preds = (test_probs_calibrated >= best_thresh).astype(int)

    # 5. Create submission
    print("\n5. Creating calibrated submission...")

    # Get test object IDs in order
    test_obj_ids = []
    for batch in test_loader:
        test_obj_ids.extend(batch['object_ids'])

    submission = pd.DataFrame({
        'object_id': test_obj_ids,
        'target': test_preds
    })

    submission_path = base_path / 'submissions' / 'submission_v11_calibrated.csv'
    submission.to_csv(submission_path, index=False)

    print(f"   Saved to {submission_path}")
    print(f"   Predictions: {test_preds.sum()} TDEs / {len(test_preds)} total ({test_preds.sum()/len(test_preds)*100:.1f}%)")

    # 6. Save calibration models
    calibration_path = base_path / 'data/processed/calibration_v11.pkl'
    with open(calibration_path, 'wb') as f:
        pickle.dump({
            'best_method': best_method,
            'platt_models': platt_models,
            'isotonic_models': isotonic_models,
            'temp_values': temp_values,
            'best_thresh': best_thresh,
            'results': results
        }, f)
    print(f"   Calibration saved to {calibration_path}")

    # Summary
    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE!")
    print("=" * 60)
    print(f"\nBefore calibration:")
    print(f"  v11 OOF F1: {original_f1:.4f}, predicted {(oof_probs >= original_thresh).sum()} OOF TDEs")
    print(f"\nAfter {best_method} calibration:")
    print(f"  OOF F1: {best_f1:.4f}, predicted {results[best_method]['n_preds']} OOF TDEs")
    print(f"  Improvement: {(best_f1 - original_f1) / original_f1 * 100:+.1f}%")
    print(f"\nTest predictions: {test_preds.sum()} TDEs (was 3684 before calibration)")
    print("=" * 60)


if __name__ == "__main__":
    main()
