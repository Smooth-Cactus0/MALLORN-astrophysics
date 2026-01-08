"""
MALLORN v16: LSTM Trained on Combined PLAsTiCC + MALLORN Data

Key strategy:
- Train on combined dataset (643 TDEs instead of 148)
- Validate ONLY on MALLORN data (competition distribution)
- Test if external data improves generalization

Expected: With 4.3x more TDEs, the LSTM should learn better patterns.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.lightcurve_dataset import LightcurveDataset, collate_fn
from models.lstm_classifier import LSTMClassifier, WeightedBCELoss


def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score."""
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        features = batch['features'].to(device)
        bands = batch['bands'].to(device)
        mask = batch['mask'].to(device)
        metadata = batch['metadata'].to(device)
        labels = batch['label'].to(device)

        if torch.isnan(features).any() or torch.isnan(metadata).any():
            continue

        optimizer.zero_grad()

        output = model(features, bands, mask, metadata)
        loss = criterion(output['logits'], labels)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, device):
    """Evaluate and return predictions."""
    model.eval()
    all_probs = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            bands = batch['bands'].to(device)
            mask = batch['mask'].to(device)
            metadata = batch['metadata'].to(device)

            output = model(features, bands, mask, metadata)

            all_probs.extend(output['probs'].cpu().numpy())
            all_ids.extend(batch['object_ids'])

            if 'label' in batch:
                all_labels.extend(batch['label'].numpy())

    return np.array(all_probs), np.array(all_labels) if all_labels else None, all_ids


def main():
    print("=" * 60)
    print("MALLORN v16: LSTM with Combined PLAsTiCC + MALLORN Data")
    print("=" * 60)

    base_path = Path(__file__).parent.parent

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load combined data
    print("\n1. Loading combined dataset...")

    combined_path = base_path / 'data/processed/combined_plasticc_mallorn.pkl'
    with open(combined_path, 'rb') as f:
        data = pickle.load(f)

    combined_lc = data['combined_lc']
    combined_meta = data['combined_meta']
    mallorn_meta = data['mallorn_meta']
    mallorn_lc = data['mallorn_lc']

    print(f"   Combined dataset: {len(combined_meta)} objects ({(combined_meta['target'] == 1).sum()} TDEs)")
    print(f"   MALLORN subset: {len(mallorn_meta)} objects ({(mallorn_meta['target'] == 1).sum()} TDEs)")

    # 2. Model configuration
    print("\n2. Model configuration...")

    config = {
        'max_length': 300,
        'hidden_dim': 128,      # Larger model for more data
        'n_layers': 2,
        'dropout': 0.3,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'n_epochs': 40,
        'patience': 7,
        'pos_weight': 12.0      # Adjusted for 8.1% TDE rate
    }

    for k, v in config.items():
        print(f"   {k}: {v}")

    # 3. Prepare data structures
    print("\n3. Preparing training data...")

    # Create labels dict for all samples
    labels_dict = dict(zip(combined_meta['object_id'], combined_meta['target']))

    # MALLORN IDs and labels for validation
    mallorn_ids = mallorn_meta['object_id'].tolist()
    mallorn_labels = mallorn_meta['target'].values

    # All combined IDs for training
    all_ids = combined_meta['object_id'].tolist()

    # 4. K-Fold training (split MALLORN, train on ALL combined for each fold)
    print("\n4. Training with 5-fold cross-validation...")
    print("   (Training on ALL combined data, validating on MALLORN only)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(mallorn_ids))
    fold_models = []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(mallorn_ids, mallorn_labels)):
        print(f"\n   --- Fold {fold + 1}/5 ---")

        # Validation: only MALLORN samples (held out)
        val_ids = [mallorn_ids[i] for i in val_idx]
        val_labels = mallorn_labels[val_idx]

        # Training: ALL combined data EXCEPT validation MALLORN samples
        train_ids = [oid for oid in all_ids if oid not in val_ids]

        n_train_tde = sum(1 for oid in train_ids if labels_dict.get(oid, 0) == 1)
        n_train_non_tde = len(train_ids) - n_train_tde

        print(f"   Training: {len(train_ids)} samples ({n_train_tde} TDE, {n_train_non_tde} non-TDE)")
        print(f"   Validation: {len(val_ids)} samples (MALLORN only)")

        # Create datasets
        train_dataset = LightcurveDataset(
            lightcurves=combined_lc,
            metadata=combined_meta,
            object_ids=train_ids,
            labels=labels_dict,
            max_length=config['max_length']
        )

        val_dataset = LightcurveDataset(
            lightcurves=mallorn_lc,
            metadata=mallorn_meta,
            object_ids=val_ids,
            labels=labels_dict,
            max_length=config['max_length']
        )

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        # Create model
        model = LSTMClassifier(
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            dropout=config['dropout']
        ).to(device)

        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"   Model parameters: {n_params:,}")

        # Loss and optimizer
        criterion = WeightedBCELoss(pos_weight=config['pos_weight'])
        optimizer = AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        # Training loop
        best_f1 = 0
        best_model_state = model.state_dict().copy()
        patience_counter = 0

        for epoch in range(config['n_epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

            # Validation
            val_probs, _, _ = evaluate(model, val_loader, device)
            thresh, val_f1 = find_optimal_threshold(val_labels, val_probs)

            scheduler.step()

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or patience_counter == 0:
                print(f"   Epoch {epoch+1}: loss={train_loss:.4f}, val_F1={val_f1:.4f} (best={best_f1:.4f})")

            if patience_counter >= config['patience']:
                print(f"   Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(best_model_state)
        fold_models.append(model.state_dict())

        # Get final validation predictions
        val_probs, _, val_obj_ids = evaluate(model, val_loader, device)
        for obj_id, prob in zip(val_obj_ids, val_probs):
            idx = mallorn_ids.index(obj_id)
            oof_probs[idx] = prob

        fold_scores.append(best_f1)
        print(f"   Fold {fold+1} best F1: {best_f1:.4f}")

    # 5. OOF evaluation (on MALLORN only)
    print("\n5. Out-of-fold evaluation (MALLORN only)...")

    best_thresh, oof_f1 = find_optimal_threshold(mallorn_labels, oof_probs)
    oof_preds = (oof_probs >= best_thresh).astype(int)

    print(f"   OOF F1: {oof_f1:.4f} @ threshold={best_thresh:.2f}")
    print(f"   Precision: {precision_score(mallorn_labels, oof_preds):.4f}")
    print(f"   Recall: {recall_score(mallorn_labels, oof_preds):.4f}")
    print(f"   Predicted TDEs: {oof_preds.sum()}")
    print(f"   Fold F1 scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"   Mean fold F1: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

    # 6. Test predictions
    print("\n6. Generating test predictions...")

    from utils.data_loader import load_all_data
    test_data = load_all_data()
    test_ids = test_data['test_meta']['object_id'].tolist()

    test_dataset = LightcurveDataset(
        lightcurves=test_data['test_lc'],
        metadata=test_data['test_meta'],
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

    # Average predictions across folds
    test_probs_all = []

    for fold_idx, state_dict in enumerate(fold_models):
        model = LSTMClassifier(
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            dropout=config['dropout']
        ).to(device)
        model.load_state_dict(state_dict)

        test_probs, _, test_obj_ids = evaluate(model, test_loader, device)
        test_probs_all.append(test_probs)

    test_probs = np.mean(test_probs_all, axis=0)
    test_preds = (test_probs >= best_thresh).astype(int)

    # 7. Create submission
    print("\n7. Creating submission...")

    submission = pd.DataFrame({
        'object_id': test_obj_ids,
        'target': test_preds
    })

    submission_path = base_path / 'submissions' / 'submission_v16_plasticc.csv'
    submission.to_csv(submission_path, index=False)
    print(f"   Saved to {submission_path}")
    print(f"   Predictions: {test_preds.sum()} TDEs / {len(test_preds)} total ({test_preds.sum()/len(test_preds)*100:.1f}%)")

    # 8. Save models
    models_path = base_path / 'data/processed/models_v16.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump({
            'fold_models': fold_models,
            'config': config,
            'best_thresh': best_thresh,
            'oof_probs': oof_probs,
            'oof_f1': oof_f1
        }, f)
    print(f"   Models saved to {models_path}")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nDL Model Comparison:")
    print(f"  v11 LSTM (148 TDEs, MALLORN only): OOF F1 = 0.1200")
    print(f"  v16 LSTM (643 TDEs, +PLAsTiCC):    OOF F1 = {oof_f1:.4f}")
    print(f"  Improvement: {(oof_f1 - 0.12) / 0.12 * 100:+.1f}%")
    print(f"\nReference (GBM):")
    print(f"  v8 GBM: OOF F1 = 0.6262, LB = 0.6481")
    print("=" * 60)


if __name__ == "__main__":
    main()
