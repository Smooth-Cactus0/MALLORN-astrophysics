"""
MALLORN v11: LSTM with Data Augmentation

Trains the same LSTM architecture but with augmented TDE samples
to address the severe class imbalance problem.

Augmentation strategy:
- 148 original TDEs → ~1,800 augmented TDEs (12x increase)
- Techniques: flux scaling, time stretching, noise injection,
  observation dropout, and mixup

This creates a more balanced dataset for the RNN to learn from.
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
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from utils.data_loader import load_all_data
from models.lightcurve_dataset import LightcurveDataset, collate_fn
from models.lstm_classifier import LSTMClassifier, WeightedBCELoss
from features.augmentation import augment_tde_dataset


def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score."""
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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
    print("MALLORN v11: LSTM with Data Augmentation")
    print("=" * 60)

    base_path = Path(__file__).parent.parent

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load data
    print("\n1. Loading data...")
    data = load_all_data()

    train_meta = data['train_meta'].copy()
    train_lc = data['train_lc'].copy()

    original_tde_count = (train_meta['target'] == 1).sum()
    original_non_tde_count = (train_meta['target'] == 0).sum()
    print(f"   Original: {original_tde_count} TDEs, {original_non_tde_count} non-TDEs")

    # 2. Generate augmented TDEs
    print("\n2. Generating augmented TDE samples...")

    aug_lc, aug_meta = augment_tde_dataset(
        train_lc,
        train_meta,
        augmentations_per_tde=10,  # 10 augmented versions each
        include_mixup=True,
        mixup_per_tde=2,           # 2 mixup versions each
        random_state=42
    )

    # Combine original + augmented
    combined_lc = pd.concat([train_lc, aug_lc], ignore_index=True)
    combined_meta = pd.concat([train_meta, aug_meta], ignore_index=True)

    # Get counts
    augmented_tde_count = (combined_meta['target'] == 1).sum()
    total_count = len(combined_meta)

    print(f"\n   After augmentation:")
    print(f"   - TDEs: {original_tde_count} -> {augmented_tde_count} ({augmented_tde_count/original_tde_count:.1f}x)")
    print(f"   - Total samples: {total_count}")
    print(f"   - Class balance: {augmented_tde_count/total_count*100:.1f}% TDE")

    # 3. Model configuration
    print("\n3. Model configuration...")

    config = {
        'max_length': 300,
        'hidden_dim': 64,
        'n_layers': 2,
        'dropout': 0.3,
        'batch_size': 64,         # Larger batch with more data
        'lr': 1e-3,               # Can use higher LR with more data
        'weight_decay': 1e-4,
        'n_epochs': 30,
        'patience': 5,
        'pos_weight': 3.0         # Less extreme weighting now (more balanced data)
    }

    for k, v in config.items():
        print(f"   {k}: {v}")

    # 4. Prepare data structures
    print("\n4. Preparing training data...")

    # Create labels dict for all samples (original + augmented)
    labels_dict = dict(zip(combined_meta['object_id'], combined_meta['target']))

    # For validation, we only use ORIGINAL samples (not augmented)
    original_ids = train_meta['object_id'].tolist()
    original_labels = train_meta['target'].values

    # 5. K-Fold training (validate on original data only)
    print("\n5. Training with 5-fold cross-validation...")
    print("   (Training on augmented data, validating on original only)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(original_ids))
    fold_models = []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(original_ids, original_labels)):
        print(f"\n   --- Fold {fold + 1}/5 ---")

        # Validation: only original samples
        val_ids = [original_ids[i] for i in val_idx]
        val_labels = original_labels[val_idx]

        # Training: original train samples + ALL augmented samples
        # (augmented samples are only from TDEs, so they're all "new")
        original_train_ids = [original_ids[i] for i in train_idx]

        # Get augmented IDs that correspond to training TDEs
        train_tde_ids = set(train_meta[train_meta['object_id'].isin(original_train_ids) &
                                        (train_meta['target'] == 1)]['object_id'])

        # Include augmented versions of training TDEs only
        aug_train_ids = [oid for oid in aug_meta['object_id']
                         if any(oid.startswith(tde_id + '_') for tde_id in train_tde_ids)]

        train_ids = original_train_ids + aug_train_ids

        print(f"   Training samples: {len(train_ids)} ({len(original_train_ids)} original + {len(aug_train_ids)} augmented)")
        print(f"   Validation samples: {len(val_ids)} (original only)")

        # Create datasets
        train_dataset = LightcurveDataset(
            lightcurves=combined_lc,
            metadata=combined_meta,
            object_ids=train_ids,
            labels=labels_dict,
            max_length=config['max_length']
        )

        val_dataset = LightcurveDataset(
            lightcurves=train_lc,  # Original data only for validation
            metadata=train_meta,
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

        # Loss and optimizer
        criterion = WeightedBCELoss(pos_weight=config['pos_weight'])
        optimizer = AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=config['n_epochs'])

        # Training loop
        best_f1 = 0
        best_model_state = model.state_dict().copy()
        patience_counter = 0

        for epoch in range(config['n_epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

            # Validation
            val_probs, val_labels_pred, _ = evaluate(model, val_loader, device)
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
            idx = original_ids.index(obj_id)
            oof_probs[idx] = prob

        fold_scores.append(best_f1)
        print(f"   Fold {fold+1} best F1: {best_f1:.4f}")

    # 6. OOF evaluation
    print("\n6. Out-of-fold evaluation (on original data)...")

    best_thresh, oof_f1 = find_optimal_threshold(original_labels, oof_probs)
    oof_preds = (oof_probs >= best_thresh).astype(int)

    print(f"   OOF F1: {oof_f1:.4f} @ threshold={best_thresh:.2f}")
    print(f"   Precision: {precision_score(original_labels, oof_preds):.4f}")
    print(f"   Recall: {recall_score(original_labels, oof_preds):.4f}")
    print(f"   Fold F1 scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"   Mean fold F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    # 7. Test predictions
    print("\n7. Generating test predictions...")

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

    # 8. Create submission
    print("\n8. Creating submission...")

    submission = pd.DataFrame({
        'object_id': test_obj_ids,
        'target': test_preds
    })

    submission_path = base_path / 'submissions' / 'submission_v11_rnn_augmented.csv'
    submission.to_csv(submission_path, index=False)
    print(f"   Saved to {submission_path}")
    print(f"   Predictions: {test_preds.sum()} TDEs / {len(test_preds)} total")

    # 9. Save models
    models_path = base_path / 'data/processed/models_v11.pkl'
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
    print(f"\nAugmentation Impact:")
    print(f"  v10 RNN (no aug):   OOF F1 = 0.1035")
    print(f"  v11 RNN (with aug): OOF F1 = {oof_f1:.4f}")
    print(f"  Improvement: {(oof_f1 - 0.1035) / 0.1035 * 100:+.1f}%")
    print(f"\nReference (GBM):")
    print(f"  v8 GBM: OOF F1 = 0.6262, LB = 0.6481")
    print("=" * 60)


if __name__ == "__main__":
    main()
