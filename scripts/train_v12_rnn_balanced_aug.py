"""
MALLORN v12: LSTM with Balanced Augmentation

Augments ALL samples (TDE and non-TDE) to increase dataset size
while preserving the original class distribution (~5% TDE).

This gives the LSTM more training data without creating a distribution
shift between training and test data.

Expected: 3,043 samples -> ~18,000 samples (6x) with same 5% TDE rate
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
from features.augmentation import augment_all_samples


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
    print("MALLORN v12: LSTM with Balanced Augmentation")
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
    original_tde_pct = original_tde_count / len(train_meta) * 100
    print(f"   Original: {len(train_meta)} samples ({original_tde_count} TDE = {original_tde_pct:.1f}%)")

    # 2. Generate augmented samples for ALL objects
    print("\n2. Generating balanced augmentation (ALL samples)...")

    aug_lc, aug_meta = augment_all_samples(
        train_lc,
        train_meta,
        augmentations_per_sample=5,  # 5x augmentation for all
        random_state=42
    )

    # Combine original + augmented
    combined_lc = pd.concat([train_lc, aug_lc], ignore_index=True)
    combined_meta = pd.concat([train_meta, aug_meta], ignore_index=True)

    # Verify distribution
    total_tde = (combined_meta['target'] == 1).sum()
    total_samples = len(combined_meta)
    combined_tde_pct = total_tde / total_samples * 100

    print(f"\n   After augmentation:")
    print(f"   - Total samples: {len(train_meta)} -> {total_samples} ({total_samples/len(train_meta):.1f}x)")
    print(f"   - TDE count: {original_tde_count} -> {total_tde}")
    print(f"   - TDE percentage: {original_tde_pct:.1f}% -> {combined_tde_pct:.1f}% (preserved!)")

    # 3. Model configuration
    print("\n3. Model configuration...")

    config = {
        'max_length': 300,
        'hidden_dim': 64,
        'n_layers': 2,
        'dropout': 0.3,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'n_epochs': 30,
        'patience': 5,
        'pos_weight': 20.0  # Back to original weighting (distribution preserved)
    }

    for k, v in config.items():
        print(f"   {k}: {v}")

    # 4. Prepare data structures
    print("\n4. Preparing training data...")

    # Create labels dict for all samples
    labels_dict = dict(zip(combined_meta['object_id'], combined_meta['target']))

    # For validation, only use ORIGINAL samples
    original_ids = train_meta['object_id'].tolist()
    original_labels = train_meta['target'].values

    # 5. K-Fold training
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

        # Training: original train + their augmented versions
        original_train_ids = set([original_ids[i] for i in train_idx])

        # Include augmented versions of training samples only
        train_ids = []
        for obj_id in combined_meta['object_id']:
            # Check if this is an original training sample or augmentation of one
            base_id = obj_id.split('_aug')[0] if '_aug' in obj_id else obj_id
            if base_id in original_train_ids:
                train_ids.append(obj_id)

        n_original_train = sum(1 for oid in train_ids if '_aug' not in oid)
        n_augmented_train = len(train_ids) - n_original_train

        print(f"   Training: {len(train_ids)} samples ({n_original_train} original + {n_augmented_train} augmented)")
        print(f"   Validation: {len(val_ids)} samples (original only)")

        # Create datasets
        train_dataset = LightcurveDataset(
            lightcurves=combined_lc,
            metadata=combined_meta,
            object_ids=train_ids,
            labels=labels_dict,
            max_length=config['max_length']
        )

        val_dataset = LightcurveDataset(
            lightcurves=train_lc,
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
    print(f"   Mean fold F1: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

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

    submission_path = base_path / 'submissions' / 'submission_v12_rnn_balanced.csv'
    submission.to_csv(submission_path, index=False)
    print(f"   Saved to {submission_path}")
    print(f"   Predictions: {test_preds.sum()} TDEs / {len(test_preds)} total ({test_preds.sum()/len(test_preds)*100:.1f}%)")

    # 9. Save models
    models_path = base_path / 'data/processed/models_v12.pkl'
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
    print(f"\nRNN Version Comparison:")
    print(f"  v10 (no aug):         OOF F1 = 0.1035 (predicted 0 TDEs)")
    print(f"  v11 (TDE-only aug):   OOF F1 = 0.1200 (predicted 3684 TDEs - too many!)")
    print(f"  v12 (balanced aug):   OOF F1 = {oof_f1:.4f} (predicted {test_preds.sum()} TDEs)")
    print(f"\nReference (GBM):")
    print(f"  v8 GBM: OOF F1 = 0.6262, LB = 0.6481 (predicted 434 TDEs)")
    print("=" * 60)


if __name__ == "__main__":
    main()
