"""
MALLORN v13: Transformer with TDE Augmentation

Uses the same augmentation strategy as v11 (TDE-only augmentation)
but with a Transformer architecture instead of LSTM.

Key differences from LSTM:
1. Self-attention captures long-range dependencies in lightcurves
2. Parallel processing of all timesteps (faster training)
3. Learnable [CLS] token aggregates sequence information
4. Better at focusing on discriminative patterns

Expected: The attention mechanism should better identify
which observations are most informative for TDE classification.
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
from utils.data_loader import load_all_data
from models.lightcurve_dataset import LightcurveDataset, collate_fn
from models.transformer_classifier import TransformerClassifier, FocalLoss
from features.augmentation import augment_tde_dataset


def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score."""
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


def train_epoch(model, dataloader, optimizer, criterion, device, grad_accum=1):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        features = batch['features'].to(device)
        bands = batch['bands'].to(device)
        mask = batch['mask'].to(device)
        metadata = batch['metadata'].to(device)
        labels = batch['label'].to(device)

        if torch.isnan(features).any() or torch.isnan(metadata).any():
            continue

        output = model(features, bands, mask, metadata)
        loss = criterion(output['logits'], labels)

        if torch.isnan(loss):
            continue

        # Scale loss for gradient accumulation
        loss = loss / grad_accum
        loss.backward()

        if (batch_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        n_batches += 1

    # Handle remaining gradients
    if n_batches % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()

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
    print("MALLORN v13: Transformer with TDE Augmentation")
    print("=" * 60)

    base_path = Path(__file__).parent.parent

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 1. Load data
    print("\n1. Loading data...")
    data = load_all_data()

    train_meta = data['train_meta'].copy()
    train_lc = data['train_lc'].copy()

    original_tde_count = (train_meta['target'] == 1).sum()
    original_non_tde_count = (train_meta['target'] == 0).sum()
    print(f"   Original: {original_tde_count} TDEs, {original_non_tde_count} non-TDEs")

    # 2. Generate augmented TDEs (same as v11)
    print("\n2. Generating augmented TDE samples...")

    aug_lc, aug_meta = augment_tde_dataset(
        train_lc,
        train_meta,
        augmentations_per_tde=10,
        include_mixup=True,
        mixup_per_tde=2,
        random_state=42
    )

    # Combine original + augmented
    combined_lc = pd.concat([train_lc, aug_lc], ignore_index=True)
    combined_meta = pd.concat([train_meta, aug_meta], ignore_index=True)

    augmented_tde_count = (combined_meta['target'] == 1).sum()
    total_count = len(combined_meta)

    print(f"\n   After augmentation:")
    print(f"   - TDEs: {original_tde_count} -> {augmented_tde_count} ({augmented_tde_count/original_tde_count:.1f}x)")
    print(f"   - Total samples: {total_count}")
    print(f"   - Class balance: {augmented_tde_count/total_count*100:.1f}% TDE")

    # 3. Model configuration
    print("\n3. Model configuration (Transformer)...")

    config = {
        'max_length': 300,
        'd_model': 64,        # Embedding dimension
        'n_heads': 4,         # Attention heads
        'n_layers': 3,        # Transformer layers
        'd_ff': 128,          # Feedforward dimension
        'dropout': 0.2,
        'batch_size': 32,     # Smaller batch for memory
        'grad_accum': 2,      # Effective batch size = 64
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'n_epochs': 40,
        'patience': 7,
        'focal_alpha': 0.25,  # Focal loss alpha
        'focal_gamma': 2.0,   # Focal loss gamma
        'pos_weight': 10.0    # Positive class weight
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
        original_train_ids = [original_ids[i] for i in train_idx]

        # Get augmented IDs that correspond to training TDEs
        train_tde_ids = set(train_meta[train_meta['object_id'].isin(original_train_ids) &
                                        (train_meta['target'] == 1)]['object_id'])

        # Include augmented versions of training TDEs only
        aug_train_ids = [oid for oid in aug_meta['object_id']
                         if any(oid.startswith(tde_id + '_') for tde_id in train_tde_ids)]

        train_ids = original_train_ids + aug_train_ids

        print(f"   Training: {len(train_ids)} samples ({len(original_train_ids)} original + {len(aug_train_ids)} augmented)")
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
        model = TransformerClassifier(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout']
        ).to(device)

        # Count parameters
        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"   Model parameters: {n_params:,}")

        # Loss and optimizer
        criterion = FocalLoss(
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma'],
            pos_weight=config['pos_weight']
        )
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
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device,
                grad_accum=config['grad_accum']
            )

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
    print(f"   Predicted TDEs: {oof_preds.sum()}")
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
        model = TransformerClassifier(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            d_ff=config['d_ff'],
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

    submission_path = base_path / 'submissions' / 'submission_v13_transformer.csv'
    submission.to_csv(submission_path, index=False)
    print(f"   Saved to {submission_path}")
    print(f"   Predictions: {test_preds.sum()} TDEs / {len(test_preds)} total ({test_preds.sum()/len(test_preds)*100:.1f}%)")

    # 9. Save models
    models_path = base_path / 'data/processed/models_v13.pkl'
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
    print(f"\nDeep Learning Comparison:")
    print(f"  v10 LSTM (no aug):        OOF F1 = 0.1035 (predicted 0 TDEs)")
    print(f"  v11 LSTM (TDE aug):       OOF F1 = 0.1200 (predicted 3684 TDEs)")
    print(f"  v13 Transformer (TDE aug): OOF F1 = {oof_f1:.4f} (predicted {test_preds.sum()} TDEs)")
    print(f"\nReference (GBM):")
    print(f"  v8 GBM: OOF F1 = 0.6262, LB = 0.6481")
    print("=" * 60)


if __name__ == "__main__":
    main()
