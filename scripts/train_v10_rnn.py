"""
MALLORN v10: LSTM with Attention

Trains a bidirectional LSTM with attention on raw lightcurves.
Designed to complement the GBM ensemble by learning temporal patterns
directly from sequences rather than hand-crafted features.

Architecture:
- Band embeddings for multi-band data
- Bidirectional LSTM for temporal patterns
- Self-attention for focusing on key moments
- Metadata fusion (redshift, EBV)
- Focal loss for class imbalance
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
from models.lstm_classifier import LSTMClassifier, FocalLoss, WeightedBCELoss


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

        # Check for NaN in inputs
        if torch.isnan(features).any() or torch.isnan(metadata).any():
            continue

        optimizer.zero_grad()

        output = model(features, bands, mask, metadata)
        loss = criterion(output['logits'], labels)

        # Skip if loss is NaN
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
    print("MALLORN v10: LSTM with Attention")
    print("=" * 60)

    base_path = Path(__file__).parent.parent

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 1. Load data
    print("\n1. Loading data...")
    data = load_all_data()

    train_ids = data['train_meta']['object_id'].tolist()
    test_ids = data['test_meta']['object_id'].tolist()

    # Create labels dict
    labels_dict = dict(zip(
        data['train_meta']['object_id'],
        data['train_meta']['target']
    ))
    y = np.array([labels_dict[obj_id] for obj_id in train_ids])

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"   Training: {len(train_ids)} objects ({n_pos} TDE, {n_neg} non-TDE)")
    print(f"   Test: {len(test_ids)} objects")

    # 2. Model configuration
    print("\n2. Model configuration...")

    # Hyperparameters (tuned for small GPU)
    config = {
        'max_length': 300,        # Max sequence length
        'hidden_dim': 64,         # LSTM hidden dimension
        'n_layers': 2,            # LSTM layers
        'dropout': 0.3,           # Dropout rate
        'batch_size': 32,         # Batch size
        'lr': 5e-4,               # Learning rate (reduced for stability)
        'weight_decay': 1e-4,     # L2 regularization
        'n_epochs': 40,           # Max epochs
        'patience': 7,            # Early stopping patience
        'pos_weight': 20.0        # Class weight for TDEs (minority class)
    }

    for k, v in config.items():
        print(f"   {k}: {v}")

    # 3. K-Fold training
    print("\n3. Training with 5-fold cross-validation...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(train_ids))
    fold_models = []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_ids, y)):
        print(f"\n   --- Fold {fold + 1}/5 ---")

        # Split data
        fold_train_ids = [train_ids[i] for i in train_idx]
        fold_val_ids = [train_ids[i] for i in val_idx]

        # Create datasets
        train_dataset = LightcurveDataset(
            lightcurves=data['train_lc'],
            metadata=data['train_meta'],
            object_ids=fold_train_ids,
            labels=labels_dict,
            max_length=config['max_length']
        )

        val_dataset = LightcurveDataset(
            lightcurves=data['train_lc'],
            metadata=data['train_meta'],
            object_ids=fold_val_ids,
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

        # Loss and optimizer (weighted BCE is more stable than focal loss)
        criterion = WeightedBCELoss(pos_weight=config['pos_weight'])
        optimizer = AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=config['n_epochs'])

        # Training loop
        best_f1 = 0
        best_model_state = model.state_dict().copy()  # Initialize with current state
        patience_counter = 0

        for epoch in range(config['n_epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

            # Validation
            val_probs, val_labels, _ = evaluate(model, val_loader, device)
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
        val_probs, _, val_ids = evaluate(model, val_loader, device)
        for obj_id, prob in zip(val_ids, val_probs):
            idx = train_ids.index(obj_id)
            oof_probs[idx] = prob

        fold_scores.append(best_f1)
        print(f"   Fold {fold+1} best F1: {best_f1:.4f}")

    # 4. OOF evaluation
    print("\n4. Out-of-fold evaluation...")

    best_thresh, oof_f1 = find_optimal_threshold(y, oof_probs)
    oof_preds = (oof_probs >= best_thresh).astype(int)

    print(f"   OOF F1: {oof_f1:.4f} @ threshold={best_thresh:.2f}")
    print(f"   Precision: {precision_score(y, oof_preds):.4f}")
    print(f"   Recall: {recall_score(y, oof_preds):.4f}")
    print(f"   Fold F1 scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"   Mean fold F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    # 5. Test predictions
    print("\n5. Generating test predictions...")

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

    # 6. Create RNN-only submission
    print("\n6. Creating RNN-only submission...")

    submission_rnn = pd.DataFrame({
        'object_id': test_obj_ids,
        'target': test_preds
    })

    submission_path = base_path / 'submissions' / 'submission_v10_rnn_only.csv'
    submission_rnn.to_csv(submission_path, index=False)
    print(f"   Saved to {submission_path}")
    print(f"   Predictions: {test_preds.sum()} TDEs / {len(test_preds)} total")

    # 7. Ensemble with GBM (v8)
    print("\n7. Ensembling with GBM (v8)...")

    # Load v8 GBM predictions
    v8_submission = pd.read_csv(base_path / 'submissions' / 'submission_v8_tuned.csv')
    v8_models = pickle.load(open(base_path / 'data/processed/models_v8.pkl', 'rb'))

    # Get v8 test probabilities
    # We need to regenerate them from the saved models
    cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
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
    test_tde = tde_cached['test']
    tde_cols = [c for c in test_tde.columns if c != 'object_id']

    test_combined = test_features[['object_id'] + selected_120].merge(
        test_tde, on='object_id', how='left'
    )

    all_feature_cols = selected_120 + tde_cols
    X_test_gbm = test_combined[all_feature_cols].values
    X_test_gbm = np.nan_to_num(X_test_gbm, nan=0, posinf=0, neginf=0)

    # Get GBM probabilities
    xgb_probs = np.mean([m.predict_proba(X_test_gbm)[:, 1] for m in v8_models['xgb_models']], axis=0)
    lgb_probs = np.mean([m.predict_proba(X_test_gbm)[:, 1] for m in v8_models['lgb_models']], axis=0)
    cat_probs = np.mean([m.predict_proba(X_test_gbm)[:, 1] for m in v8_models['cat_models']], axis=0)

    w = v8_models['best_weights']
    gbm_probs = w[0] * xgb_probs + w[1] * lgb_probs + w[2] * cat_probs

    # Align test IDs
    gbm_test_ids = test_combined['object_id'].tolist()

    # Create lookup
    rnn_prob_dict = dict(zip(test_obj_ids, test_probs))
    rnn_probs_aligned = np.array([rnn_prob_dict.get(oid, 0.5) for oid in gbm_test_ids])

    # Try different ensemble weights
    print("\n   Testing ensemble weights...")
    best_ensemble_f1 = 0
    best_ensemble_weight = 0.5

    # We don't have test labels, so use OOF correlation as proxy
    # For now, just try a few weights
    for rnn_weight in [0.2, 0.3, 0.4, 0.5]:
        gbm_weight = 1 - rnn_weight
        # Just track the weight, will evaluate on LB
        print(f"   Weight: GBM={gbm_weight:.1f}, RNN={rnn_weight:.1f}")

    # Use moderate weight for RNN (it's new and unproven)
    rnn_weight = 0.3
    gbm_weight = 0.7

    ensemble_probs = gbm_weight * gbm_probs + rnn_weight * rnn_probs_aligned

    # Use v8 threshold as baseline (could optimize further)
    ensemble_thresh = v8_models['best_thresh']
    ensemble_preds = (ensemble_probs >= ensemble_thresh).astype(int)

    # 8. Create ensemble submission
    print("\n8. Creating ensemble submission...")

    submission_ensemble = pd.DataFrame({
        'object_id': gbm_test_ids,
        'target': ensemble_preds
    })

    submission_path = base_path / 'submissions' / 'submission_v10_ensemble.csv'
    submission_ensemble.to_csv(submission_path, index=False)
    print(f"   Saved to {submission_path}")
    print(f"   Predictions: {ensemble_preds.sum()} TDEs / {len(ensemble_preds)} total")
    print(f"   Ensemble weights: GBM={gbm_weight}, RNN={rnn_weight}")

    # 9. Save everything
    print("\n9. Saving models...")

    models_path = base_path / 'data/processed/models_v10.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump({
            'fold_models': fold_models,
            'config': config,
            'best_thresh': best_thresh,
            'oof_probs': oof_probs,
            'oof_f1': oof_f1
        }, f)
    print(f"   Saved to {models_path}")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nRNN Performance:")
    print(f"  OOF F1: {oof_f1:.4f}")
    print(f"  Mean fold F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"\nSubmissions created:")
    print(f"  1. submission_v10_rnn_only.csv (RNN alone)")
    print(f"  2. submission_v10_ensemble.csv (70% GBM + 30% RNN)")
    print(f"\nComparison:")
    print(f"  v8 GBM:    OOF=0.6262, LB=0.6481")
    print(f"  v10 RNN:   OOF={oof_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
