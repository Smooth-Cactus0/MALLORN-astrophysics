"""
MALLORN v17: Heavy Augmentation (50x) with 1D-CNN

Key changes:
1. 50x augmentation per TDE (instead of 10x)
2. More aggressive augmentation parameters
3. 1D-CNN architecture (better for fixed patterns)
4. MALLORN data only (PLAsTiCC has domain shift)

Expected: 148 TDEs * 50 = 7,400 augmented TDEs (vs 1,776 in v11)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from utils.data_loader import load_all_data
from models.lightcurve_dataset import LightcurveDataset, collate_fn


class HeavyLightcurveAugmenter:
    """
    More aggressive augmentation for severe class imbalance.
    """

    def __init__(self, random_state=42):
        self.rng = np.random.RandomState(random_state)

    def augment_single(self, lc, n_augmentations=50):
        """Generate many augmented versions."""
        augmented = []

        for i in range(n_augmentations):
            aug = lc.copy()

            # 1. Flux scaling (always) - wider range
            scale = self.rng.uniform(0.3, 3.0)
            aug['Flux'] = aug['Flux'] * scale
            aug['Flux_err'] = aug['Flux_err'] * scale

            # 2. Time stretching (90% chance) - wider range
            if self.rng.random() < 0.9:
                stretch = self.rng.uniform(0.6, 1.5)
                t_min = aug['Time (MJD)'].min()
                aug['Time (MJD)'] = t_min + (aug['Time (MJD)'] - t_min) * stretch

            # 3. Noise injection (80% chance) - more noise
            if self.rng.random() < 0.8:
                noise_scale = self.rng.uniform(0.3, 2.0)
                noise = self.rng.normal(0, aug['Flux_err'].values * noise_scale)
                aug['Flux'] = aug['Flux'] + noise

            # 4. Observation dropout (60% chance) - more aggressive
            if self.rng.random() < 0.6:
                dropout = self.rng.uniform(0.1, 0.4)
                n_keep = max(5, int(len(aug) * (1 - dropout)))
                keep_idx = self.rng.choice(len(aug), size=n_keep, replace=False)
                keep_idx = np.sort(keep_idx)
                aug = aug.iloc[keep_idx].reset_index(drop=True)

            # 5. Flux offset (50% chance) - simulate baseline variations
            if self.rng.random() < 0.5:
                offset = self.rng.uniform(-0.5, 0.5) * aug['Flux'].std()
                aug['Flux'] = aug['Flux'] + offset

            # 6. Band-specific noise (40% chance)
            if self.rng.random() < 0.4:
                band_noise = {'u': 1.5, 'g': 1.0, 'r': 0.8, 'i': 0.9, 'z': 1.1, 'y': 1.3}
                for band, scale in band_noise.items():
                    mask = aug['Filter'] == band
                    if mask.any():
                        noise = self.rng.normal(0, aug.loc[mask, 'Flux_err'].values * scale * 0.3)
                        aug.loc[mask, 'Flux'] = aug.loc[mask, 'Flux'] + noise

            augmented.append(aug)

        return augmented


class CNN1DClassifier(nn.Module):
    """
    1D CNN for lightcurve classification.

    Architecture:
    - Multiple conv layers with increasing channels
    - Global average + max pooling
    - Metadata fusion
    - Classification head
    """

    def __init__(
        self,
        in_channels=4,
        hidden_channels=[32, 64, 128],
        kernel_sizes=[7, 5, 3],
        dropout=0.3,
        n_bands=6,
        n_metadata=2
    ):
        super().__init__()

        # Band embedding
        self.band_embedding = nn.Embedding(n_bands, 8)

        # Initial projection (4 features + 8 band embed = 12)
        self.input_proj = nn.Conv1d(in_channels + 8, hidden_channels[0], kernel_size=1)

        # Conv blocks
        self.conv_blocks = nn.ModuleList()
        in_ch = hidden_channels[0]
        for out_ch, ks in zip(hidden_channels, kernel_sizes):
            block = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(out_ch, out_ch, kernel_size=ks, padding=ks//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
            self.conv_blocks.append(block)
            in_ch = out_ch

        # Metadata projection
        self.meta_proj = nn.Sequential(
            nn.Linear(n_metadata, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels[-1] * 2 + 32, 64),  # *2 for avg+max pool
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, features, bands, mask, metadata):
        batch_size, seq_len, _ = features.shape

        # Handle NaN
        features = torch.nan_to_num(features, nan=0.0)
        metadata = torch.nan_to_num(metadata, nan=0.0)

        # Get band embeddings
        band_emb = self.band_embedding(bands)  # (batch, seq_len, 8)

        # Concatenate features with band embeddings
        x = torch.cat([features, band_emb], dim=-1)  # (batch, seq_len, 12)

        # Transpose for conv: (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        # Input projection
        x = self.input_proj(x)

        # Apply conv blocks
        for block in self.conv_blocks:
            x = block(x)

        # Global pooling (average + max)
        avg_pool = x.mean(dim=-1)  # (batch, channels)
        max_pool = x.max(dim=-1)[0]  # (batch, channels)
        pooled = torch.cat([avg_pool, max_pool], dim=-1)

        # Metadata fusion
        meta_emb = self.meta_proj(metadata)
        combined = torch.cat([pooled, meta_emb], dim=-1)

        # Classification
        logits = self.classifier(combined)
        logits_clamped = torch.clamp(logits, -20, 20)
        probs = torch.sigmoid(logits_clamped).squeeze(-1)

        return {'logits': logits, 'probs': probs}


def augment_tdes_heavy(lightcurves, metadata, augmentations_per_tde=50, random_state=42):
    """Generate heavy augmentation for TDEs."""
    augmenter = HeavyLightcurveAugmenter(random_state=random_state)

    tde_ids = metadata[metadata['target'] == 1]['object_id'].tolist()
    print(f"  Found {len(tde_ids)} TDE objects to augment (50x each)")

    grouped = {obj_id: group.copy() for obj_id, group in lightcurves.groupby('object_id')}

    augmented_lcs = []
    augmented_meta = []

    for i, obj_id in enumerate(tde_ids):
        if (i + 1) % 20 == 0:
            print(f"    Augmenting TDE {i+1}/{len(tde_ids)}")

        original_lc = grouped.get(obj_id)
        if original_lc is None or len(original_lc) < 5:
            continue

        original_meta = metadata[metadata['object_id'] == obj_id].iloc[0]

        aug_lcs = augmenter.augment_single(original_lc, n_augmentations=augmentations_per_tde)

        for j, aug_lc in enumerate(aug_lcs):
            new_id = f"{obj_id}_aug{j}"
            aug_lc['object_id'] = new_id
            augmented_lcs.append(aug_lc)

            new_meta = original_meta.copy()
            new_meta['object_id'] = new_id
            augmented_meta.append(new_meta)

    if augmented_lcs:
        aug_lc_df = pd.concat(augmented_lcs, ignore_index=True)
        aug_meta_df = pd.DataFrame(augmented_meta)
        print(f"  Generated {len(aug_meta_df)} augmented TDE samples")
        return aug_lc_df, aug_meta_df

    return pd.DataFrame(), pd.DataFrame()


def find_optimal_threshold(y_true, y_prob):
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        features = batch['features'].to(device)
        bands = batch['bands'].to(device)
        mask = batch['mask'].to(device)
        metadata = batch['metadata'].to(device)
        labels = batch['label'].to(device)

        if torch.isnan(features).any():
            continue

        optimizer.zero_grad()
        output = model(features, bands, mask, metadata)
        logits = output['logits'].squeeze(-1)  # Only squeeze last dim
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        loss = criterion(logits, labels)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, device):
    model.eval()
    all_probs, all_labels, all_ids = [], [], []

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
    print("MALLORN v17: Heavy Augmentation (50x) with 1D-CNN")
    print("=" * 60)

    base_path = Path(__file__).parent.parent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 1. Load data
    print("\n1. Loading data...")
    data = load_all_data()
    train_meta = data['train_meta'].copy()
    train_lc = data['train_lc'].copy()

    print(f"   Original: {len(train_meta)} samples ({(train_meta['target']==1).sum()} TDEs)")

    # 2. Heavy augmentation
    print("\n2. Generating heavy augmentation (50x per TDE)...")
    aug_lc, aug_meta = augment_tdes_heavy(train_lc, train_meta, augmentations_per_tde=50, random_state=42)

    combined_lc = pd.concat([train_lc, aug_lc], ignore_index=True)
    combined_meta = pd.concat([train_meta, aug_meta], ignore_index=True)

    total_tde = (combined_meta['target'] == 1).sum()
    print(f"   After augmentation: {len(combined_meta)} samples ({total_tde} TDEs)")
    print(f"   TDE percentage: {total_tde/len(combined_meta)*100:.1f}%")

    # 3. Configuration
    print("\n3. Model configuration (1D-CNN)...")
    config = {
        'max_length': 300,
        'hidden_channels': [32, 64, 128],
        'kernel_sizes': [7, 5, 3],
        'dropout': 0.3,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'n_epochs': 50,
        'patience': 10,
        'pos_weight': 3.0  # Lower weight since we have more TDEs now
    }

    for k, v in config.items():
        print(f"   {k}: {v}")

    # 4. Training
    print("\n4. Training with 5-fold cross-validation...")

    labels_dict = dict(zip(combined_meta['object_id'], combined_meta['target']))
    original_ids = train_meta['object_id'].tolist()
    original_labels = train_meta['target'].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(original_ids))
    fold_models = []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(original_ids, original_labels)):
        print(f"\n   --- Fold {fold + 1}/5 ---")

        val_ids = [original_ids[i] for i in val_idx]
        val_labels = original_labels[val_idx]

        # Training: original + augmented (excluding val originals)
        original_train_ids = set([original_ids[i] for i in train_idx])
        train_ids = []
        for oid in combined_meta['object_id']:
            base_id = oid.split('_aug')[0] if '_aug' in oid else oid
            if base_id in original_train_ids:
                train_ids.append(oid)

        print(f"   Training: {len(train_ids)} samples")
        print(f"   Validation: {len(val_ids)} samples")

        train_dataset = LightcurveDataset(combined_lc, combined_meta, train_ids, labels_dict, config['max_length'])
        val_dataset = LightcurveDataset(train_lc, train_meta, val_ids, labels_dict, config['max_length'])

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

        model = CNN1DClassifier(
            hidden_channels=config['hidden_channels'],
            kernel_sizes=config['kernel_sizes'],
            dropout=config['dropout']
        ).to(device)

        if fold == 0:
            print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        pos_weight = torch.tensor([config['pos_weight']]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        best_f1 = 0
        best_model_state = model.state_dict().copy()
        patience_counter = 0

        for epoch in range(config['n_epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_probs, _, _ = evaluate(model, val_loader, device)
            thresh, val_f1 = find_optimal_threshold(val_labels, val_probs)
            scheduler.step()

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or patience_counter == 0:
                print(f"   Epoch {epoch+1}: loss={train_loss:.4f}, val_F1={val_f1:.4f} (best={best_f1:.4f})")

            if patience_counter >= config['patience']:
                print(f"   Early stopping at epoch {epoch+1}")
                break

        model.load_state_dict(best_model_state)
        fold_models.append(model.state_dict())

        val_probs, _, val_obj_ids = evaluate(model, val_loader, device)
        for obj_id, prob in zip(val_obj_ids, val_probs):
            idx = original_ids.index(obj_id)
            oof_probs[idx] = prob

        fold_scores.append(best_f1)
        print(f"   Fold {fold+1} best F1: {best_f1:.4f}")

    # 5. OOF evaluation
    print("\n5. Out-of-fold evaluation...")
    best_thresh, oof_f1 = find_optimal_threshold(original_labels, oof_probs)
    oof_preds = (oof_probs >= best_thresh).astype(int)

    print(f"   OOF F1: {oof_f1:.4f} @ threshold={best_thresh:.2f}")
    print(f"   Precision: {precision_score(original_labels, oof_preds):.4f}")
    print(f"   Recall: {recall_score(original_labels, oof_preds):.4f}")
    print(f"   Predicted TDEs: {oof_preds.sum()}")
    print(f"   Mean fold F1: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

    # 6. Test predictions
    print("\n6. Generating test predictions...")
    test_ids = data['test_meta']['object_id'].tolist()
    test_dataset = LightcurveDataset(data['test_lc'], data['test_meta'], test_ids, None, config['max_length'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    test_probs_all = []
    for state_dict in fold_models:
        model = CNN1DClassifier(
            hidden_channels=config['hidden_channels'],
            kernel_sizes=config['kernel_sizes'],
            dropout=config['dropout']
        ).to(device)
        model.load_state_dict(state_dict)
        test_probs, _, test_obj_ids = evaluate(model, test_loader, device)
        test_probs_all.append(test_probs)

    test_probs = np.mean(test_probs_all, axis=0)
    test_preds = (test_probs >= best_thresh).astype(int)

    # 7. Save
    submission = pd.DataFrame({'object_id': test_obj_ids, 'target': test_preds})
    submission_path = base_path / 'submissions' / 'submission_v17_heavy_aug.csv'
    submission.to_csv(submission_path, index=False)
    print(f"   Saved to {submission_path}")
    print(f"   Predictions: {test_preds.sum()} TDEs ({test_preds.sum()/len(test_preds)*100:.1f}%)")

    with open(base_path / 'data/processed/models_v17.pkl', 'wb') as f:
        pickle.dump({'fold_models': fold_models, 'config': config, 'best_thresh': best_thresh,
                     'oof_probs': oof_probs, 'oof_f1': oof_f1}, f)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nDL Comparison:")
    print(f"  v11 LSTM (10x aug):  OOF F1 = 0.1200")
    print(f"  v17 CNN (50x aug):   OOF F1 = {oof_f1:.4f}")
    print(f"  GBM v8 (reference):  OOF F1 = 0.6262")
    print("=" * 60)


if __name__ == "__main__":
    main()
