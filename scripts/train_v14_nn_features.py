"""
MALLORN v14: Neural Network on Engineered Features

Instead of raw lightcurves, this uses the same engineered features
as the GBM models. This lets the neural network benefit from the
domain knowledge encoded in the features.

Hypothesis: The GBM's advantage comes from its features, not the
algorithm. If true, a neural network on those features should
perform competitively.

This also serves as a potential ensemble member that captures
non-linear interactions the GBM might miss.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class FeatureDataset(Dataset):
    """Simple dataset for tabular features."""

    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron for tabular classification.

    Architecture designed for the feature dimensionality (~145 features):
    - BatchNorm for input normalization
    - Residual connections in hidden layers
    - Dropout for regularization
    - Skip connections to preserve original features
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64, 32],
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_dim = input_dim

        # Input projection with batch norm
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.hidden_layers.append(layer)

        # Skip connection from input
        self.skip_proj = nn.Linear(input_dim, hidden_dims[-1])

        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1]),  # Concat with skip
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], 1)
        )

    def forward(self, x):
        # Input normalization
        x_norm = self.input_bn(x)

        # Input projection
        h = torch.relu(self.input_proj(x_norm))

        # Hidden layers
        for layer in self.hidden_layers:
            h = layer(h)

        # Skip connection
        skip = self.skip_proj(x_norm)

        # Combine and output
        combined = torch.cat([h, skip], dim=-1)
        logits = self.output(combined)

        return logits


def find_optimal_threshold(y_true, y_prob):
    """Find threshold that maximizes F1 score."""
    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


def main():
    print("=" * 60)
    print("MALLORN v14: Neural Network on Engineered Features")
    print("=" * 60)

    base_path = Path(__file__).parent.parent

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 1. Load pre-computed features (same as GBM v8)
    print("\n1. Loading engineered features...")
    from utils.data_loader import load_all_data

    data = load_all_data()

    # Load v4 features (base features)
    cached = pd.read_pickle(base_path / 'data/processed/features_v4_cache.pkl')
    train_features = cached['train_features']
    test_features = cached['test_features']

    # Load feature selection results
    selection = pd.read_pickle(base_path / 'data/processed/selected_features.pkl')
    importance_df = selection['importance_df']
    high_corr_df = selection['high_corr_df']

    # Get top 120 non-correlated features (same as v8)
    corr_to_drop = set()
    for _, row in high_corr_df.iterrows():
        if row['feature_1'] not in corr_to_drop:
            corr_to_drop.add(row['feature_2'])
    clean_features = importance_df[~importance_df['feature'].isin(corr_to_drop)]
    selected_120 = clean_features.head(120)['feature'].tolist()

    # Load TDE physics features
    tde_cached = pd.read_pickle(base_path / 'data/processed/tde_physics_cache.pkl')
    train_tde = tde_cached['train']
    test_tde = tde_cached['test']
    tde_cols = [c for c in train_tde.columns if c != 'object_id']

    # Combine features
    train_combined = train_features[['object_id'] + selected_120].merge(
        train_tde, on='object_id', how='left'
    )
    test_combined = test_features[['object_id'] + selected_120].merge(
        test_tde, on='object_id', how='left'
    )
    train_combined = train_combined.merge(
        data['train_meta'][['object_id', 'target']], on='object_id'
    )

    all_feature_cols = selected_120 + tde_cols
    print(f"   Train samples: {len(train_combined)}")
    print(f"   Test samples: {len(test_combined)}")
    print(f"   Number of features: {len(all_feature_cols)} (120 base + {len(tde_cols)} TDE)")

    X_train = train_combined[all_feature_cols].values
    y_train = train_combined['target'].values
    X_test = test_combined[all_feature_cols].values
    test_ids = test_combined['object_id'].values

    # Handle missing values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"   TDEs in training: {y_train.sum()} ({y_train.mean()*100:.1f}%)")

    # 2. Configuration
    print("\n2. Model configuration...")

    config = {
        'hidden_dims': [128, 64, 32],
        'dropout': 0.3,
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'n_epochs': 100,
        'patience': 15,
        'pos_weight': 19.0  # Roughly the class imbalance ratio
    }

    for k, v in config.items():
        print(f"   {k}: {v}")

    # 3. K-Fold training
    print("\n3. Training with 5-fold cross-validation...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(X_train))
    fold_models = []
    fold_scalers = []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n   --- Fold {fold + 1}/5 ---")

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        fold_scalers.append(scaler)

        # Create datasets
        train_dataset = FeatureDataset(X_tr_scaled, y_tr)
        val_dataset = FeatureDataset(X_val_scaled, y_val)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        # Create model
        model = MLPClassifier(
            input_dim=X_tr.shape[1],
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout']
        ).to(device)

        if fold == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"   Model parameters: {n_params:,}")

        # Loss and optimizer
        pos_weight = torch.tensor([config['pos_weight']]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = CosineAnnealingLR(optimizer, T_max=config['n_epochs'])

        # Training loop
        best_f1 = 0
        best_model_state = model.state_dict().copy()
        patience_counter = 0

        for epoch in range(config['n_epochs']):
            # Train
            model.train()
            total_loss = 0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = model(features).squeeze()
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            # Validate
            model.eval()
            val_probs = []
            with torch.no_grad():
                for features, _ in val_loader:
                    features = features.to(device)
                    logits = model(features).squeeze()
                    probs = torch.sigmoid(logits)
                    val_probs.extend(probs.cpu().numpy())

            val_probs = np.array(val_probs)
            thresh, val_f1 = find_optimal_threshold(y_val, val_probs)

            scheduler.step()

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or patience_counter == 0:
                print(f"   Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, val_F1={val_f1:.4f} (best={best_f1:.4f})")

            if patience_counter >= config['patience']:
                print(f"   Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(best_model_state)
        fold_models.append(model.state_dict())

        # Get final validation predictions
        model.eval()
        val_probs = []
        with torch.no_grad():
            for features, _ in val_loader:
                features = features.to(device)
                logits = model(features).squeeze()
                probs = torch.sigmoid(logits)
                val_probs.extend(probs.cpu().numpy())

        oof_probs[val_idx] = val_probs
        fold_scores.append(best_f1)
        print(f"   Fold {fold+1} best F1: {best_f1:.4f}")

    # 4. OOF evaluation
    print("\n4. Out-of-fold evaluation...")

    best_thresh, oof_f1 = find_optimal_threshold(y_train, oof_probs)
    oof_preds = (oof_probs >= best_thresh).astype(int)

    print(f"   OOF F1: {oof_f1:.4f} @ threshold={best_thresh:.2f}")
    print(f"   Precision: {precision_score(y_train, oof_preds):.4f}")
    print(f"   Recall: {recall_score(y_train, oof_preds):.4f}")
    print(f"   Predicted TDEs: {oof_preds.sum()}")
    print(f"   Fold F1 scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"   Mean fold F1: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")

    # 5. Test predictions
    print("\n5. Generating test predictions...")

    test_probs_all = []

    for fold_idx, (state_dict, scaler) in enumerate(zip(fold_models, fold_scalers)):
        model = MLPClassifier(
            input_dim=X_train.shape[1],
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout']
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        X_test_scaled = scaler.transform(X_test)
        test_dataset = FeatureDataset(X_test_scaled)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

        test_probs = []
        with torch.no_grad():
            for features in test_loader:
                features = features.to(device)
                logits = model(features).squeeze()
                probs = torch.sigmoid(logits)
                test_probs.extend(probs.cpu().numpy())

        test_probs_all.append(test_probs)

    test_probs = np.mean(test_probs_all, axis=0)
    test_preds = (test_probs >= best_thresh).astype(int)

    # 6. Create submission
    print("\n6. Creating submission...")

    submission = pd.DataFrame({
        'object_id': test_ids,
        'target': test_preds
    })

    submission_path = base_path / 'submissions' / 'submission_v14_nn_features.csv'
    submission.to_csv(submission_path, index=False)
    print(f"   Saved to {submission_path}")
    print(f"   Predictions: {test_preds.sum()} TDEs / {len(test_preds)} total ({test_preds.sum()/len(test_preds)*100:.1f}%)")

    # 7. Save models
    models_path = base_path / 'data/processed/models_v14.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump({
            'fold_models': fold_models,
            'fold_scalers': fold_scalers,
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
    print(f"\nModel Comparison (same features):")
    print(f"  v8 GBM (XGB+LGB+CB):     OOF F1 = 0.6262, LB = 0.6481")
    print(f"  v14 Neural Network:      OOF F1 = {oof_f1:.4f}")
    print(f"\nDeep Learning (raw sequences):")
    print(f"  v11 LSTM:       OOF F1 = 0.1200")
    print(f"  v13 Transformer: OOF F1 = 0.1097")
    print("=" * 60)


if __name__ == "__main__":
    main()
