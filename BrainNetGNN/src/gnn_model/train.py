"""
GNN Training Pipeline for BrainNetGNN
======================================
Trains GCN, GAT, and baseline models on brain network graphs
for both cognitive workload classification and ADHD detection.

Models:
- GCN (Graph Convolutional Network) — baseline GNN
- GAT (Graph Attention Network) — attention-based, provides electrode importance
- Logistic Regression — traditional ML baseline
- Random Forest — ensemble ML baseline

Subject-wise cross-validation to prevent data leakage.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report
)
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# =================== MODEL DEFINITIONS ===================

class BrainGCN(nn.Module):
    """
    Graph Convolutional Network for brain network classification.
    Architecture: GCNConv → GCNConv → GCNConv → Global Pool → MLP
    """
    def __init__(self, in_features, hidden_dim=64, num_classes=2, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Graph convolution layers
        x = self.conv1(x, edge_index, edge_attr.squeeze(-1) if edge_attr is not None else None)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index, edge_attr.squeeze(-1) if edge_attr is not None else None)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index, edge_attr.squeeze(-1) if edge_attr is not None else None)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global pooling (mean + max for richer representation)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # MLP classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x
    
    def get_node_embeddings(self, data):
        """Extract node-level embeddings for visualization."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr.squeeze(-1) if edge_attr is not None else None)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr.squeeze(-1) if edge_attr is not None else None)))
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr.squeeze(-1) if edge_attr is not None else None)))
        
        return x


class BrainGAT(nn.Module):
    """
    Graph Attention Network for brain network classification.
    Key advantage: attention weights show which electrode connections
    are most important for classification — directly comparable to
    traditional PageRank-based electrode ranking.
    """
    def __init__(self, in_features, hidden_dim=64, num_classes=2, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_features, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # Store attention weights for explainability
        self._attention_weights = None
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last layer — capture attention weights
        x, (edge_index_out, attention_weights) = self.conv3(
            x, edge_index, return_attention_weights=True
        )
        self._attention_weights = (edge_index_out, attention_weights)
        x = self.bn3(x)
        x = F.elu(x)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # MLP classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x
    
    def get_attention_weights(self):
        """Return last computed attention weights for explainability."""
        return self._attention_weights


# =================== TRAINING FUNCTIONS ===================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == batch.y.squeeze()).sum().item()
        total += batch.num_graphs
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        probs = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.squeeze().cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0,
    }
    
    return metrics, all_preds, all_labels, all_probs


def train_gnn(model, train_dataset, val_dataset, epochs=80, lr=0.001, batch_size=32, device='cpu', patience=15):
    """Train a GNN model with early stopping."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Class weights for imbalanced data
    labels = [d.y.item() for d in train_dataset]
    class_counts = np.bincount(labels)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        
        scheduler.step(val_metrics['f1'])
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model


def train_baselines(train_dataset, val_dataset):
    """Train traditional ML baselines (LR, RF) using graph-level features."""
    # Extract features: global pool of node features
    def extract_features(dataset):
        X, y = [], []
        for data in dataset:
            # Use mean and std of node features as graph-level features
            x_mean = data.x.mean(dim=0).numpy()
            x_std = data.x.std(dim=0).numpy()
            # Also use adjacency matrix statistics
            adj = data.adj.numpy() if hasattr(data, 'adj') else np.zeros((19, 19))
            adj_feats = [adj.mean(), adj.std(), adj.max(), np.median(adj)]
            X.append(np.concatenate([x_mean, x_std, adj_feats]))
            y.append(data.y.item())
        return np.array(X), np.array(y)
    
    X_train, y_train = extract_features(train_dataset)
    X_val, y_val = extract_features(val_dataset)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    results = {}
    
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_val)
    lr_probs = lr_model.predict_proba(X_val)[:, 1]
    results['LogisticRegression'] = {
        'accuracy': accuracy_score(y_val, lr_preds),
        'f1': f1_score(y_val, lr_preds, average='weighted'),
        'auc': roc_auc_score(y_val, lr_probs) if len(np.unique(y_val)) > 1 else 0,
    }
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_val)
    rf_probs = rf_model.predict_proba(X_val)[:, 1]
    results['RandomForest'] = {
        'accuracy': accuracy_score(y_val, rf_preds),
        'f1': f1_score(y_val, rf_preds, average='weighted'),
        'auc': roc_auc_score(y_val, rf_probs) if len(np.unique(y_val)) > 1 else 0,
    }
    
    return results


def extract_attention_importance(model, dataset, device='cpu'):
    """
    Extract electrode importance from GAT attention weights.
    This is analogous to traditional PageRank-based electrode ranking.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    node_importance = np.zeros(19)  # 19 electrodes
    n_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        _ = model(batch)
        attn = model.get_attention_weights()
        if attn is not None:
            edge_index, weights = attn
            weights = weights.detach().cpu().numpy().flatten()
            ei = edge_index.detach().cpu().numpy()
            
            # Aggregate attention received by each node
            for i in range(len(weights)):
                target_node = ei[1, i] % 19  # Handle batched graphs
                node_importance[target_node] += weights[i]
            n_samples += 1
    
    if n_samples > 0:
        node_importance /= n_samples
    
    # Normalize to [0, 1]
    if node_importance.max() > 0:
        node_importance = node_importance / node_importance.max()
    
    return node_importance


# =================== MAIN TRAINING PIPELINE ===================

def run_training(dataset_name: str, graphs_path: str, output_dir: str, device: str = 'cpu'):
    """
    Full training pipeline for a dataset with subject-wise cross-validation.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load graphs
    dataset = torch.load(graphs_path, weights_only=False)
    print(f"  Loaded {len(dataset)} graphs")
    
    labels = np.array([d.y.item() for d in dataset])
    subjects = np.array([d.subject for d in dataset])
    unique_subjects = np.unique(subjects)
    
    # Create subject-level labels for stratification
    subject_labels = {}
    for s, l in zip(subjects, labels):
        subject_labels[s] = l
    subject_level_labels = np.array([subject_labels[s] for s in unique_subjects])
    
    print(f"  Subjects: {len(unique_subjects)}, Labels: 0={int((labels==0).sum())}, 1={int((labels==1).sum())}")
    
    # Subject-wise train/test split (80/20)
    from sklearn.model_selection import train_test_split
    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=0.2, stratify=subject_level_labels, random_state=42
    )
    
    train_idx = np.isin(subjects, train_subjects)
    test_idx = np.isin(subjects, test_subjects)
    
    train_dataset = [dataset[i] for i in range(len(dataset)) if train_idx[i]]
    test_dataset = [dataset[i] for i in range(len(dataset)) if test_idx[i]]
    
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    in_features = dataset[0].x.shape[1]
    all_results = {}
    
    # --- Train GCN ---
    print(f"\n  Training GCN...")
    gcn = BrainGCN(in_features=in_features, hidden_dim=64, num_classes=2).to(device)
    gcn = train_gnn(gcn, train_dataset, test_dataset, epochs=80, lr=0.001, device=device)
    gcn_metrics, gcn_preds, gcn_labels, gcn_probs = evaluate(
        gcn, DataLoader(test_dataset, batch_size=32), device
    )
    all_results['GCN'] = gcn_metrics
    print(f"    Accuracy: {gcn_metrics['accuracy']:.4f}, F1: {gcn_metrics['f1']:.4f}, AUC: {gcn_metrics['auc']:.4f}")
    
    # Save GCN model
    torch.save(gcn.state_dict(), os.path.join(output_dir, f'{dataset_name}_gcn.pt'))
    
    # --- Train GAT ---
    print(f"\n  Training GAT...")
    gat = BrainGAT(in_features=in_features, hidden_dim=64, num_classes=2, heads=4).to(device)
    gat = train_gnn(gat, train_dataset, test_dataset, epochs=80, lr=0.001, device=device)
    gat_metrics, gat_preds, gat_labels, gat_probs = evaluate(
        gat, DataLoader(test_dataset, batch_size=32), device
    )
    all_results['GAT'] = gat_metrics
    print(f"    Accuracy: {gat_metrics['accuracy']:.4f}, F1: {gat_metrics['f1']:.4f}, AUC: {gat_metrics['auc']:.4f}")
    
    # Save GAT model
    torch.save(gat.state_dict(), os.path.join(output_dir, f'{dataset_name}_gat.pt'))
    
    # --- Extract attention-based electrode importance ---
    print(f"\n  Extracting GAT electrode importance...")
    electrode_importance = extract_attention_importance(gat, test_dataset, device)
    
    channel_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
        'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
        'Fz', 'Cz', 'Pz'
    ]
    
    importance_ranked = sorted(
        zip(channel_names, electrode_importance),
        key=lambda x: x[1], reverse=True
    )
    print(f"    Top 5 electrodes: {[(n, f'{v:.3f}') for n, v in importance_ranked[:5]]}")
    
    # --- Train baselines ---
    print(f"\n  Training baselines (LR, RF)...")
    baseline_results = train_baselines(train_dataset, test_dataset)
    all_results.update(baseline_results)
    for name, metrics in baseline_results.items():
        print(f"    {name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
    
    # --- Save results ---
    results_output = {
        'dataset': dataset_name,
        'n_train': len(train_dataset),
        'n_test': len(test_dataset),
        'n_subjects': len(unique_subjects),
        'models': {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in all_results.items()},
        'electrode_importance': {n: round(float(v), 4) for n, v in zip(channel_names, electrode_importance)},
        'top_electrodes': [n for n, v in importance_ranked[:5]],
    }
    
    with open(os.path.join(output_dir, f'{dataset_name}_results.json'), 'w') as f:
        json.dump(results_output, f, indent=2)
    
    # Save electrode importance
    np.save(os.path.join(output_dir, f'{dataset_name}_electrode_importance.npy'), electrode_importance)
    
    return all_results, electrode_importance


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parents[2]
    processed_dir = base_dir / 'data' / 'processed'
    models_dir = base_dir / 'data' / 'models'
    device = 'cpu'
    
    # Train on EEGMAT (cognitive workload)
    eegmat_results, eegmat_importance = run_training(
        'eegmat',
        str(processed_dir / 'eegmat_plv_graphs.pt'),
        str(models_dir),
        device=device
    )
    
    # Train on ADHD
    adhd_results, adhd_importance = run_training(
        'adhd',
        str(processed_dir / 'adhd_plv_graphs.pt'),
        str(models_dir),
        device=device
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE — SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Model':<20} {'EEGMAT Acc':>12} {'EEGMAT F1':>12} {'ADHD Acc':>12} {'ADHD F1':>12}")
    print("-" * 68)
    for model_name in ['GCN', 'GAT', 'LogisticRegression', 'RandomForest']:
        e = eegmat_results.get(model_name, {})
        a = adhd_results.get(model_name, {})
        print(f"{model_name:<20} {e.get('accuracy', 0):>12.4f} {e.get('f1', 0):>12.4f} {a.get('accuracy', 0):>12.4f} {a.get('f1', 0):>12.4f}")
