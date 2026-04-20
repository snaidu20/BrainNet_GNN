"""
Brain Graph Construction Module for BrainNetGNN
================================================
Converts preprocessed EEG epochs into graph representations using
multiple connectivity metrics (PLV, PLI, Coherence, Correlation).

Each EEG epoch becomes a graph where:
- Nodes = EEG electrodes (19 nodes in standard 10-20 system)
- Edges = functional connectivity between electrode pairs
- Node features = frequency band power + graph-theoretic metrics
- Edge weights = connectivity strength

References:
- Prior work: Mutual information, transfer entropy for FBN construction
- Chiarion et al. (2023): Connectivity tutorial — PLV, PLI, coherence
- Li et al. (2025): PLI + coherence fusion for ADHD GCN (97.29%)
"""

import numpy as np
import os
import json
from scipy import signal
from scipy.stats import pearsonr
import networkx as nx
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import warnings

warnings.filterwarnings('ignore')

# Standard 10-20 montage positions (approximate 2D for visualization)
ELECTRODE_POSITIONS = {
    'Fp1': (-0.31, 0.95), 'Fp2': (0.31, 0.95),
    'F7': (-0.81, 0.59), 'F3': (-0.39, 0.59), 'Fz': (0.0, 0.59),
    'F4': (0.39, 0.59), 'F8': (0.81, 0.59),
    'T3': (-1.0, 0.0), 'C3': (-0.5, 0.0), 'Cz': (0.0, 0.0),
    'C4': (0.5, 0.0), 'T4': (1.0, 0.0),
    'T5': (-0.81, -0.59), 'P3': (-0.39, -0.59), 'Pz': (0.0, -0.59),
    'P4': (0.39, -0.59), 'T6': (0.81, -0.59),
    'O1': (-0.31, -0.95), 'O2': (0.31, -0.95)
}

CHANNEL_ORDER = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'Fz', 'Cz', 'Pz'
]

# Frequency bands for node features
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}


def compute_plv(epoch: np.ndarray, sfreq: float = 128.0) -> np.ndarray:
    """
    Compute Phase Locking Value (PLV) connectivity matrix.
    PLV measures phase synchronization between channels.
    
    Args:
        epoch: (n_channels, n_samples)
        sfreq: sampling frequency
    
    Returns:
        plv_matrix: (n_channels, n_channels) symmetric connectivity matrix
    """
    n_channels = epoch.shape[0]
    # Compute analytic signal using Hilbert transform
    analytic = signal.hilbert(epoch, axis=1)
    phase = np.angle(analytic)
    
    plv_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phase[i] - phase[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[i, j] = plv
            plv_matrix[j, i] = plv
    
    return plv_matrix


def compute_pli(epoch: np.ndarray, sfreq: float = 128.0) -> np.ndarray:
    """
    Compute Phase Lag Index (PLI) connectivity matrix.
    PLI is robust to volume conduction (zero-lag interactions discarded).
    
    Args:
        epoch: (n_channels, n_samples)
        sfreq: sampling frequency
    
    Returns:
        pli_matrix: (n_channels, n_channels) symmetric connectivity matrix
    """
    n_channels = epoch.shape[0]
    analytic = signal.hilbert(epoch, axis=1)
    phase = np.angle(analytic)
    
    pli_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phase[i] - phase[j]
            pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))
            pli_matrix[i, j] = pli
            pli_matrix[j, i] = pli
    
    return pli_matrix


def compute_coherence(epoch: np.ndarray, sfreq: float = 128.0) -> np.ndarray:
    """
    Compute magnitude-squared coherence matrix averaged over all frequencies.
    
    Args:
        epoch: (n_channels, n_samples)
        sfreq: sampling frequency
    
    Returns:
        coh_matrix: (n_channels, n_channels) symmetric connectivity matrix
    """
    n_channels = epoch.shape[0]
    coh_matrix = np.zeros((n_channels, n_channels))
    
    nperseg = min(256, epoch.shape[1] // 2)
    
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            f, Cxy = signal.coherence(epoch[i], epoch[j], fs=sfreq, nperseg=nperseg)
            # Average coherence across frequencies
            coh = np.mean(Cxy)
            coh_matrix[i, j] = coh
            coh_matrix[j, i] = coh
    
    return coh_matrix


def compute_correlation(epoch: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation matrix (simplest connectivity metric).
    
    Args:
        epoch: (n_channels, n_samples)
    
    Returns:
        corr_matrix: (n_channels, n_channels) absolute correlation matrix
    """
    corr = np.corrcoef(epoch)
    np.fill_diagonal(corr, 0)
    return np.abs(corr)


def compute_band_power(epoch: np.ndarray, sfreq: float = 128.0) -> np.ndarray:
    """
    Compute power spectral density in each frequency band for each channel.
    Used as node features.
    
    Args:
        epoch: (n_channels, n_samples)
        sfreq: sampling frequency
    
    Returns:
        band_powers: (n_channels, n_bands) — power in each band
    """
    n_channels = epoch.shape[0]
    n_bands = len(FREQ_BANDS)
    band_powers = np.zeros((n_channels, n_bands))
    
    for ch in range(n_channels):
        freqs, psd = signal.welch(epoch[ch], fs=sfreq, nperseg=min(256, epoch.shape[1] // 2))
        for b, (band_name, (fmin, fmax)) in enumerate(FREQ_BANDS.items()):
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            band_powers[ch, b] = np.mean(psd[idx]) if np.any(idx) else 0
    
    # Log transform for better distribution
    band_powers = np.log1p(band_powers)
    
    return band_powers


def compute_graph_metrics(adj_matrix: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Compute graph-theoretic metrics for each node using NetworkX.
    These mirror established brain network methods (clustering coefficient,
    path length, betweenness centrality, node strength).
    
    Args:
        adj_matrix: (n_nodes, n_nodes) connectivity matrix
        threshold: threshold to binarize for some metrics
    
    Returns:
        node_metrics: (n_nodes, n_metrics) — metrics per node
    """
    n_nodes = adj_matrix.shape[0]
    
    # Create weighted graph
    G = nx.from_numpy_array(adj_matrix)
    
    # Create thresholded binary graph
    binary_adj = (adj_matrix > threshold).astype(float)
    G_bin = nx.from_numpy_array(binary_adj)
    
    metrics = np.zeros((n_nodes, 5))
    
    # 1. Node strength (sum of edge weights)
    for i in range(n_nodes):
        metrics[i, 0] = adj_matrix[i].sum()
    
    # 2. Clustering coefficient (weighted)
    cc = nx.clustering(G, weight='weight')
    for i in range(n_nodes):
        metrics[i, 1] = cc[i]
    
    # 3. Betweenness centrality
    bc = nx.betweenness_centrality(G, weight='weight')
    for i in range(n_nodes):
        metrics[i, 2] = bc[i]
    
    # 4. Degree centrality (binary graph)
    dc = nx.degree_centrality(G_bin)
    for i in range(n_nodes):
        metrics[i, 3] = dc[i]
    
    # 5. Eigenvector centrality
    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
        for i in range(n_nodes):
            metrics[i, 4] = ec[i]
    except:
        pass  # May not converge for some graphs
    
    return metrics


def epoch_to_pyg_graph(
    epoch: np.ndarray,
    label: int,
    connectivity_method: str = 'plv',
    sfreq: float = 128.0,
    threshold: float = 0.2,
    include_graph_metrics: bool = True
) -> Data:
    """
    Convert a single EEG epoch to a PyTorch Geometric Data object.
    
    Args:
        epoch: (n_channels, n_samples) preprocessed EEG data
        label: class label (0 or 1)
        connectivity_method: 'plv', 'pli', 'coherence', 'correlation'
        sfreq: sampling frequency
        threshold: edge weight threshold for graph sparsification
        include_graph_metrics: whether to compute graph metrics as node features
    
    Returns:
        PyG Data object with:
        - x: node features (band_powers + graph_metrics)
        - edge_index: sparse edge connectivity
        - edge_attr: edge weights
        - y: label
        - adj: full adjacency matrix (for dashboard visualization)
    """
    # Compute connectivity matrix
    if connectivity_method == 'plv':
        adj_matrix = compute_plv(epoch, sfreq)
    elif connectivity_method == 'pli':
        adj_matrix = compute_pli(epoch, sfreq)
    elif connectivity_method == 'coherence':
        adj_matrix = compute_coherence(epoch, sfreq)
    elif connectivity_method == 'correlation':
        adj_matrix = compute_correlation(epoch)
    else:
        raise ValueError(f"Unknown method: {connectivity_method}")
    
    # Handle NaN values
    adj_matrix = np.nan_to_num(adj_matrix, nan=0.0)
    
    # --- Node features ---
    # Band power features
    band_powers = compute_band_power(epoch, sfreq)  # (19, 5)
    
    # Graph metrics features
    if include_graph_metrics:
        graph_metrics = compute_graph_metrics(adj_matrix, threshold)  # (19, 5)
        node_features = np.concatenate([band_powers, graph_metrics], axis=1)  # (19, 10)
    else:
        node_features = band_powers
    
    # Normalize node features
    feat_mean = node_features.mean(axis=0, keepdims=True)
    feat_std = node_features.std(axis=0, keepdims=True) + 1e-8
    node_features = (node_features - feat_mean) / feat_std
    
    # --- Edge construction ---
    # Threshold adjacency matrix to create sparse graph
    mask = adj_matrix > threshold
    rows, cols = np.where(mask)
    edge_weights = adj_matrix[mask]
    
    # If too few edges after thresholding, use top-k approach
    if len(rows) < 19:  # At least 1 edge per node on average
        n_edges = 19 * 3  # ~3 edges per node
        flat_idx = np.argsort(adj_matrix.flatten())[::-1]
        rows_all, cols_all = np.unravel_index(flat_idx, adj_matrix.shape)
        # Remove self-loops and duplicates
        valid = rows_all != cols_all
        rows = rows_all[valid][:n_edges]
        cols = cols_all[valid][:n_edges]
        edge_weights = adj_matrix[rows, cols]
    
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1)
    
    # Build Data object
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([label], dtype=torch.long),
        adj=torch.tensor(adj_matrix, dtype=torch.float32)
    )
    
    return data


def build_graph_dataset(
    epochs: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    connectivity_method: str = 'plv',
    sfreq: float = 128.0,
    threshold: float = 0.2,
    output_path: str = None
) -> list:
    """
    Convert all epochs to PyG graph dataset.
    
    Args:
        epochs: (n_epochs, n_channels, n_samples)
        labels: (n_epochs,)
        subjects: (n_epochs,)
        connectivity_method: connectivity metric to use
        sfreq: sampling frequency
        threshold: edge weight threshold
        output_path: path to save the dataset
    
    Returns:
        List of PyG Data objects
    """
    dataset = []
    
    for i in tqdm(range(len(epochs)), desc=f"Building {connectivity_method} graphs"):
        try:
            data = epoch_to_pyg_graph(
                epochs[i], labels[i],
                connectivity_method=connectivity_method,
                sfreq=sfreq,
                threshold=threshold
            )
            data.subject = subjects[i]
            dataset.append(data)
        except Exception as e:
            continue
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(dataset, output_path)
        print(f"  Saved {len(dataset)} graphs to {output_path}")
    
    return dataset


def compute_global_graph_metrics(adj_matrix: np.ndarray, threshold: float = 0.3) -> dict:
    """
    Compute global (graph-level) metrics — used for dashboard display.
    These match established brain network analysis metrics.
    """
    binary_adj = (adj_matrix > threshold).astype(float)
    G = nx.from_numpy_array(binary_adj)
    G_weighted = nx.from_numpy_array(adj_matrix)
    
    metrics = {}
    
    # Average clustering coefficient
    metrics['clustering_coeff'] = nx.average_clustering(G_weighted, weight='weight')
    
    # Average shortest path length (if connected)
    if nx.is_connected(G):
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)
    else:
        # Use largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
        if len(subG) > 1:
            metrics['avg_path_length'] = nx.average_shortest_path_length(subG)
        else:
            metrics['avg_path_length'] = 0
    
    # Small-worldness (sigma) — ratio of clustering to path length
    # compared to random graph
    n = adj_matrix.shape[0]
    density = binary_adj.sum() / (n * (n - 1))
    if density > 0:
        # Expected values for random graph
        C_random = density
        L_random = np.log(n) / np.log(density * n) if density * n > 1 else n
        
        C = metrics['clustering_coeff']
        L = metrics['avg_path_length'] if metrics['avg_path_length'] > 0 else 1
        
        metrics['small_worldness'] = (C / max(C_random, 1e-8)) / (L / max(L_random, 1e-8))
    else:
        metrics['small_worldness'] = 0
    
    # Global efficiency
    metrics['global_efficiency'] = nx.global_efficiency(G)
    
    # Graph density
    metrics['density'] = nx.density(G)
    
    return metrics


if __name__ == '__main__':
    from pathlib import Path
    
    base_dir = Path(__file__).resolve().parents[2]
    processed_dir = base_dir / 'data' / 'processed'
    
    # Load preprocessed data
    for dataset_name in ['eegmat', 'adhd']:
        print(f"\n{'='*60}")
        print(f"BUILDING GRAPHS: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        epochs = np.load(processed_dir / f'{dataset_name}_epochs.npy')
        labels = np.load(processed_dir / f'{dataset_name}_labels.npy')
        subjects = np.load(processed_dir / f'{dataset_name}_subjects.npy', allow_pickle=True)
        
        print(f"  Loaded {len(epochs)} epochs, shape {epochs.shape}")
        
        # Build PLV graphs (primary method — best for cognitive tasks)
        dataset = build_graph_dataset(
            epochs, labels, subjects,
            connectivity_method='plv',
            sfreq=128.0,
            threshold=0.2,
            output_path=str(processed_dir / f'{dataset_name}_plv_graphs.pt')
        )
        
        # Print sample stats
        if dataset:
            sample = dataset[0]
            print(f"\n  Sample graph:")
            print(f"    Nodes: {sample.x.shape[0]}, Features: {sample.x.shape[1]}")
            print(f"    Edges: {sample.edge_index.shape[1]}")
            print(f"    Label: {sample.y.item()}")
            
            # Compute global metrics for a sample
            adj = sample.adj.numpy()
            gmetrics = compute_global_graph_metrics(adj)
            print(f"    Global metrics: {json.dumps({k: round(v, 4) for k, v in gmetrics.items()})}")
    
    print(f"\n{'='*60}")
    print("GRAPH CONSTRUCTION COMPLETE")
    print(f"{'='*60}")
