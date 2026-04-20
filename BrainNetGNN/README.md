# BrainNetGNN

**Graph Neural Network Analysis of Brain Functional Connectivity from EEG**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MNE](https://img.shields.io/badge/MNE--Python-1.12-green.svg)](https://mne.tools/)

BrainNetGNN applies Graph Neural Networks (GCN, GAT) to functional brain networks constructed from multi-channel EEG recordings for **cognitive workload classification** and **ADHD detection**. The project includes an interactive dashboard for real-time brain network visualization and GNN-based prediction.

This work extends established graph-theoretic brain network analysis methods — including Minimum Connected Component (MCC), Shortest Path Networks, and transfer entropy-based directed connectivity — by integrating modern Graph Neural Networks, public benchmark datasets, and interactive visualization.

---

## Key Features

- **EEG-to-Graph Pipeline**: Converts raw EEG to functional brain networks using PLV, PLI, coherence, and correlation connectivity metrics
- **GNN Classification**: GCN and GAT models for cognitive state classification with attention-based electrode importance
- **Graph Theory Metrics**: Clustering coefficient, average path length, small-worldness, global efficiency — following established brain network analysis methods
- **Interactive Dashboard**: Real-time brain network visualization with Dash Cytoscape, prediction gauges, and ADHD vs. healthy comparison
- **Dual-Task Design**: Cognitive workload detection (EEGMAT) and ADHD screening (Nasrabadi dataset)

---

## Architecture

```
Raw EEG (19-channel, 10-20 system)
    │
    ├─ Preprocessing (MNE-Python)
    │   ├─ Bandpass filter (1-45 Hz)
    │   ├─ 50 Hz notch filter
    │   ├─ Resampling to 128 Hz
    │   └─ 4-second epoch segmentation
    │
    ├─ Graph Construction
    │   ├─ Connectivity matrix (PLV / PLI / Coherence / Correlation)
    │   ├─ Node features: 5 frequency band powers + 5 graph metrics
    │   ├─ Edge sparsification (threshold-based)
    │   └─ PyTorch Geometric Data objects
    │
    ├─ GNN Models
    │   ├─ GCN (3-layer + global pool + MLP)
    │   ├─ GAT (3-layer, 4 heads + attention extraction)
    │   ├─ Logistic Regression baseline
    │   └─ Random Forest baseline
    │
    └─ Interactive Dashboard (Plotly Dash)
        ├─ Brain network graph (Cytoscape)
        ├─ Real-time GNN prediction
        ├─ Graph theory metrics gauges
        ├─ Electrode importance heatmap
        ├─ Connectivity comparison (class A vs. class B)
        └─ Frequency band power chart
```

---

## Datasets

| Dataset | Source | Subjects | Channels | Task | Sampling Rate | Access |
|---------|--------|----------|----------|------|---------------|--------|
| **EEGMAT** | [PhysioNet](https://physionet.org/content/eegmat/1.0.0/) | 36 | 19 EEG (10-20) | Mental arithmetic vs. rest | 500 Hz (↓128 Hz) | Open access |
| **ADHD** | [IEEE DataPort / Kaggle](https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd) | 121 (61 ADHD, 60 control) | 19 EEG (10-20) | Visual attention task | 128 Hz | Open access |

### Preprocessing Statistics

| Dataset | Epochs | Epoch Shape | Class Balance |
|---------|--------|-------------|---------------|
| EEGMAT | 2,025 | (19, 512) | 1,499 baseline / 526 task |
| ADHD | 2,705 | (19, 512) | 1,246 control / 1,459 ADHD |

---

## Results

Subject-wise train/test split (80/20) to prevent data leakage:

### Cognitive Workload (EEGMAT)

| Model | Accuracy | F1 Score | AUC |
|-------|----------|----------|-----|
| **GAT** | 0.726 | **0.695** | **0.554** |
| GCN | **0.736** | 0.694 | 0.550 |
| Random Forest | 0.721 | 0.689 | 0.670 |
| Logistic Regression | 0.632 | 0.658 | **0.729** |

### ADHD Detection

| Model | Accuracy | F1 Score | AUC |
|-------|----------|----------|-----|
| **GAT** | **0.615** | **0.610** | **0.618** |
| GCN | 0.587 | 0.574 | 0.592 |
| Random Forest | 0.609 | 0.609 | 0.647 |
| Logistic Regression | 0.481 | 0.482 | 0.485 |

> **Note**: Results use strict subject-wise cross-validation. Literature figures (95%+) typically use within-subject or within-epoch splits. Our approach is more realistic for clinical deployment.

---

## Graph Theory Metrics

Following established brain network analysis methodology, we compute the following graph-theoretic measures for each brain network:

| Metric | Description | Reference |
|--------|-------------|-----------|
| **Clustering Coefficient** | Local connectivity density | Ramasamy et al. (2017), Neural Processing Letters |
| **Average Path Length** | Global integration efficiency | Ramasamy et al. (2018), J. Integrative Neuroscience |
| **Small-Worldness (σ)** | Balance of segregation and integration | Ramasamy et al. (2015), Neurocomputing |
| **Global Efficiency** | Inverse of average shortest path | Latora & Marchiori (2001) |
| **Betweenness Centrality** | Node importance in information flow | Freeman (1977) |

---

## Project Structure

```
BrainNetGNN/
├── src/
│   ├── preprocessing/
│   │   └── eeg_preprocessor.py      # EEG filtering, epoching (MNE-Python)
│   ├── graph_construction/
│   │   └── brain_graph_builder.py    # PLV/PLI/coherence → PyG graphs
│   ├── gnn_model/
│   │   └── train.py                  # GCN, GAT, baselines + attention extraction
│   └── dashboard/
│       └── app.py                    # Interactive Plotly Dash dashboard
├── data/
│   ├── raw/                          # Original EEG files
│   │   ├── eegmat/                   # PhysioNet EEGMAT (72 EDF files)
│   │   └── adhd/                     # Nasrabadi ADHD dataset (CSV)
│   ├── processed/                    # Preprocessed epochs + PyG graphs
│   └── models/                       # Trained model weights + results
├── docs/
│   └── dashboard_v2.png              # Dashboard screenshot
├── brainnet_study/                   # Research notes
│   ├── ramasamy_papers_analysis.md   # 16-paper deep analysis
│   ├── datasets_and_tools.md         # 15 datasets cataloged
│   └── use_cases_and_architecture.md # GNN architecture review
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/snaidu20/BrainNetGNN.git
cd BrainNetGNN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Download Datasets

**EEGMAT** (auto-downloads from PhysioNet):
```bash
cd data/raw/eegmat
wget -r -N -c -np https://physionet.org/files/eegmat/1.0.0/
```

**ADHD** (from Kaggle):
```bash
# Download from https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd
# Place adhdata.csv in data/raw/adhd/
```

### 2. Preprocess EEG Data

```bash
python src/preprocessing/eeg_preprocessor.py
```

### 3. Build Brain Graphs

```bash
python src/graph_construction/brain_graph_builder.py
```

### 4. Train Models

```bash
python src/gnn_model/train.py
```

### 5. Launch Dashboard

```bash
python src/dashboard/app.py
# Open http://localhost:8050
```

---

## Methodology — Extending Brain Network Analysis with GNNs

### Prior Work in Graph-Theoretic Brain Network Analysis

Established methods for functional brain network (FBN) analysis have contributed foundational techniques:

- **MCC Algorithm** (Neurocomputing 2015) — Parameter-free spanning subgraph for FBN binarization
- **Transfer Entropy Directed FBNs** (Neural Processing Letters 2017) — Directed connectivity via normalized transfer entropy
- **Shortest Path Networks** (J. Integrative Neuroscience 2018) — Weighted network construction using path traversal frequency
- **PageRank Electrode Ranking** (Springer 2021) — Weighted PageRank for electrode importance scoring

### What BrainNetGNN Adds

| Gap in Prior Work | BrainNetGNN Solution |
|---------------------|---------------------|
| Small private datasets (9-16 subjects) | Public datasets: 36 (EEGMAT) + 121 (ADHD) subjects |
| Single paradigm (driving) | Multiple tasks: arithmetic, visual attention |
| No public benchmarks | PhysioNet + IEEE DataPort datasets |
| No real-time processing | Interactive dashboard with live graph updates |
| No clinical populations | ADHD children dataset (61 ADHD + 60 control) |
| No deep learning | GCN + GAT with attention-based explainability |
| No open-source code | Full pipeline: preprocessing → GNN → dashboard |

### Key Innovation

The GAT attention mechanism provides a modern, learnable alternative to traditional PageRank-based electrode ranking — both identify which brain regions contribute most to classification, but GAT attention is task-adaptive and end-to-end differentiable.

---

## Real-World Applications

1. **Driver Fatigue Monitoring** — Detect cognitive overload before accidents by classifying brain network topology in real-time
2. **ADHD Screening** — Identify abnormal brain connectivity patterns (frontal hypo-connectivity) as an objective diagnostic aid
3. **Student Engagement Tracking** — Monitor cognitive load during online learning to improve educational outcomes
4. **Clinical Brain Network Biomarkers** — Quantitative graph-theoretic metrics as potential biomarkers for neurological conditions

---

## References

### Brain Network Analysis Foundations
- Ramasamy, V. et al. (2015). "Minimum Connected Component — A Novel Approach to Detection of Cognitive Load Induced Changes in Functional Brain Networks." *Neurocomputing*, 170, 15-31.
- Ramasamy, V. et al. (2017). "Directed Connectivity Analysis of Functional Brain Networks during Cognitive Activity Using Transfer Entropy." *Neural Processing Letters*, 45(3), 807-824.
- Ramasamy, V. et al. (2018). "Shortest Path Based Network Analysis to Characterize Cognitive Load States." *J. Integrative Neuroscience*, 17(2), 213-230.

### GNN on EEG — State of the Art
- Li et al. (2025). "ADHD detection from EEG signals using GCN based on multi-domain features." *Frontiers in Neuroscience*. (97.29%)
- Pandey et al. (2025). "WL-GraphTrax: A Graph-Transformer Framework for EEG-Based Cognitive Workload Classification." *IEEE Access*. (95.68%)
- Chiarion et al. (2023). "Connectivity Analysis in EEG Data: A Tutorial Review." *Bioengineering*.

### Datasets
- Zyma, I. et al. (2019). "Electroencephalograms during Mental Arithmetic Task Performance." *PhysioNet*. https://physionet.org/content/eegmat/1.0.0/
- Nasrabadi et al. (2020). "EEG data for ADHD / Control children." *IEEE DataPort*.

---

## Author

**Sai Kumar Naidu** — MS in Computer Science, Florida Atlantic University  
GitHub: [snaidu20](https://github.com/snaidu20)

---

## License

This project is for academic and research purposes. Datasets are used under their respective licenses (PhysioNet: ODC-BY; ADHD: IEEE DataPort open access).
