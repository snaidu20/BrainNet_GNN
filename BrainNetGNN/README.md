# BrainNetGNN: Graph Neural Networks for EEG-Based Brain Network Analysis

BrainNetGNN is an end-to-end research and applied-computing project for analyzing EEG-derived functional brain networks with graph neural networks. The pipeline converts multichannel EEG recordings into graph-structured representations, learns predictive patterns with graph models, and presents results through an interactive visualization dashboard. The current implementation targets two use cases: cognitive workload classification and ADHD-related EEG pattern analysis.

## Project Overview

This project combines signal preprocessing, brain-network construction, graph feature engineering, graph neural network modeling, and visualization into a single workflow. EEG signals are segmented into epochs, transformed into functional connectivity graphs, enriched with spectral and graph-theoretic node features, and then passed to graph learning models such as GCN and GAT. Alongside these models, the project includes conventional baselines and a dashboard for interpretation and demonstration.

## What This Project Does

- Builds functional brain networks from 19-channel EEG recordings.
- Preprocesses EEG signals and divides them into fixed analysis windows.
- Computes connectivity matrices using PLV, PLI, coherence, and correlation.
- Generates node features from multi-band spectral power and graph measures.
- Trains graph neural networks, including GCN and GAT, for graph classification.
- Compares graph models against standard machine-learning baselines.
- Provides a dashboard for predictions, connectivity inspection, and electrode-level interpretation.

## Key Discoveries

### 1. Functional connectivity can be turned into a learnable graph pipeline
The project shows that EEG functional connectivity is not only useful for handcrafted graph analysis, but can also serve as a structured input for graph neural networks. Instead of stopping at manually designed network descriptors, the workflow enables the model to learn predictive patterns directly from graph topology and node features.

### 2. Multiple connectivity metrics offer complementary signal views
By constructing graphs from PLV, PLI, coherence, and correlation, the project explores how different connectivity definitions capture different aspects of EEG relationships. This supports a broader experimental view of brain-network modeling rather than relying on a single connectivity assumption.

### 3. Graph attention improves interpretability potential
The use of GAT introduces a mechanism for estimating task-relevant node importance through attention weights. This creates a pathway for electrode-level interpretation and makes the system more useful for explainable AI discussions, demos, and decision-support interfaces.

### 4. Subject-wise evaluation makes results more realistic
The project uses a strict subject-wise split, which is a stronger evaluation protocol than random sample-level splitting. This reduces leakage across train and test sets and gives a more honest estimate of how well the model may generalize to unseen individuals.

### 5. The contribution is broader than a single model
This work is not only a model experiment. It also demonstrates an integrated applied-computing system that includes data preprocessing, graph construction, learning, benchmarking, and user-facing visualization.

## Current Results

The current implementation reports moderate but meaningful performance under realistic subject-wise evaluation:

- EEGMAT workload classification: strongest reported result around 0.736 accuracy with GCN.
- ADHD task: strongest reported result around 0.615 accuracy with GAT.

These numbers suggest that the problem is genuinely challenging and that the evaluation is not inflated by easy leakage. At the same time, they indicate that the system is still in a research-prototype stage rather than a high-confidence production or clinical solution.

## Why This Matters

This project matters because it demonstrates a usable bridge between neuroscience-inspired graph construction and modern graph learning. It shows how EEG-based brain networks can support:

- cognitive state monitoring,
- neurodevelopmental-condition screening research,
- interpretable graph-based modeling,
- dashboard-driven decision support,
- future real-time operator or learner monitoring systems.

The strongest practical value today is in research workflows, academic demonstrations, and human-in-the-loop decision-support settings.

## Limitations

### 1. Predictive performance is still moderate
Although the evaluation setup is realistic, the current accuracy levels are not yet strong enough for high-stakes autonomous use. In particular, clinical or screening scenarios would require much stronger reliability, calibration, and external validation.

### 2. The pipeline is not fully end-to-end
The graph neural networks operate on graphs that are already constructed through handcrafted preprocessing and connectivity estimation. This means the system still depends heavily on front-end design choices such as windowing, filtering, and graph-construction method.

### 3. Real-time deployment would need optimization
The present workflow includes multiple computational stages before inference, including preprocessing, feature extraction, and graph construction. For true real-time use, latency and throughput would need to be measured and optimized carefully.

### 4. Generalization needs broader validation
The project uses public datasets, which is valuable for reproducibility, but stronger claims would require testing across additional cohorts, acquisition conditions, and external datasets.

### 5. Interpretability needs deeper validation
Attention weights are useful for suggesting node importance, but they should not automatically be treated as ground-truth neurophysiological explanations. Stronger interpretability claims would require systematic validation against domain knowledge or complementary explainability methods.

## Real-World Impact Potential

The project has promising real-world potential in lower-risk and assistive environments such as:

- workload estimation during complex tasks,
- attention and engagement monitoring,
- educational analytics,
- research decision-support dashboards,
- prototype neuro-AI interfaces.

For medical or clinical deployment, the current system should be viewed as an early-stage research platform rather than a deployment-ready diagnostic tool.

## What Makes This Project Valuable

The main value of BrainNetGNN lies in the combination of three strengths:

1. **Technical depth**: EEG preprocessing, graph construction, feature engineering, and GNN modeling.
2. **Research relevance**: brain-network learning with interpretable graph-based analysis.
3. **Applied-computing orientation**: dashboard integration, baseline comparison, and translational usability.

This makes the project suitable for research portfolios, graduate applications, and future extension into more advanced graph-learning studies.

## Recommended Next Steps

To strengthen the project further, the next improvements should include:

- dynamic or temporal graph modeling across EEG windows,
- stronger external validation across datasets,
- uncertainty estimation and calibration analysis,
- latency benchmarking for real-time claims,
- deeper comparison across connectivity-construction strategies,
- improved explainability beyond attention alone.

## Repository Positioning

BrainNetGNN should be presented as a graph-learning research prototype for EEG-based functional brain-network analysis. It is best positioned as an interpretable and extensible system for benchmarking, visualization, and applied neuro-AI exploration rather than as a finalized diagnostic product.
