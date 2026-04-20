"""
BrainNetGNN Interactive Dashboard
==================================
Interactive visualization of brain network analysis using GNNs.

Features:
1. Brain network graph visualization (Cytoscape) with electrode positions
2. Real-time EEG replay with prediction gauge
3. Graph theory metrics panel (established brain network metrics)
4. GNN attention-based electrode importance heatmap
5. ADHD vs Healthy brain topology comparison
6. Connectivity method selector

Built with Plotly Dash + Dash Cytoscape
"""

import os
import sys
import json
import numpy as np
import torch
import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.graph_construction.brain_graph_builder import (
    compute_plv, compute_pli, compute_coherence, compute_correlation,
    compute_band_power, compute_global_graph_metrics,
    ELECTRODE_POSITIONS, CHANNEL_ORDER, FREQ_BANDS
)
from src.gnn_model.train import BrainGCN, BrainGAT

# =================== DATA LOADING ===================

def load_data():
    """Load preprocessed data and trained models."""
    data_dir = project_root / 'data'
    processed = data_dir / 'processed'
    models = data_dir / 'models'
    
    result = {}
    
    for name in ['eegmat', 'adhd']:
        epochs = np.load(processed / f'{name}_epochs.npy')
        labels = np.load(processed / f'{name}_labels.npy')
        subjects = np.load(processed / f'{name}_subjects.npy', allow_pickle=True)
        
        # Load results
        results_path = models / f'{name}_results.json'
        if results_path.exists():
            with open(results_path) as f:
                train_results = json.load(f)
        else:
            train_results = {}
        
        # Load electrode importance
        imp_path = models / f'{name}_electrode_importance.npy'
        importance = np.load(imp_path) if imp_path.exists() else np.ones(19) / 19
        
        result[name] = {
            'epochs': epochs,
            'labels': labels,
            'subjects': subjects,
            'results': train_results,
            'importance': importance
        }
    
    # Load GAT models
    for name in ['eegmat', 'adhd']:
        model_path = models / f'{name}_gat.pt'
        if model_path.exists():
            model = BrainGAT(in_features=10, hidden_dim=64, num_classes=2, heads=4)
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
            model.eval()
            result[name]['model'] = model
    
    return result


# =================== GRAPH VISUALIZATION ===================

def build_cytoscape_elements(adj_matrix, importance=None, threshold=0.25):
    """Build Cytoscape elements from adjacency matrix."""
    elements = []
    
    # Scale positions for display
    scale = 200
    
    for i, ch in enumerate(CHANNEL_ORDER):
        pos = ELECTRODE_POSITIONS[ch]
        imp = importance[i] if importance is not None else 0.5
        
        elements.append({
            'data': {
                'id': ch,
                'label': ch,
                'importance': float(imp),
                'size': 20 + imp * 30,
            },
            'position': {
                'x': pos[0] * scale + 250,
                'y': -pos[1] * scale + 250  # Flip Y for display
            },
            'classes': 'electrode'
        })
    
    # Add edges above threshold
    for i in range(len(CHANNEL_ORDER)):
        for j in range(i + 1, len(CHANNEL_ORDER)):
            weight = adj_matrix[i, j]
            if weight > threshold:
                elements.append({
                    'data': {
                        'source': CHANNEL_ORDER[i],
                        'target': CHANNEL_ORDER[j],
                        'weight': float(weight),
                        'width': max(1, weight * 5),
                    },
                    'classes': 'connection'
                })
    
    return elements


def create_brain_heatmap(importance, title="Electrode Importance"):
    """Create a plotly figure showing electrode importance as a brain map."""
    fig = go.Figure()
    
    # Head outline
    theta = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta) * 1.1, y=np.sin(theta) * 1.1,
        mode='lines', line=dict(color='#555', width=2),
        showlegend=False, hoverinfo='skip'
    ))
    
    # Nose
    fig.add_trace(go.Scatter(
        x=[-.1, 0, .1], y=[1.1, 1.25, 1.1],
        mode='lines', line=dict(color='#555', width=2),
        showlegend=False, hoverinfo='skip'
    ))
    
    # Electrodes
    x_pos = [ELECTRODE_POSITIONS[ch][0] for ch in CHANNEL_ORDER]
    y_pos = [ELECTRODE_POSITIONS[ch][1] for ch in CHANNEL_ORDER]
    
    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode='markers+text',
        marker=dict(
            size=importance * 40 + 10,
            color=importance,
            colorscale='YlOrRd',
            showscale=True,
            colorbar=dict(title="Importance", x=1.05),
            line=dict(width=1, color='#333')
        ),
        text=CHANNEL_ORDER,
        textposition='top center',
        textfont=dict(size=9, color='#333'),
        hovertemplate='%{text}: %{marker.color:.3f}<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(visible=False, range=[-1.4, 1.6]),
        yaxis=dict(visible=False, range=[-1.3, 1.4], scaleanchor='x'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=10, r=10, t=40, b=10),
        height=350
    )
    
    return fig


def create_metrics_gauge(value, title, max_val=1.0):
    """Create a gauge chart for a single metric."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 13}},
        number={'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, max_val], 'tickfont': {'size': 10}},
            'bar': {'color': '#3b82f6'},
            'bgcolor': '#f1f5f9',
            'borderwidth': 1,
            'bordercolor': '#e2e8f0',
            'steps': [
                {'range': [0, max_val * 0.33], 'color': '#dbeafe'},
                {'range': [max_val * 0.33, max_val * 0.66], 'color': '#93c5fd'},
                {'range': [max_val * 0.66, max_val], 'color': '#60a5fa'},
            ]
        }
    ))
    fig.update_layout(
        height=180, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='white', font=dict(color='#1e293b')
    )
    return fig


def create_prediction_bar(probs, labels=['Class 0', 'Class 1']):
    """Create horizontal bar chart for prediction probabilities."""
    colors = ['#60a5fa', '#f87171']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=probs,
        orientation='h',
        marker_color=colors,
        text=[f'{p:.1%}' for p in probs],
        textposition='auto',
        textfont=dict(size=14, color='white')
    ))
    fig.update_layout(
        height=120,
        xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
        yaxis=dict(tickfont=dict(size=12)),
        margin=dict(l=80, r=20, t=10, b=10),
        paper_bgcolor='white', plot_bgcolor='white'
    )
    return fig


def create_comparison_figure(adj_class0, adj_class1, metric_name="PLV"):
    """Side-by-side brain network comparison."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Control / Baseline', 'ADHD / Task'],
        horizontal_spacing=0.05
    )
    
    for col, adj in enumerate([adj_class0, adj_class1], 1):
        fig.add_trace(go.Heatmap(
            z=adj,
            x=CHANNEL_ORDER,
            y=CHANNEL_ORDER,
            colorscale='RdBu_r',
            zmin=0, zmax=1,
            showscale=col == 2,
            colorbar=dict(title=metric_name, x=1.02) if col == 2 else None,
            hovertemplate='%{x} → %{y}: %{z:.3f}<extra></extra>'
        ), row=1, col=col)
    
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor='white',
        font=dict(size=10)
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
    fig.update_yaxes(tickfont=dict(size=8))
    
    return fig


def create_band_power_chart(epoch, sfreq=128.0):
    """Bar chart of frequency band power across electrodes."""
    band_powers = compute_band_power(epoch, sfreq)  # (19, 5)
    bands = list(FREQ_BANDS.keys())
    
    fig = go.Figure()
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
    for b, (band, color) in enumerate(zip(bands, colors)):
        fig.add_trace(go.Bar(
            name=band.capitalize(),
            x=CHANNEL_ORDER,
            y=band_powers[:, b],
            marker_color=color
        ))
    
    fig.update_layout(
        barmode='group',
        height=250,
        margin=dict(l=40, r=10, t=10, b=30),
        legend=dict(orientation='h', y=1.15, x=0.5, xanchor='center', font=dict(size=10)),
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(title='Log Power', title_font=dict(size=10), tickfont=dict(size=9)),
        paper_bgcolor='white', plot_bgcolor='white'
    )
    
    return fig


# =================== DASH APP ===================

# Load all data
print("Loading data and models...")
DATA = load_data()
print("Data loaded successfully!")

# Stylesheet for Cytoscape
cyto_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'label': 'data(label)',
            'width': 'data(size)',
            'height': 'data(size)',
            'font-size': '10px',
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': 'mapData(importance, 0, 1, #dbeafe, #dc2626)',
            'border-width': 2,
            'border-color': '#475569',
            'color': '#1e293b',
            'font-weight': 'bold'
        }
    },
    {
        'selector': 'edge',
        'style': {
            'width': 'data(width)',
            'line-color': 'mapData(weight, 0, 1, #e2e8f0, #3b82f6)',
            'opacity': 0.6,
            'curve-style': 'bezier'
        }
    }
]

# Create Dash app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="BrainNetGNN Dashboard"
)

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("BrainNetGNN", style={
            'margin': '0', 'fontSize': '28px', 'fontWeight': '700', 'color': '#1e293b'
        }),
        html.P("Graph Neural Network Analysis of Brain Functional Connectivity", style={
            'margin': '2px 0 0', 'fontSize': '13px', 'color': '#64748b'
        }),
    ], style={
        'padding': '16px 24px', 'borderBottom': '1px solid #e2e8f0',
        'background': 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)'
    }),
    
    # Controls Row
    html.Div([
        html.Div([
            html.Label("Dataset", style={'fontWeight': '600', 'fontSize': '12px', 'color': '#475569'}),
            dcc.Dropdown(
                id='dataset-selector',
                options=[
                    {'label': 'Cognitive Workload (EEGMAT)', 'value': 'eegmat'},
                    {'label': 'ADHD Detection', 'value': 'adhd'}
                ],
                value='adhd',
                clearable=False,
                style={'fontSize': '13px'}
            )
        ], style={'width': '220px', 'marginRight': '16px'}),
        
        html.Div([
            html.Label("Connectivity", style={'fontWeight': '600', 'fontSize': '12px', 'color': '#475569'}),
            dcc.Dropdown(
                id='connectivity-selector',
                options=[
                    {'label': 'Phase Locking Value (PLV)', 'value': 'plv'},
                    {'label': 'Phase Lag Index (PLI)', 'value': 'pli'},
                    {'label': 'Coherence', 'value': 'coherence'},
                    {'label': 'Correlation', 'value': 'correlation'}
                ],
                value='plv',
                clearable=False,
                style={'fontSize': '13px'}
            )
        ], style={'width': '220px', 'marginRight': '16px'}),
        
        html.Div([
            html.Label("Subject", style={'fontWeight': '600', 'fontSize': '12px', 'color': '#475569'}),
            dcc.Dropdown(id='subject-selector', clearable=False, style={'fontSize': '13px'})
        ], style={'width': '150px', 'marginRight': '16px'}),
        
        html.Div([
            html.Label("Epoch", style={'fontWeight': '600', 'fontSize': '12px', 'color': '#475569'}),
            dcc.Slider(
                id='epoch-slider', min=0, max=10, step=1, value=0,
                marks=None, tooltip={"placement": "bottom"}
            )
        ], style={'width': '200px', 'marginRight': '16px'}),
        
        html.Div([
            html.Label("Edge Threshold", style={'fontWeight': '600', 'fontSize': '12px', 'color': '#475569'}),
            dcc.Slider(
                id='threshold-slider', min=0.1, max=0.8, step=0.05, value=0.3,
                marks=None, tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={'width': '180px'}),
    ], style={
        'display': 'flex', 'alignItems': 'flex-end', 'padding': '12px 24px',
        'gap': '4px', 'borderBottom': '1px solid #e2e8f0', 'flexWrap': 'wrap'
    }),
    
    # Main Content — 3 column layout
    html.Div([
        # Left Column — Brain Graph + Prediction
        html.Div([
            # Dataset info card
            html.Div(id='info-card', style={
                'background': '#f8fafc', 'borderRadius': '8px', 'padding': '12px',
                'marginBottom': '12px', 'border': '1px solid #e2e8f0'
            }),
            
            # Brain graph
            html.Div([
                html.H3("Brain Connectivity Network", style={
                    'margin': '0 0 8px', 'fontSize': '15px', 'color': '#1e293b'
                }),
                cyto.Cytoscape(
                    id='brain-graph',
                    layout={'name': 'preset'},
                    style={'width': '100%', 'height': '420px', 'border': '1px solid #e2e8f0', 'borderRadius': '8px'},
                    stylesheet=cyto_stylesheet,
                    elements=[],
                    userPanningEnabled=True,
                    userZoomingEnabled=True,
                    minZoom=0.5,
                    maxZoom=2.0
                )
            ], style={
                'background': 'white', 'borderRadius': '8px', 'padding': '12px',
                'border': '1px solid #e2e8f0'
            }),
            
            # Prediction bar
            html.Div([
                html.H3("GNN Prediction", style={
                    'margin': '0 0 4px', 'fontSize': '15px', 'color': '#1e293b'
                }),
                dcc.Graph(id='prediction-bar', config={'displayModeBar': False})
            ], style={
                'background': 'white', 'borderRadius': '8px', 'padding': '12px',
                'border': '1px solid #e2e8f0', 'marginTop': '12px'
            })
        ], style={'flex': '1.2', 'minWidth': '400px'}),
        
        # Middle Column — Graph Metrics + Band Power
        html.Div([
            html.Div([
                html.H3("Graph Theory Metrics", style={
                    'margin': '0 0 8px', 'fontSize': '15px', 'color': '#1e293b'
                }),
                html.Div([
                    html.Div([dcc.Graph(id='gauge-clustering', config={'displayModeBar': False})],
                             style={'width': '50%'}),
                    html.Div([dcc.Graph(id='gauge-pathlength', config={'displayModeBar': False})],
                             style={'width': '50%'}),
                ], style={'display': 'flex'}),
                html.Div([
                    html.Div([dcc.Graph(id='gauge-smallworld', config={'displayModeBar': False})],
                             style={'width': '50%'}),
                    html.Div([dcc.Graph(id='gauge-efficiency', config={'displayModeBar': False})],
                             style={'width': '50%'}),
                ], style={'display': 'flex'}),
            ], style={
                'background': 'white', 'borderRadius': '8px', 'padding': '12px',
                'border': '1px solid #e2e8f0'
            }),
            
            # Band power chart
            html.Div([
                html.H3("Frequency Band Power", style={
                    'margin': '0 0 4px', 'fontSize': '15px', 'color': '#1e293b'
                }),
                dcc.Graph(id='band-power-chart', config={'displayModeBar': False})
            ], style={
                'background': 'white', 'borderRadius': '8px', 'padding': '12px',
                'border': '1px solid #e2e8f0', 'marginTop': '12px'
            }),
        ], style={'flex': '1', 'minWidth': '350px'}),
        
        # Right Column — Electrode Importance + Comparison
        html.Div([
            # Electrode importance heatmap
            html.Div([
                html.H3("Electrode Importance (GAT Attention)", style={
                    'margin': '0 0 4px', 'fontSize': '15px', 'color': '#1e293b'
                }),
                dcc.Graph(id='electrode-heatmap', config={'displayModeBar': False})
            ], style={
                'background': 'white', 'borderRadius': '8px', 'padding': '12px',
                'border': '1px solid #e2e8f0'
            }),
            
            # Class comparison
            html.Div([
                html.H3("Connectivity Comparison", style={
                    'margin': '0 0 4px', 'fontSize': '15px', 'color': '#1e293b'
                }),
                dcc.Graph(id='comparison-heatmap', config={'displayModeBar': False})
            ], style={
                'background': 'white', 'borderRadius': '8px', 'padding': '12px',
                'border': '1px solid #e2e8f0', 'marginTop': '12px'
            }),
            
            # Model results
            html.Div(id='model-results', style={
                'background': 'white', 'borderRadius': '8px', 'padding': '12px',
                'border': '1px solid #e2e8f0', 'marginTop': '12px'
            }),
        ], style={'flex': '1', 'minWidth': '350px'}),
        
    ], style={
        'display': 'flex', 'gap': '16px', 'padding': '16px 24px',
        'flexWrap': 'wrap'
    }),
    
    # Footer
    html.Div([
        html.P(
            "BrainNetGNN — Graph Neural Network Analysis of Brain Functional Connectivity. "
            "Data: EEGMAT (PhysioNet, 36 subjects) + ADHD (IEEE DataPort, 121 subjects). "
            "Built by Sai Kumar Naidu.",
            style={'margin': '0', 'fontSize': '11px', 'color': '#94a3b8'}
        )
    ], style={
        'padding': '12px 24px', 'borderTop': '1px solid #e2e8f0', 'textAlign': 'center'
    }),
    
    # Hidden stores
    dcc.Store(id='current-adj-matrix'),
    dcc.Store(id='current-epoch-data'),
    
], style={
    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    'backgroundColor': '#f1f5f9', 'minHeight': '100vh'
})


# =================== CALLBACKS ===================

@app.callback(
    Output('subject-selector', 'options'),
    Output('subject-selector', 'value'),
    Input('dataset-selector', 'value')
)
def update_subject_list(dataset):
    d = DATA[dataset]
    subjects = np.unique(d['subjects'])
    options = [{'label': f"{s} ({'ADHD' if d['labels'][d['subjects']==s][0]==1 else 'Control'})" if dataset == 'adhd' 
                else f"{s} ({'Task' if d['labels'][d['subjects']==s][0]==1 else 'Rest'})",
                'value': s} for s in subjects[:30]]
    return options, options[0]['value'] if options else None


@app.callback(
    Output('epoch-slider', 'max'),
    Input('subject-selector', 'value'),
    State('dataset-selector', 'value')
)
def update_epoch_slider(subject, dataset):
    if not subject:
        return 0
    d = DATA[dataset]
    mask = d['subjects'] == subject
    return int(mask.sum()) - 1


@app.callback(
    Output('brain-graph', 'elements'),
    Output('prediction-bar', 'figure'),
    Output('gauge-clustering', 'figure'),
    Output('gauge-pathlength', 'figure'),
    Output('gauge-smallworld', 'figure'),
    Output('gauge-efficiency', 'figure'),
    Output('band-power-chart', 'figure'),
    Output('electrode-heatmap', 'figure'),
    Output('comparison-heatmap', 'figure'),
    Output('info-card', 'children'),
    Output('model-results', 'children'),
    Input('dataset-selector', 'value'),
    Input('connectivity-selector', 'value'),
    Input('subject-selector', 'value'),
    Input('epoch-slider', 'value'),
    Input('threshold-slider', 'value')
)
def update_all(dataset, connectivity, subject, epoch_idx, threshold):
    d = DATA[dataset]
    
    # Get the specific epoch
    mask = d['subjects'] == subject
    indices = np.where(mask)[0]
    if epoch_idx >= len(indices):
        epoch_idx = 0
    idx = indices[epoch_idx]
    
    epoch = d['epochs'][idx]  # (19, 512)
    label = d['labels'][idx]
    
    # Compute connectivity matrix
    conn_funcs = {
        'plv': compute_plv, 'pli': compute_pli,
        'coherence': compute_coherence, 'correlation': compute_correlation
    }
    adj = conn_funcs[connectivity](epoch, sfreq=128.0)
    adj = np.nan_to_num(adj, nan=0.0)
    
    # --- Brain graph ---
    importance = d['importance']
    elements = build_cytoscape_elements(adj, importance, threshold)
    
    # --- Prediction ---
    if 'model' in d:
        from src.graph_construction.brain_graph_builder import epoch_to_pyg_graph
        from torch_geometric.loader import DataLoader as PyGLoader
        
        graph = epoch_to_pyg_graph(epoch, label, connectivity_method=connectivity)
        graph.batch = torch.zeros(19, dtype=torch.long)
        
        with torch.no_grad():
            out = d['model'](graph)
            probs = torch.softmax(out, dim=1)[0].numpy()
    else:
        probs = np.array([0.5, 0.5])
    
    if dataset == 'adhd':
        pred_labels = ['Control', 'ADHD']
    else:
        pred_labels = ['Baseline', 'Task']
    pred_fig = create_prediction_bar(probs, pred_labels)
    
    # --- Graph metrics ---
    gm = compute_global_graph_metrics(adj, threshold)
    g1 = create_metrics_gauge(gm['clustering_coeff'], 'Clustering Coeff')
    g2 = create_metrics_gauge(gm['avg_path_length'], 'Avg Path Length', max_val=5)
    g3 = create_metrics_gauge(gm['small_worldness'], 'Small-Worldness', max_val=5)
    g4 = create_metrics_gauge(gm['global_efficiency'], 'Global Efficiency')
    
    # --- Band power ---
    bp_fig = create_band_power_chart(epoch, sfreq=128.0)
    
    # --- Electrode importance ---
    heatmap_fig = create_brain_heatmap(importance, f"Electrode Importance ({dataset.upper()})")
    
    # --- Comparison: average connectivity for each class ---
    class0_mask = d['labels'] == 0
    class1_mask = d['labels'] == 1
    
    # Sample up to 50 epochs from each class for average
    n_sample = min(50, class0_mask.sum(), class1_mask.sum())
    c0_idx = np.random.choice(np.where(class0_mask)[0], n_sample, replace=False)
    c1_idx = np.random.choice(np.where(class1_mask)[0], n_sample, replace=False)
    
    adj0_avg = np.mean([conn_funcs[connectivity](d['epochs'][i], 128.0) for i in c0_idx[:10]], axis=0)
    adj1_avg = np.mean([conn_funcs[connectivity](d['epochs'][i], 128.0) for i in c1_idx[:10]], axis=0)
    adj0_avg = np.nan_to_num(adj0_avg, nan=0.0)
    adj1_avg = np.nan_to_num(adj1_avg, nan=0.0)
    
    comp_fig = create_comparison_figure(adj0_avg, adj1_avg, connectivity.upper())
    
    # --- Info card ---
    actual = 'ADHD' if label == 1 else 'Control' if dataset == 'adhd' else ('Task' if label == 1 else 'Baseline')
    predicted = pred_labels[np.argmax(probs)]
    correct = '✓' if predicted == actual else '✗'
    
    info = html.Div([
        html.Div([
            html.Span(f"Subject: {subject}", style={'fontWeight': '600', 'marginRight': '16px'}),
            html.Span(f"Epoch: {epoch_idx + 1}/{len(indices)}", style={'marginRight': '16px'}),
            html.Span(f"True: {actual}", style={'marginRight': '16px', 'color': '#059669' if label == 0 else '#dc2626'}),
            html.Span(f"Predicted: {predicted} {correct}",
                       style={'fontWeight': '600', 'color': '#059669' if correct == '✓' else '#dc2626'}),
        ], style={'fontSize': '13px', 'color': '#334155'}),
        html.Div([
            html.Span(f"Connectivity: {connectivity.upper()}", style={'marginRight': '16px'}),
            html.Span(f"Edges: {sum(1 for e in elements if 'source' in e.get('data', {}))}"),
            html.Span(f" | Nodes: 19", style={'marginRight': '16px'}),
            html.Span(f" | Density: {gm['density']:.3f}"),
        ], style={'fontSize': '11px', 'color': '#64748b', 'marginTop': '4px'}),
    ])
    
    # --- Model results table ---
    results = d.get('results', {}).get('models', {})
    results_div = html.Div([
        html.H3("Model Comparison", style={
            'margin': '0 0 8px', 'fontSize': '15px', 'color': '#1e293b'
        }),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Model", style={'textAlign': 'left', 'padding': '4px 8px', 'fontSize': '12px'}),
                html.Th("Acc", style={'textAlign': 'right', 'padding': '4px 8px', 'fontSize': '12px'}),
                html.Th("F1", style={'textAlign': 'right', 'padding': '4px 8px', 'fontSize': '12px'}),
                html.Th("AUC", style={'textAlign': 'right', 'padding': '4px 8px', 'fontSize': '12px'}),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(name, style={'padding': '4px 8px', 'fontSize': '12px', 'fontWeight': '600' if 'GAT' in name else 'normal'}),
                    html.Td(f"{m.get('accuracy', 0):.3f}", style={'textAlign': 'right', 'padding': '4px 8px', 'fontSize': '12px'}),
                    html.Td(f"{m.get('f1', 0):.3f}", style={'textAlign': 'right', 'padding': '4px 8px', 'fontSize': '12px'}),
                    html.Td(f"{m.get('auc', 0):.3f}", style={'textAlign': 'right', 'padding': '4px 8px', 'fontSize': '12px'}),
                ], style={'borderTop': '1px solid #e2e8f0'})
                for name, m in results.items()
            ])
        ], style={'width': '100%', 'borderCollapse': 'collapse'})
    ]) if results else html.Div("No results available")
    
    return elements, pred_fig, g1, g2, g3, g4, bp_fig, heatmap_fig, comp_fig, info, results_div


# =================== RUN ===================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  BrainNetGNN Dashboard")
    print("  http://localhost:8050")
    print("=" * 50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=8050)
