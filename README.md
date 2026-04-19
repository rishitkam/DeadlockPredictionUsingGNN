# 🔒 DeadlockPredictionUsingGNN
### A Production-Grade, Research-Quality OS Deadlock Prediction Framework using Graph Neural Networks

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.x-orange)](https://pyg.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_Demo-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚀 What This Project Is

This repository implements a **complete, end-to-end OS Deadlock Prediction Research System** built around **Graph Neural Networks (GNNs)**, specifically **Relational Graph Convolutional Networks (RGCNs)** and a novel **Temporal RGCN + GRU hybrid architecture**.

Instead of making toy examples, this project simulates a **real operating system environment** — with Poisson-distributed process arrivals, FIFO resource queues, multi-instance resources, and hold-cycle dynamics — and trains a neural network to detect and **predict future deadlocks** with **98.16% accuracy and a 0.9990 AUC-ROC** on 200,000 simulated OS states.

The system spans three major capabilities that together form a coherent OS research systems framework:

> 1. **Static Deadlock Detection** using RGCN on Resource Allocation Graph snapshots  
> 2. **Temporal Deadlock Forecasting** using RGCN + GRU on evolving RAG sequences  
> 3. **Explainable AI Visualizations** using Monte Carlo Shapley Node Attribution inside a live Streamlit demo  

---

## 🏆 Key Achievements

| Metric | Value |
|---|---|
| Static Model Accuracy (200k dataset) | **98.16%** |
| Static F1 Score | **97.77%** |
| Temporal Accuracy (50k sequence dataset) | **91.53%** |
| Temporal F1 Score | **89.55%** |
| AUC-ROC (Static / Temporal) | **0.9990** / **0.9667** |
| Inference Latency | **0.0007s** (Graph) / **0.0028s** (Sequence) |
| Dataset Generation Speed | **~1,100 graphs/second** |
| Temporal Sequences Generated | **50,000** sequences @ 974 seq/sec |
| Temporal Deadlock Balance | **42% deadlock rate** (near-ideal for ML) |

---

## 🖥 Multi-Tab Streamlit Interactive Application

To demonstrate the real-world application of this research, the repository ships with a **Production-Grade Streamlit Web Application** (`demo.py`) featuring two distinct analytical modes:

### Mode 1: Static Snapshot & Shapley XAI
Instead of outputting simple confidence integers, the UI integrates **Monte Carlo Shapley Node Attributions**. When the model predicts an OS deadlock, the interface dynamically paints a heatmap directly onto the topological RAG graphic (via a `steelblue → darkorange → crimson` color map), visually flagging **exactly which OS processes are mechanically responsible for the crash**.
*This bridges the gap between Black-Box neural networks and actionable Kernel-level interventions.*

### Mode 2: Temporal Sequence Animation
When engaging the "Temporal Sequence" tab, Streamlit natively spins up the Python `OSEngine` in memory. It simulates a raw OS timeline, captures 4 consecutive sequential snapshots, chains them into a PyTorch Geographic multidimensional tensor, and runs them through the **Recurrent GRU**. It natively outputs the *future trajectory probability* at `T+1` and provides an interactive slider to manually scrub through the chronological timeline frame-by-frame. 

---

## 🧠 Machine Learning Architecture

### 1. Static RGCN Model (`deadlock_gnn/models/rgcn_model.py`)

The **Relational Graph Convolutional Network (RGCN)** handles heterogeneous edge types — `request` edges (Process → Resource) and `assignment` edges (Resource → Process) — meaning each edge type gets its own learned weight matrix. This is fundamentally more powerful than standard GCNs which treat all edges identically.

```
Input: 7-dimensional node feature vectors
  ↓ RGCNConv Layer 1 (→ 64 channels)
  ↓ RGCNConv Layer 2 (→ 64 channels)  
  ↓ RGCNConv Layer 3 (→ 64 channels)
  ↓ Global Add Pooling (graph-level embedding)
  ↓ Linear(64 → 32) + ReLU + Dropout
  ↓ Linear(32 → 1) + Sigmoid
Output: P(deadlock) ∈ [0, 1]
```

### 2. Temporal RGCN + GRU Model (`deadlock_gnn/models/temporal_rgcn_gru.py`)

The temporal model takes **4 consecutive RAG snapshots** of an evolving OS state and predicts whether the next OS tick will produce a deadlock — a true **time-series forecasting** problem over graphs.

```
For each timestep t in [t-3, t-2, t-1, t]:
    G_t → RGCN Encoder → Global Pooling → embedding_t (64-dim)

[embedding_t-3, embedding_t-2, embedding_t-1, embedding_t]
    → GRU (2 layers, hidden=128)
    → Final hidden state
    → Linear(128 → 64) + ReLU + Dropout
    → Linear(64 → 1) + Sigmoid
Output: P(deadlock at t+1) ∈ [0, 1]
```

### 3. Hybrid Ensemble (`deadlock_gnn/models/ensemble.py`)

Before calling the GNN, the system runs a **classical Banker's Algorithm safety check**. If the state is provably safe, it outputs `SAFE` with high confidence immediately, skipping the neural path. If the classical algorithm is uncertain, it defers to the GNN. This hybrid approach is more trustworthy than pure ML.

### 4. Monte Carlo Shapley Explainer (`deadlock_gnn/explain/shapley.py`)

Implements **Monte Carlo approximation of Shapley Values** from cooperative game theory. For each node in the graph, it computes that node's marginal contribution to the deadlock prediction score across `T` random permutations. This provides rigorous, theoretically grounded explanations of *why* the model predicted a deadlock.

---

## 📡 How Data Is Generated

### Why Synthetic Data?

Real OS schedulers silently terminate deadlocked processes. This means real OS logs structurally **cannot** contain deadlock topology because the kernel destroys it before it can be logged. Every major academic paper on RAG-based ML (including work from MIT, CMU, and ETH Zurich) uses synthetic simulation for exactly this reason.

### The OS Simulation Engine (`dataset_generator/`)

The generator is not a naïve random graph producer — it is a **discrete-time OS simulator** that mechanically enforces real operating system constraints:

- **Poisson Process Arrivals**: New processes arrive according to `λ=0.4` rate parameter, matching realistic cloud workload burst distributions
- **Resource Lifecycle**: Processes go through `ready → running → waiting → finished` state transitions
- **FIFO Queue Blocking**: Processes that cannot acquire a resource join a waiting queue, exactly as Linux `wait_queue_head_t` works
- **Hold Duration Cycles**: Processes hold acquired resources for `3–10 ticks` before releasing them, causing circular wait conditions to form organically
- **Multi-Instance Resources**: Resources have capacity `1–5`, allowing multiple processes to hold the same resource simultaneously (matching mutexes with multiple permits)
- **Burst Probability**: A 5% chance per tick of spawning a burst of up to 5 additional processes, simulating real-world traffic spikes

### Ground-Truth Labeling

After each simulation, the system extracts the OS state and runs **two independent classical OS algorithms** to determine the label:

**Wait-For-Graph (WFG) DFS Cycle Detection**
Converts the bipartite RAG (Process↔Resource graph) into a pure Process→Process WFG by collapsing resources, then runs directed DFS to detect cycles. A cycle = deadlock.

**Banker's Algorithm Safety Check**
Checks whether there exists a safe execution sequence for all waiting processes given remaining resource capacities.

### Node Feature Vector (7 dimensions)

| Index | Feature | Description |
|---|---|---|
| 0 | `is_process` | Binary flag |
| 1 | `is_resource` | Binary flag |
| 2 | `norm_degree` | Normalized total degree |
| 3 | `in_out_ratio` | in_deg / (out_deg + 1) |
| 4 | `request_count` | Number of request edges |
| 5 | `assignment_count` | Number of assignment edges |
| 6 | `resource_utilization` | allocated / capacity |

> **Note:** The `is_in_cycle` ground-truth feature was explicitly removed to eliminate data leakage. The model was forced to learn structural patterns, not read the answer directly.

---

## 📁 Full Project Structure

```
DeadlockPredictionUsingGNN/
│
├── deadlock_gnn/                  ← Core GNN package
│   ├── algorithms/
│   │   ├── wfg.py                 ← Wait-For-Graph cycle detection
│   │   └── bankers.py             ← Banker's Algorithm (safe state check)
│   ├── data/
│   │   ├── generator.py           ← Static RAG generator (heuristic, small runs)
│   │   ├── converter.py           ← NetworkX → PyTorch Geometric converter
│   │   └── temporal_generator.py  ← 🆕 Temporal sequence generator
│   ├── models/
│   │   ├── rgcn_model.py          ← Static RGCN model (3-layer)
│   │   ├── sage_model.py          ← Original SAGEConv baseline
│   │   ├── ensemble.py            ← Hybrid Banker's + GNN detector
│   │   └── temporal_rgcn_gru.py   ← 🆕 Temporal RGCN + GRU model
│   ├── explain/
│   │   ├── shapley.py             ← Monte Carlo Shapley attribution
│   │   └── subgraphx.py           ← GNNExplainer wrapper
│   └── viz/
│       └── rag_plot.py            ← 🆕 Shapley-colormap RAG visualizer
│
├── dataset_generator/             ← Production OS simulator
│   ├── config/config.yaml         ← All simulation parameters
│   ├── workload/workload_generator.py  ← Poisson arrival engine
│   ├── process/
│   │   ├── process.py             ← Process state machine
│   │   └── process_engine.py      ← OS scheduler simulation
│   ├── resources/
│   │   ├── resource.py            ← Multi-instance resource
│   │   └── resource_manager.py    ← Allocator + queue manager
│   ├── rag/rag_builder.py         ← NetworkX RAG assembler
│   ├── algorithms/
│   │   ├── wait_for_graph.py      ← WFG (for generator labeling)
│   │   └── bankers_algorithm.py   ← Banker's (for generator labeling)
│   ├── converter/pyg_converter.py ← RAG → PyG tensor
│   ├── dataset/dataset_writer.py  ← Disk-streaming .pt writer
│   └── scripts/generate_dataset.py ← 🚀 Main generation entrypoint
│
├── tests/
│   ├── test_wfg.py                ← Unit tests for WFG
│   └── test_bankers.py            ← Unit tests for Banker's
│
├── train.py                       ← Static training (in-memory, small datasets)
├── train_massive.py               ← Static training (disk-streaming, 200k+)
├── train_temporal.py              ← 🆕 Temporal training (RGCN+GRU)
├── evaluate.py                    ← Evaluation (original dataset)
├── evaluate_massive.py            ← Evaluation (200k disk dataset)
├── evaluate_temporal.py           ← 🆕 Evaluation (temporal sequence dataset)
├── demo.py                        ← 🆕 Multi-Tab Streamlit UI with Shapley XAI
│
├── ── LEGACY / REFERENCE FILES ──
├── deadlock_gnn.py                ← Original monolithic script (DO NOT use for training)
├── config.yaml                    ← Legacy config (superseded by dataset_generator/config/)
│
├── deadlock_rgcn_best.pt          ← Model: trained on 3k in-memory graphs
├── deadlock_rgcn_massive.pt       ← Model: trained on 200k disk graphs ✅ (primary)
├── deadlock_temporal_model.pt     ← Model: temporal RGCN+GRU
│
└── README.md
```

> **Legacy files note:** `deadlock_gnn.py` was the original monolithic prototype. It is preserved for academic reference but has been replaced entirely by the modular `deadlock_gnn/` package. `config.yaml` at the root is similarly superseded by `dataset_generator/config/config.yaml`.

---

## ⚡ Exact Parameters Used

### Dataset Generation (Static)
```yaml
simulation:
  steps: 50
  poisson_lambda: 0.4
  seed: 42

os_environment:
  number_of_resources: 15
  max_resource_capacity: 5
  max_process_burst: 5
  burst_probability: 0.05

process_behavior:
  min_requests_per_process: 2
  max_requests_per_process: 6
  hold_duration_min: 3
  hold_duration_max: 10

generator:
  dataset_size: 200000
  num_workers: 8
```

### Static RGCN Training
```
Epochs          : 33 (early stop)
Batch Size      : 128
Learning Rate   : 0.005
Hidden Channels : 64
Dropout         : 0.3
Optimizer       : Adam
Loss            : BCEWithLogitsLoss
Train/Val Split : 80/20
```

### Temporal Dataset Generation
```yaml
temporal:
  sequence_length: 4       # G_t-3, G_t-2, G_t-1, G_t
  dataset_size: 50000
  snapshot_interval: 5     # ticks between snapshots
  num_workers: 8
```
Result: **50,000 sequences** in **51.3 seconds** at **974 sequences/sec**. Deadlock rate: **42%** (balanced).

### Temporal RGCN+GRU Training
```
Epochs          : 10
Batch Size      : 32
Learning Rate   : 3e-4 (with ReduceLROnPlateau)
RGCN Hidden     : 64
GRU Hidden      : 128
GRU Layers      : 2
Dropout         : 0.3
Grad Clip       : 1.0
```

---

## 🛠 How to Run Everything

### 1. Install Dependencies
```bash
pip install torch torch_geometric networkx scikit-learn streamlit matplotlib pyyaml
```

### 2. Generate the Static 200k Dataset
```bash
python dataset_generator/scripts/generate_dataset.py
# Output: dataset/graph_000000.pt ... graph_199999.pt
```

### 3. Train on the Massive Dataset
```bash
python train_massive.py --dir dataset --epochs 5 --batch_size 128
# Output: deadlock_rgcn_massive.pt, training_dashboard_massive.png
```

### 4. Evaluate the Static Model
```bash
python evaluate_massive.py
# Output: Accuracy, F1, AUC-ROC, Confusion Matrix, ROC+PR curve PNGs
```

### 5. Generate Temporal Sequences
```bash
python deadlock_gnn/data/temporal_generator.py
# Output: dataset_temporal/seq_000000.pt ... seq_049999.pt
```

### 6. Train the Temporal RGCN+GRU
```bash
python train_temporal.py --dir dataset_temporal --epochs 10 --batch_size 32
# Output: deadlock_temporal_model.pt, training_dashboard_temporal.png
```

### 7. Evaluate the Temporal Model
```bash
python evaluate_temporal.py
# Output: Evaluates the sequence matrices producing 91.53% Accuracy F1, Confusion Matrix
```

### 8. Launch the Interactive Streamlit Demo
```bash
streamlit run demo.py
```
In the UI:
- **Tab 1: Static Snapshot & Shapley Analysis**: Generates OS snapshot and overlays XAI heatmap
- **Tab 2: Temporal Sequence Animation**: Animates timeline frames running predictive GRU loop
- Adjust parameters to manually scrub frames

### 9. Run Unit Tests
```bash
pytest tests/
```

---

## 🔬 Research Context

Standard OS deadlock detection is a solved problem algorithmically (WFG, Banker's). The research contribution here is:

1. **Scalability**: Classical algorithms run in O(V+E) per query. The RGCN learns a compressed latent representation that amortizes this cost over millions of training graphs.
2. **Generalization**: A trained model can detect emergent deadlock *patterns* that no single classical algorithm was explicitly designed to recognize.
3. **Temporal Forecasting**: No classical algorithm predicts *future* deadlocks from historical OS evolution. This is purely a machine learning contribution.
4. **Explainability**: Pure classical algorithms tell you *if* a deadlock exists. Shapley attribution tells you *which process is most responsible* — critical for root cause analysis.

---

## 📊 Results Summary

```
================================
STATIC MODEL (200k Graphs)
================================
Accuracy  : 98.16%
Precision : 97.93%
Recall    : 97.62%
F1 Score  : 97.77%
AUC-ROC   : 0.9990
Latency   : 0.0007s per graph

Confusion Matrix (40,000 Test graphs):
  TN: 23,083   FP: 342
  FN: 395      TP: 16,180

================================
TEMPORAL SEQUENCE GRU (10k Sequences)
================================
Accuracy  : 91.53%
Precision : 92.60%
Recall    : 86.70%
F1 Score  : 89.55%
AUC-ROC   : 0.9667
Latency   : 0.0028s per sequence

Confusion Matrix (10,000 Test sequences):
  TN: 5,523    FP: 290
  FN: 557      TP: 3,630
```

---

*Built with PyTorch Geometric, NetworkX, Streamlit, and a lot of OS theory.*