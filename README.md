# 🔒 DeadlockGNN — Real-Time OS Deadlock Prediction with Graph Neural Networks

> **Predicting operating system deadlocks before they happen using a hybrid Relational GCN + GRU architecture, trained on 200,000+ synthetic kernel states, with live monitoring of a real xv6-riscv kernel.**

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture Diagram](#-architecture-diagram)
3. [Data Sources & Dataset Types](#-data-sources--dataset-types)
4. [Feature Engineering: The Resource Allocation Graph](#-feature-engineering-the-resource-allocation-graph)
5. [Models](#-models)
6. [Frontend: The Streamlit Dashboard](#-frontend-the-streamlit-dashboard)
7. [Live xv6 Kernel Monitoring](#-live-xv6-kernel-monitoring)
8. [Metrics & Evaluation](#-metrics--evaluation)
9. [Step-by-Step Execution Commands](#-step-by-step-execution-commands)

---

## 🚀 Project Overview

**DeadlockGNN** is an end-to-end AI system that applies Graph Neural Network techniques to the classic Operating Systems problem of **deadlock detection and prediction**.

### What it Does

The system can operate in two modes:

| Mode | Description |
|------|-------------|
| **Static Snapshot** | Given a single graph snapshot of OS state, predict if a deadlock currently exists |
| **Temporal Forecast** | Given a sequence of 8 historical OS snapshots, predict if the **next** OS state will deadlock |

### Why It's Interesting

Traditional deadlock detection algorithms (like Banker's Algorithm) only work on *current* state and require complete knowledge of future resource needs. DeadlockGNN instead **learns** structural patterns of deadlock from graph topology alone, and can **forecast** deadlocks before they happen — just like how a weather model predicts rain from patterns, not from knowing all future conditions.

---

## 🏗 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│                                                                 │
│  Synthetic OS Simulator ──► 200,000 RAG snapshots (dataset/)   │
│  Temporal Simulator     ──► 50,000 sequences (dataset_temporal/)│
│  xv6-riscv Kernel       ──►  Real-time [GNN_TRACE] log stream  │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GRAPH REPRESENTATION                        │
│                                                                 │
│   Resource Allocation Graph (RAG) — NetworkX DiGraph           │
│   Nodes: Processes (P) + Resources (R)                         │
│   Edges: Request (P→R) | Assignment (R→P)                      │
│   Node Features: 7-dimensional vector per node                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MODEL LAYER                              │
│                                                                 │
│  Static: DeadlockRGCN (3-layer RGCN + MLP Classifier)          │
│  Temporal: TemporalRGCNGRU (RGCN Encoder + 2-layer GRU)        │
│  XAI: Monte-Carlo Shapley Attribution                           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FRONTEND LAYER                             │
│                                                                 │
│   Streamlit Dashboard (demo.py)                                 │
│   Tab 1: Static Snapshot + Shapley XAI                         │
│   Tab 2: Temporal Sequence Animation                            │
│   Tab 3: 🔴 Live xv6 Kernel Monitoring                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Data Sources & Dataset Types

The project uses **three types of data**, each getting progressively more real:

### 1. Synthetic Static Dataset (`dataset/`)

- **Count**: 200,000+ individual graph snapshots (`.pt` files)
- **Generator**: `deadlock_gnn/data/generator.py` — Parameterized RAG generator
- **How it's created**: Randomly places 4-12 processes and 3-10 resources with probabilistic request/assignment edges and controllable deadlock prevalence.
- **Deadlock Label**: Ground-truth label is computed using the **Wait-For Graph (WFG)** algorithm with a DFS cycle detector (`deadlock_gnn/algorithms/wfg.py`).
- **Balance**: ~50/50 deadlocked vs. safe graphs, achieved by rejection sampling during generation.
- **Config reference**: `config.yaml` controls `target_per_class`, `max_capacity`, etc.

### 2. Synthetic Temporal Dataset (`dataset_temporal/`)

- **Count**: 50,000 sequences of 8 time-ordered snapshots
- **Generator**: `deadlock_gnn/data/temporal_generator.py` + `dataset_generator/`
- **How it's created**: An OS simulator runs a Poisson-arrival process scheduler that models real OS behaviors:
  - Process arrival/departure (Poisson with λ=0.4)
  - Burst arrivals (5% probability, spawning up to 5 processes)
  - Resource holding & release with configurable hold durations (3–10 ticks)
  - Priority inversion toggling
  - Configurable multi-capacity resources (up to 5 cores per resource)
- **Format**: Each `.pt` file is a dict `{"graphs": [G_0...G_7], "label": tensor}` where `label` is whether `G_8` (the next step) is deadlocked.
- **Sequence length**: 8 timesteps (configurable in `dataset_generator/config/config.yaml`)

### 3. Real xv6-riscv Kernel Data (Live)

- **Source**: A real RISC-V OS kernel running inside QEMU emulator
- **How it's captured**: Custom `[GNN_TRACE]` instrumentation in `xv6-riscv/kernel/spinlock.c` and `proc.c` emits structured log lines to QEMU's console:
  ```
  [GNN_TRACE] LOCK_ACQUIRE P5 Rproc 1642
  [GNN_TRACE] LOCK_RELEASE P5 Rvirtio_disk 1645
  [GNN_TRACE] PROCESS_CREATE P7 - 1650
  [GNN_TRACE] PROCESS_EXIT P4 - 1658
  ```
- **Events tracked**: `LOCK_ACQUIRE`, `LOCK_RELEASE`, `PROCESS_CREATE`, `PROCESS_EXIT`, `PROCESS_SLEEP`, `PROCESS_WAKE`
- **Bridge**: Logs stream through a UNIX FIFO pipe (`/tmp/xv6_gnn_pipe`) to a background Python listener that parses events and builds a live RAG.

---

## 🧪 Feature Engineering: The Resource Allocation Graph

Every OS state is converted into a **Relational Resource Allocation Graph** before being fed to the GNN.

### Node Types

| Type | Description | Color in Dashboard |
|------|-------------|-------------------|
| Process (P) | An OS process/thread | Blue |
| Resource (R) | A hardware/software resource | Red |

### Edge Types

| Type | Direction | Meaning | Edge Type ID |
|------|-----------|---------|-------------|
| Request | P → R | Process is waiting/requesting this resource | 0 |
| Assignment | R → P | Resource is currently held by this process | 1 |

> **Deadlock Condition**: A deadlock exists when there is a **circular cycle** in the RAG. E.g., P1 → R1 → P2 → R2 → P1.

### Node Feature Vector (7-dimensional)

Each node carries a 7-dimensional feature vector:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `is_process` | 1.0 if Process node, 0.0 if Resource |
| 1 | `is_resource` | 0.0 if Process node, 1.0 if Resource |
| 2 | `norm_degree` | Normalized total degree (in+out) |
| 3 | `in_out_ratio` | in_degree / (out_degree + 1) |
| 4 | `num_request` | Number of request edges incident to this node |
| 5 | `num_assign` | Number of assignment edges incident to this node |
| 6 | `util_ratio` | For resources: holders / capacity (utilization) |

---

## 🧠 Models

### Model 1: `DeadlockRGCN` — Static Snapshot GNN

**File**: `deadlock_gnn/models/rgcn_model.py`  
**Saved as**: `deadlock_rgcn_massive.pt` (trained on 200k graphs) or `deadlock_rgcn_best.pt`

**Architecture**:
```
Input (7-dim node features)
    ↓
RGCNConv Layer 1 (7 → 64, 2 relation types) + ReLU
    ↓
RGCNConv Layer 2 (64 → 64) + ReLU
    ↓
RGCNConv Layer 3 (64 → 64)
    ↓
Global Add Pooling (graph-level embedding)
    ↓
Linear(64 → 32) + ReLU + Dropout(0.3)
    ↓
Linear(32 → 1) → BCEWithLogitsLoss → Sigmoid → Deadlock Probability
```

**Key design choices**:
- **RGCN** (Relational GCN) is used instead of plain GCN because we have **two different edge types** (Request vs. Assignment). Each relation type has its own weight matrix, letting the model learn differently from "P is waiting for R" vs. "R is held by P".
- **Global Add Pooling** aggregates all node embeddings into a single graph-level vector before classification.
- **Gradient Clipping** (max_norm=1.0) prevents exploding gradients during training.
- **Warmup + Cosine LR Scheduler** provides stable training.
- **BCEWithLogitsLoss with pos_weight** handles class imbalance.

**Training**: `python train.py` or `python train_massive.py` (same model, 200k graphs)

---

### Model 2: `TemporalRGCNGRU` — Temporal Forecasting Model

**File**: `deadlock_gnn/models/temporal_rgcn_gru.py`  
**Saved as**: `deadlock_temporal_model.pt`

**Architecture**:
```
Input: Sequence of 8 RAG snapshots [G_0, G_1, ..., G_7]
    ↓ (for each timestep)
Shared RGCN Encoder (same 3-layer RGCN as static model)
    ↓
Global Add Pooling → Graph Embedding Vector e_t (shape: [64])
    ↓
Stack all 8 embeddings: [B, T=8, H=64]
    ↓
2-layer GRU (hidden_size=128, batch_first=True)
    ↓
Last hidden state h_T (shape: [B, 128])
    ↓
Linear(128 → 64) + ReLU + Dropout(0.3)
    ↓
Linear(64 → 1) → Sigmoid → P(deadlock at T+1)
```

**Key design choices**:
- **Transfer Learning**: The RGCN encoder weights are **directly copied** from the pretrained `deadlock_rgcn_massive.pt` (`load_static_weights()` method). This means the temporal model doesn't need to re-learn graph topology from scratch — it inherits the OS intuition from 200k+ graph training runs.
- **2-layer GRU** processes the temporal sequence to learn OS state evolution patterns.
- **Shared Encoder**: The same RGCN encoder processes all timesteps, which keeps the model efficient and forces it to learn a general graph embedding.
- **Predicts T+1**: The model forecasts the *next* OS state, not the current one — this is true predictive capability.

---

### Model 3: XAI — Monte-Carlo Shapley Attribution

**File**: `deadlock_gnn/explain/shapley.py`

When enabled in the UI, this computes **Shapley values** for each node in the graph using Monte-Carlo sampling. This tells you: "Which processes and resources contributed most to the predicted deadlock risk?"

**How it works**: Randomly masks subsets of nodes and measures how much the prediction changes. Nodes that, when removed, drop the risk score the most, get the highest Shapley importance.

---

### Ensemble Detection (`deadlock_gnn/models/ensemble.py`)

The `hybrid_detect()` function combines:
1. **Structural check**: Direct WFG cycle detection (algorithmic, 100% accurate for known graphs)
2. **GNN probability**: Neural network prediction

If a cycle is detected structurally, it's always reported as `DEADLOCK`. The GNN probability provides a **confidence score** and can detect *near-deadlock* situations that the pure algorithm would miss.

---

## 🎛 Frontend: The Streamlit Dashboard

**File**: `demo.py`  
**Launch**: `streamlit run demo.py`

The dashboard has **3 tabs**:

### Tab 1: Static Snapshot & Shapley XAI

**Inputs** (sidebar sliders):
- **Processes**: Number of OS processes to simulate (2–20)
- **Resources**: Number of shareable resources (2–15)
- **Request Probability**: P(a process requests a resource edge)
- **Assignment Probability**: P(a resource is assigned to a process)
- **Show Shapley Importance**: Toggle XAI explanations
- **Top-K Nodes**: How many nodes to highlight in XAI
- **Monte Carlo Samples (T)**: Accuracy vs. speed tradeoff for Shapley

**Outputs**:
- Status badge: `⚠️ DEADLOCK DETECTED` or `✅ SAFE`
- GNN Probability metric (% deadlock risk)
- Detected cycle path (e.g., `P1 → R2 → P3 → R1 → P1`)
- RAG visualization with color-coded nodes and highlighted deadlock cycle
- Shapley importance bar chart (optional)

### Tab 2: Temporal Sequence Animation

**Inputs**:
- **Time-Steps**: How many OS ticks to simulate (3–10)
- **OS Ticks per Step**: Granularity of each snapshot

**Outputs**:
- `Predicted DEADLOCK Probability at T+1` (from the GRU model)
- Warning/safe status for the next OS tick
- A **timeline scrubber** to animate through the OS state evolution
- RAG visualization for each time-step

### Tab 3: Live xv6 Kernel Monitoring 🔴

**Inputs**:
- **xv6-riscv Path**: Directory path to the xv6-riscv repo
- **Start/Stop Monitoring** button

**What happens when you Start Monitoring**:
1. Creates a UNIX FIFO pipe at `/tmp/xv6_gnn_pipe`
2. Launches a new macOS Terminal window via AppleScript running `make qemu | tee /tmp/xv6_gnn_pipe`
3. A background thread reads from the FIFO, parses every `[GNN_TRACE]` line, and updates the live RAG graph
4. The dashboard auto-refreshes every second, running both models on the live graph
5.  The live RAG is visualized in real-time

**Outputs**:
- `Static Deadlock Risk` — Instant RGCN prediction on current kernel state
- `Temporal Forecast (T+1)` — GRU prediction of next tick (after 8 snapshots accumulate)
- `🚨 HIGH DEADLOCK RISK` warning (if >70%)
- Live RAG topology graph

---

## 📊 Metrics & Evaluation

The models are evaluated using full classification metrics on a held-out test set:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of all predicted deadlocks, how many were real |
| **Recall** | Of all real deadlocks, how many we caught |
| **F1 Score** | Harmonic mean of Precision and Recall (primary metric) |
| **AUC-ROC** | Area under the ROC curve (discrimination power) |
| **AUC-PR** | Area under Precision-Recall curve (better for imbalanced) |
| **Inference Latency** | Wall-clock time for full test set evaluation |

### Evaluation Plots Generated

| File | Description |
|------|-------------|
| `roc_curve.png` | ROC curve for static RGCN model |
| `pr_curve.png` | Precision-Recall curve for static RGCN model |
| `roc_curve_massive.png` | ROC curve for massive RGCN model |
| `pr_curve_massive.png` | PR curve for massive RGCN model |
| `pr_curve_temporal.png` | PR curve for temporal GRU model |
| `roc_curve_temporal.png` | ROC curve for temporal GRU model |
| `training_dashboard.png` | Loss + Val F1 curves for static training |
| `training_dashboard_massive.png` | training curves for massive model |
| `training_dashboard_temporal.png` | Training curves for temporal model |
| `confusion_matrix_temporal.png` | Confusion matrix for temporal model |

---

## ⚡ Step-by-Step Execution Commands

> All commands should be run from the project root:
> ```bash
> cd "/Users/rishitkamboj/College/Operating systems /antigravity_monday"
> ```
> Activate the conda environment first:
> ```bash
> conda activate ./.conda
> ```

---

### 🔧 0. Environment Setup

```bash
# Install all dependencies
pip install -r requirements.txt

# requirements.txt includes:
# torch, torch_geometric, streamlit, networkx, scikit-learn, matplotlib, pyyaml
```

---

### 📊 1. Generate the Static Dataset (200k graphs)

```bash
# Generate individual RAG snapshots into dataset/
python dataset_generator/scripts/generate_dataset.py

# OR generate inline (done automatically during training)
# The train.py script generates data on-the-fly from config.yaml
```

---

### 🔄 2. Generate the Temporal Dataset (50k sequences)

```bash
# Generate temporal sequences into dataset_temporal/
python deadlock_gnn/data/temporal_generator.py \
    --dir dataset_temporal \
    --seq_len 8 \
    --n_sequences 50000
```

---

### 🧠 3. Train the Static RGCN Model (Fast, ~3k graphs)

```bash
# Train the base RGCN model (uses config.yaml, ~1500 graphs per class)
python train.py --config config.yaml

# Outputs:
#   deadlock_rgcn_best.pt     ← saved model checkpoint
#   test_dataset.pt            ← test split for evaluation
#   training_dashboard.png     ← loss + F1 curves
```

### 🧠 3b. Train the Massive Static RGCN Model (200k graphs)

```bash
# Train on 200,000 graphs (takes longer but much better generalization)
python train_massive.py

# Outputs:
#   deadlock_rgcn_massive.pt   ← saved model checkpoint
```

---

### ⏱ 4. Train the Temporal GRU Model

```bash
# Train the temporal forecasting model (requires dataset_temporal/ to be populated)
python train_temporal.py \
    --dir dataset_temporal \
    --epochs 30 \
    --batch_size 32 \
    --lr 3e-4 \
    --hidden 64 \
    --gru_hidden 128

# This automatically does transfer learning from deadlock_rgcn_massive.pt
# Outputs:
#   deadlock_temporal_model.pt        ← saved model checkpoint
#   training_dashboard_temporal.png   ← loss + F1 curves
```

---

### 📈 5. Evaluate the Static RGCN Model

```bash
# Evaluate the standard RGCN model
python evaluate.py

# Evaluate the massive RGCN model
python evaluate_massive.py

# Outputs:
#   Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR
#   roc_curve.png, pr_curve.png
```

### 📈 5b. Evaluate the Temporal Model

```bash
python evaluate_temporal.py

# Outputs:
#   F1, AUC, Precision, Recall for temporal sequences
#   roc_curve_temporal.png, pr_curve_temporal.png
#   confusion_matrix_temporal.png
```

---

### 🖥 6. Launch the Streamlit Dashboard

```bash
# Start the interactive demo
streamlit run demo.py

# Access at: http://localhost:8501
# Or whichever port Streamlit picks (8502, 8503, etc. if occupied)
```

---

### 🐧 7. Live xv6 Kernel Monitoring

```bash
# Step 1: Start the Streamlit dashboard (if not already running)
streamlit run demo.py

# Step 2: In the browser, go to Tab 3 "Live xv6 Monitoring"

# Step 3: Click "🚀 Start xv6 Monitoring"
#   → A new macOS Terminal window opens automatically
#   → xv6 builds and boots inside QEMU

# Step 4: Try these stress commands in the xv6 terminal:
stressfs       # Stress file system locks
usertests      # Run all OS test suite (generates lots of lock events)

# Step 5 (optional): To stress-test deadlock detection during normal OS use:
#   Just observe the GNN monitoring the kernel! Every lock/unlock is tracked.

# Step 6: Quit xv6 QEMU when done
# Press: Ctrl+A, then X
```

---

### 🔬 8. Run the Test Suite

```bash
# Run unit tests
pytest tests/ -v

# Run specific test
pytest tests/test_wfg.py -v
```

---

## 📁 Project File Structure

```
.
├── demo.py                          # 🎛 Main Streamlit dashboard
├── train.py                         # 🧠 Static RGCN training (fast, ~3k graphs)
├── train_massive.py                 # 🧠 Static RGCN training (200k graphs)
├── train_temporal.py                # ⏱  Temporal GRU training
├── evaluate.py                      # 📈 Evaluate static RGCN
├── evaluate_massive.py              # 📈 Evaluate massive RGCN
├── evaluate_temporal.py             # 📈 Evaluate temporal model
├── config.yaml                      # ⚙️  Training hyperparameters
│
├── deadlock_gnn/                    # 🧩 Core ML library
│   ├── models/
│   │   ├── rgcn_model.py            # DeadlockRGCN (static)
│   │   ├── temporal_rgcn_gru.py     # TemporalRGCNGRU
│   │   ├── sage_model.py            # DeadlockGNN (GraphSAGE variant)
│   │   └── ensemble.py              # Hybrid detection (algo + GNN)
│   ├── data/
│   │   ├── generator.py             # RAG generator
│   │   └── converter.py             # NetworkX → PyG Data
│   ├── algorithms/
│   │   └── wfg.py                   # Wait-For Graph + DFS cycle detector
│   ├── explain/
│   │   └── shapley.py               # Monte-Carlo Shapley XAI
│   └── viz/
│       └── rag_plot.py              # RAG visualization (matplotlib)
│
├── dataset_generator/               # 🏭 OS simulator for data generation
│   ├── config/config.yaml           # Simulator config (Poisson λ, resources, etc.)
│   ├── process/process_engine.py    # OS scheduler simulation
│   ├── rag/rag_builder.py           # Build RAG from OS state
│   └── converter/pyg_converter.py  # Convert to PyG format
│
├── runtime/                         # 🔴 Live xv6 monitoring
│   ├── runtime_inference.py         # Live inference engine (RGCN + GRU)
│   └── xv6_bridge/
│       ├── xv6_stream_listener.py   # FIFO pipe listener + Terminal launcher
│       ├── event_parser.py          # Parse [GNN_TRACE] log lines
│       └── rag_builder.py           # Incremental RAG from events
│
├── xv6-riscv/                       # 🐧 Instrumented kernel (git submodule)
│   ├── kernel/spinlock.c            # LOCK_ACQUIRE/RELEASE instrumentation
│   ├── kernel/proc.c                # PROCESS_CREATE/EXIT/SLEEP/WAKE instrumentation
│   └── ...
│
├── dataset/                         # 💾 200k+ static graph .pt files
├── dataset_temporal/                # 💾 50k temporal sequence .pt files
│   
├── deadlock_rgcn_best.pt            # 🏆 Saved static model (fast training)
├── deadlock_rgcn_massive.pt         # 🏆 Saved static model (200k training)
├── deadlock_temporal_model.pt       # 🏆 Saved temporal model
└── test_dataset.pt                  # 🧪 Test set (from training split)
```

---

## 🔑 Key Technical Highlights

1. **Real OS Kernel Integration**: The GNN reads from a real kernel — not a simulation — using instrumented spinlocks in xv6-riscv.

2. **Cross-Domain Transfer Learning**: The temporal model directly inherits RGCN weights trained on 200k synthetic graphs, avoiding the need to retrain the spatial encoder. This is the same approach as using a pretrained ImageNet model as a backbone for a new vision task.

3. **Relational GNN**: Regular GNNs treat all edges equally. RGCN (Relational Graph Convolutional Network) uses different weight matrices per edge type, correctly modeling the asymmetric semantics of "waiting for" vs. "holding" in a deadlock scenario.

4. **Wait-For Graph (WFG) Algorithm**: The ground-truth labels are computed using a classical OS algorithm. The GNN learns to replicate (and extend) this algorithm through graph structure learning alone.

5. **Shapley XAI**: Post-hoc explainability shows *which processes and resources* are contributing to the predicted deadlock score — not just that one was predicted.

---

*Built for the Operating Systems course — demonstrating that modern AI can augment classical OS theory.*