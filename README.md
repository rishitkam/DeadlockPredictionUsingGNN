# DeadlockPredictionUsingGNN

A comprehensive suite for generating, verifying, and predicting synchronization deadlocks in Resource Allocation Graphs (RAG) using classical Operating System algorithms combined with Relational Graph Convolutional Networks (RGCN).

Based on specifications adapted from modern architecture research (DRIP Framework, Code Property Graph GNNs), this project treats deadlock detection as a heterogeneous graph classification problem.

---

## 📖 What This Code Does

1. **Simulates Operating System States**: Generates synthetic multi-process, multi-resource Allocation Graphs mapping dynamically assigning and requesting behaviors.
2. **Classical OS Validation**: Implements structural Wait-For Graph (WFG) traversals and safe-state checking vectors (Systematic Banker's Algorithm) to deterministically judge the ground-truth stability of the state. 
3. **Machine Learning Extrapolation**: Employs PyTorch Geometric (PyG) and RGCNs to predict deadlocks instantaneously without relying purely on brute-force cycle detection, learning from an 8-dimensional node topographical feature space.
4. **Hybrid Ensembling & Explainability**: Couples the Machine Learning models back with classical checks to yield high-confidence "Hybrid" tags, while offering Monte Carlo Shapley values and SubgraphX extraction to decipher *why* the GNN made its decision.

---

## 📈 The Dataset & Features Environment

There is no external `.csv` or downloaded dataset. The pipeline is **completely self-sustaining** through randomized graph generation simulating real environments!

### How the Data is Generated (`data/generator.py`):
- During training, the command calls `generate_rag()`. It randomly instantiates $P$ processes and $R$ resources.
- It iterates across all resources and assigns **"Assignment Edges"** ($R \rightarrow P$) based on fixed assignment probabilities.
- It concurrently generates **"Request Edges"** ($P \rightarrow R$) matching a request probability.
- If `max_capacity > 1`, resources handle multiple concurrent holders (up to the capacity barrier).

### How the AI Learns It (`data/converter.py`):
Before hitting the neural network, the raw graph is transformed into a `torch_geometric` native object:
- **Ground Truth Calculation**: A localized `wfg.py` detects if the generated graph actually cycles. If yes, `label = 1.0`, else `0.0`.
- **8-Dim Node Vector**: Each node compiles a fingerprint array:
  1. `is_process` (binary)
  2. `is_resource` (binary)
  3. `normalized degree` (load)
  4. `in-degree / (out-degree + 1)`
  5. `number of request edges`
  6. `number of assignment edges`
  7. `resource utilisation ratio` (`holders / capacity`)
  8. `is_in_cycle` (ground-truth WFG cycle membership marker)
- **Heterogeneous Edges**: The RGCN model splits edge aggregations into separate matrices depending on the newly assigned tensor: `0` for Request, `1` for Assignment.

---

## 🗂 Codebase Structure

* `config.yaml` 🎛️: Core control file toggling learning rates, batch sizes, epochs, and resource capacity constraints.
* `train.py` 🧠: End-to-end dataset generation, stratifying into K-folds, Cosine LR training, and model checkpointing (`.pt` emission).
* `evaluate.py` 📊: Standalone inference script to measure AUC-ROC, Precision/Recall, and map evaluation visualizations directly to `.png` files.
* `demo.py` 🖥️: Streamlit frontend.
* **`deadlock_gnn/`**:
  * `algorithms/`: Standard OS concepts (`bankers.py`, `wfg.py`).
  * `data/`: RAG `generator.py` and PyG `converter.py`.
  * `models/`: ML Topologies (`rgcn_model.py`, `sage_model.py`) and the unified classifier (`ensemble.py`).
  * `explain/`: Neural interpretation frameworks (`shapley.py`, `subgraphx.py`).
  * `viz/`: Graphic rendering utilities (`rag_plot.py`).
* **`tests/`** 🧪: PyTest suites covering the mathematical purity of the Classical OS functions.

---

## 🚀 Execution & Running Instructions

### 1. Installation 
Ensure you are using Python 3.10+ and install all primary dependencies:
```bash
pip install -r requirements.txt
```

### 2. Verify Core Mechanics (Testing)
Run the isolated `pytest` suite to verify that the Banker's Algorithm handles allocations properly and WFG accurately catches deadlock loops.
```bash
python -m pytest tests/
```

### 3. Generate & Train the Model
You must assemble a model checkpoint before moving to analysis. You can tweak properties (like `epochs` or `seed`) natively inside `config.yaml` before running:
```bash
python train.py
```
*\*This emits mathematically balanced graphs internally, saves the model to `deadlock_rgcn_best.pt`, isolates test graphs to `test_dataset.pt`, and graphs `training_dashboard.png`.*

### 4. Evaluate the Metrics
Process the held-out test datasets.
```bash
python evaluate.py
```
*\*This dumps all terminal evaluation states (F1, AUC, Latency) and dynamically plots `roc_curve.png` and `pr_curve.png` locally in folder.*

### 5. Launch the Visual Interactive Web Demo
Interact precisely with sliders mapping parameters mathematically equivalent to the simulated configurations:
```bash
streamlit run demo.py
```