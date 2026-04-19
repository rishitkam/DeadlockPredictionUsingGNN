# AI Handoff Context: DeadlockPredictionUsingGNN

**Target Audience:** Another AI coding assistant (or human developer) taking over or extending this repository.

## 📌 Project Overview
This project predicts operating system deadlocks in Resource Allocation Graphs (RAG) using a hybrid approach combining classical deterministic OS algorithms (Wait-For Graph DFS, Banker's Algorithm) with deep learning (PyTorch Geometric `RGCN` models). 

The goal was to move from a monolithic script into a scalable, production-ready package architecture capable of extrapolating deadlock probabilities in multi-instance resource systems.

---

## 🛠 Project Journey & Milestones (Conversation Summary)

### Phase 1: Architectural Refactor
- Started with a single monolithic `deadlock_gnn.py` file.
- The AI broke the codebase down into highly modular packages:
  - `algorithms/`: Mathematical ground truths (`bankers.py`, `wfg.py`)
  - `data/`: Random RAG generation with multi-instances (`generator.py`) and PyG tensor converting (`converter.py`).
  - `models/`: Classical GCN (`sage_model.py`), Relational GCN (`rgcn_model.py`), and the Fusion architecture (`ensemble.py`).
  - `explain/`: SubgraphX wrappers & Monte Carlo Shapley explanations (`shapley.py`).
  - `viz/`: NetworkX graph visualizations.

### Phase 2: Feature Engineering & Multi-Instance Resources
- Implemented **heterogeneous edge types** (Type 0 for Request edges mapping $P \rightarrow R$, Type 1 for Assignment edges mapping $R \rightarrow P$).
- Implemented `max_capacity` capabilities. The OS simulator generated resources with $K>1$ slots.
- **Bug Discovery**: Setting $K=3$ universally made deadlocks mathematically too rare during training (an imbalanced dataset of 8000 safe graphs, 0 deadlocked). *Resolution:* Re-seeded `max_capacity: 1` in `config.yaml` to ensure enough cycle instances generated to properly train the network.

### Phase 3: The Data Leakage Fix 
- **The Issue**: Early iterations of the model predicted output mathematically between exactly $0.0\%$ and $100.0\%$. The AI discovered massive **Data Leakage**. In the initial feature spec, the 8th dimension of the node vector was the WFG `is_in_cycle` ground-truth logic. The RGCN learned to cheat using just index [7].
- **The Fix**: The AI stripped the leakage tag (`in_cycle`) out of the `converter.py` logic, dropping the topological mapping to 7-dimensions (processes, resources, request-edges, util-ratios, etc.) without handing it the answer key. 
- **Results**: The model naturally learned cycle convolutions, scoring an impressive 98.01% F1 and 0.9984 AUC-ROC.

### Phase 4: CI/CD & Deployments
- Created `train.py` (which logs `training_dashboard.png`) and `evaluate.py` (which plots `roc_curve.png` and `pr_curve.png`).
- Handled macOS local environment GitHub credentials switching the Git Origin from SSH to HTTPS.
- Wrote extensive UI updates to `demo.py` utilizing Streamlit mapping hybrid outputs perfectly.

### Phase 5: Technical Discussions
The user and AI discussed theoretical future steps:
- Why generative random-seeding creates drastic variations in consecutive UI clicks.
- The realities of Cloud Datasets vs Code-Property Graphs (Joern, CPGs).
- Methods on integrating the Python RGCN inference agent inside the minimalistic C-based **XV6 kernel**.

---

## 🏗 Current State of Codebase

* **Dataset:** Generated synthetically inside Python. 7-dimensional node embeddings. 
* **Model:** Best pipeline is `deadlock_rgcn_best.pt` stored locally after execution.
* **Streamlit:** Highly stable UI via `demo.py`.
* **Git Context:** Changes are synced securely up to the `rishit` origin branch on GitHub (`rishitkam/DeadlockPredictionUsingGNN`).

---

## 🚦 Next Steps / Prompt Ideas for the Next AI
1. **Real-World Trace Integrations:** Implementing data processors mapping real trace datasets (Google/Alibaba Cluster Traces, DeadlockBench, Joern CPGs) into the `converter.py` PyG structures for authentic production training.
2. **Temporal Sequences (Ablation):** The generator currently yields *static* snapshots. The next jump involves time-series RAG evolution using GRUs (updating node states dynamically tick-by-tick).
3. **XV6 Integration (C/Python API):** Building a serial-port host watcher that scans XV6 `proc.c` outputs, builds the RAG locally using this pipeline, and pipes a `kill` OS trap back into QEMU.
4. **Advanced UI Explanations:** Actively drawing the Shapley Values inside the `.png` Node Visualizer in `demo.py`.
