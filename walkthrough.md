# Walkthrough: DeadlockGNN Feature Implementation

The `deadlock_gnn.py` monolithic script has been successfully refactored and expanded into a full package architecture mirroring the `DeadlockGNN_Feature_Spec.docx` requirements.

## Changes Made

### 1. Package Structure
Created a structured python package with `deadlock_gnn/`.
*   Moved algorithms, dataset logic, ML models, interpretability tools, and visualization to their respective sub-directories (`algorithms/`, `data/`, `models/`, `explain/`, `viz/`).
*   Created a standard `requirements.txt`.

### 2. Classical Algorithms
*   **WFG Refactoring**: Refactored `deadlock_gnn/algorithms/wfg.py` into `build_wfg` and `detect_cycle_dfs`. It exposes both `is_deadlocked` and explicit `cycle_nodes`. Includes multi-instance wait handling.
*   **Banker's Algorithm**: Implemented standard banker matrices in `deadlock_gnn/algorithms/bankers.py` with `bankers_check` to report safe states and safe sequences. Added `rag_to_banker_matrices` converter.
*   **Ensemble**: Implemented `hybrid_detect` in `models/ensemble.py`, mixing GNN probabilities, WFG ground truth, and Banker's algorithm safe checks.

### 3. ML Architecture
*   **Rich Features & Edge Types**: Added full 8-dimensional node feature mapping (e.g., node type, in/out ratio, resource util ratio, explicit WFG cycle marking) and PyTorch Geometric `edge_type` to differentiate request edges and assignment edges inside `deadlock_gnn/data/converter.py`.
*   **Models**: Implemented `DeadlockRGCN` via `torch_geometric.nn.RGCNConv` to correctly operate on the heterogeneous edges. Preserved the old implementation as `DeadlockGNN (SAGE)` for ablation tracking.
*   **Training Script**: Created the CLI entrypoint `train.py` which reads from `config.yaml`. Supports Stratified K-fold split (`test_train_split`), Cosine Annealing Learning Rate, Gradient Clipping, and loss logging.
*   **Evaluation Endpoint**: Built `evaluate.py` to test the best `.pt` checkpoints against held-out datasets. Emits full ROC-AUC, F1, PR-AUC, Accuracy, Precision, Recall, and Confusion Matrices.

### 4. Interpretability & App
*   **Explain**: Added Monte Carlo Shapley Value computations (`shapley_attribution`) and PyG Explainer wrapping (`get_explainer`) for highlighting contributing node topologies.
*   **Visualization**: Setup `viz/rag_plot.py` allowing custom node importance tracking alongside NetworkX RAG visualizations.
*   **Demo**: Developed an interactive Streamlit Application `demo.py`. Users can dynamically adjust graphs using sliders, click generate, and automatically run the `hybrid_detect` to visually showcase cycles and deadlock probabilities.

## Verification Run

*   Executed the `pytest` unit suites on the classic tests (e.g. `tests/test_bankers.py`, `tests/test_wfg.py`). All mathematical algorithms resolve their toy cases successfully.
*   Executed an initial `train.py` run processing graphs and generating checkpoints. (Note: Adjusting `max_capacity` parameter may be necessary to increase synthetic deadlock frequency in future experiments!).
*   Executed `evaluate.py` which securely loads the RGCN checkpoints, config states, and resolves ROC metrics seamlessly.

> [!TIP]
> You can now test the interactive visualizer via: `streamlit run demo.py`!
