"""
Deadlock Detection via Graph Neural Networks
OS Project
"""

import random

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_add_pool

# ==========================================
# 1. SYNTHETIC DATA GENERATION
# ==========================================


def is_deadlocked(G: nx.DiGraph, resources: list[str]) -> bool:
    """
    Detects deadlock using the Wait-For Graph (WFG) approach.

    A deadlock exists if there is a cycle in the WFG. The WFG is derived
    from the RAG by collapsing resource nodes: if process P1 waits for
    resource R, and R is held by P2, then P1 -> P2 in the WFG.

    This is more OS-accurate than simply checking for cycles in the raw RAG,
    because it correctly handles the single-instance resource assumption.
    """
    wfg = nx.DiGraph()
    # Edge conventions in the RAG:
    #   Assignment edge: R -> P  (so P appears in G.successors(R), i.e. G.out_edges(R))
    #   Request edge  : P -> R  (so P appears in G.predecessors(R), i.e. G.in_edges(R))
    for r in resources:
        holders = [v for _, v in G.out_edges(r)]  # R -> P: P holds R
        waiters = [
            u
            for u, _ in G.in_edges(r)  # P -> R: P requests R
            if u not in holders
        ]

        # If no one holds the resource, no wait-for edge can be formed
        if not holders:
            continue

        # Each waiter must wait for every holder (single-instance: one holder max)
        for waiter in waiters:
            for holder in holders:
                if waiter != holder:
                    wfg.add_edge(waiter, holder)

    try:
        next(nx.simple_cycles(wfg))
        return True
    except StopIteration:
        return False


def generate_rag(
    num_processes: int,
    num_resources: int,
    p_request: float = 0.25,
    p_assign: float = 0.25,
) -> tuple[nx.DiGraph, float]:
    """
    Generates a single directed Resource Allocation Graph (RAG).

    Edge conventions (standard OS notation):
      Request edge : P -> R  (process requests resource)
      Assignment edge: R -> P  (resource assigned to process)
    """
    G = nx.DiGraph()
    processes = [f"P{i}" for i in range(num_processes)]
    resources = [f"R{i}" for i in range(num_resources)]

    G.add_nodes_from(processes, node_type="process")
    G.add_nodes_from(resources, node_type="resource")

    for r in resources:
        assigned_to = (
            None  # Enforce single-instance: each resource held by at most one process
        )
        shuffled_procs = processes[:]
        random.shuffle(shuffled_procs)

        for p in shuffled_procs:
            rand = random.random()
            if rand < p_assign and assigned_to is None:
                G.add_edge(r, p)  # Assignment edge: R -> P
                assigned_to = p
            elif rand < p_assign + p_request:
                if not G.has_edge(r, p):  # Don't request what you already hold
                    G.add_edge(p, r)  # Request edge: P -> R

    label = 1.0 if is_deadlocked(G, resources) else 0.0
    return G, label


# ==========================================
# 2. CONVERSION TO PYTORCH GEOMETRIC
# ==========================================


def convert_to_pyg_data(nx_graph: nx.DiGraph, label: float) -> Data:
    """
    Converts a NetworkX RAG to a PyG Data object.

    Node features (3-dim):
      [1, 0, deg] for process nodes  (deg = normalized degree)
      [0, 1, deg] for resource nodes
    The degree feature gives the GNN a lightweight structural hint.
    """
    nodes = list(nx_graph.nodes)
    mapping = {node: i for i, node in enumerate(nodes)}
    max_deg = max(dict(nx_graph.degree()).values(), default=1)

    x = []
    for node in nodes:
        ntype = nx_graph.nodes[node].get("node_type")
        deg = nx_graph.degree(node) / max_deg  # Normalize to [0, 1]
        if ntype == "process":
            x.append([1.0, 0.0, deg])
        else:
            x.append([0.0, 1.0, deg])
    x = torch.tensor(x, dtype=torch.float)

    edges = [(mapping[u], mapping[v]) for u, v in nx_graph.edges()]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))


# ==========================================
# 3. GRAPH NEURAL NETWORK MODEL
# ==========================================


class DeadlockGNN(nn.Module):
    """
    3-layer GraphSAGE network for binary graph classification.

    SAGEConv is used instead of GCNConv because:
      - It aggregates neighbor features without symmetric normalization,
        preserving the directional semantics of the RAG edges.
      - It handles variable-degree nodes more robustly.

    global_add_pool is used instead of mean_pool because the *count* of
    active nodes (requesting/waiting processes) is informative for deadlock.
    """

    def __init__(
        self, in_channels: int = 3, hidden_channels: int = 64, dropout: float = 0.5
    ):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_add_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


# ==========================================
# 4. TRAINING & EVALUATION HELPERS
# ==========================================


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch).view(-1)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.sigmoid(out).round()
        correct += (preds == data.y).sum().item()
        total += len(data.y)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for data in loader:
        data = data.to(device)
        out = torch.sigmoid(model(data.x, data.edge_index, data.batch).view(-1))
        all_probs.extend(out.cpu().tolist())
        all_preds.extend(out.round().cpu().tolist())
        all_labels.extend(data.y.cpu().tolist())
    return all_labels, all_preds, all_probs


def print_metrics(labels, preds, split="Test"):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    print(f"\n--- {split} Results ---")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall   : {rec * 100:.2f}%")
    print(f"  F1 Score : {f1 * 100:.2f}%")
    print(f"  Confusion matrix (TN FP / FN TP):\n  {cm}")
    print("-" * 30)


# ==========================================
# 5. MAIN
# ==========================================

if __name__ == "__main__":
    SEED = 89
    random.seed(SEED)
    torch.manual_seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- Dataset generation ---
    print("\n1. Generating synthetic RAG dataset...")
    dataset = []
    num_deadlocked = 0

    target_per_class = 2000

    for _ in range(10000):  # More graphs for a better distribution sample
        p_count = random.randint(4, 12)
        r_count = random.randint(3, 10)
        g, label = generate_rag(p_count, r_count)
        if len(g.edges) > 0:
            if label == 1.0 and num_deadlocked >= target_per_class:
                continue
            if label == 0.0 and (len(dataset) - num_deadlocked) >= target_per_class:
                continue
            dataset.append(convert_to_pyg_data(g, label))
            num_deadlocked += int(label)

    total = len(dataset)
    print(f"  Total valid graphs : {total}")
    print(
        f"  Deadlocked         : {num_deadlocked} ({num_deadlocked / total * 100:.1f}%)"
    )
    print(
        f"  Safe               : {total - num_deadlocked} ({(total - num_deadlocked) / total * 100:.1f}%)"
    )

    # Compute positive weight for BCEWithLogitsLoss to handle class imbalance
    num_safe = total - num_deadlocked
    pos_weight = torch.tensor(
        [num_safe / max(num_deadlocked, 1)], dtype=torch.float
    ).to(DEVICE)

    # --- Train/val/test split (70/15/15) ---
    random.shuffle(dataset)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)
    train_ds = dataset[:train_end]
    val_ds = dataset[train_end:val_end]
    test_ds = dataset[val_end:]

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # --- Model setup ---
    print("\n2. Initializing model...")
    model = DeadlockGNN(in_channels=3, hidden_channels=64, dropout=0.5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Training loop with early stopping ---
    print("\n3. Training...")
    best_val_f1 = 0.0
    patience = 10
    patience_count = 0
    best_state = None
    epochs = 60

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        scheduler.step()

        val_labels, val_preds, _ = evaluate(model, val_loader, DEVICE)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        if (epoch) % 5 == 0:
            print(
                f"  Epoch {epoch:03d} | Loss {train_loss:.4f} | Train acc {train_acc * 100:.1f}% | Val F1 {val_f1 * 100:.1f}%"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(
                    f"  Early stopping at epoch {epoch} (best val F1: {best_val_f1 * 100:.1f}%)"
                )
                break

    # --- Restore best model and evaluate ---
    if best_state:
        model.load_state_dict(best_state)

    print("\n4. Final evaluation...")
    train_labels, train_preds, _ = evaluate(model, train_loader, DEVICE)
    print_metrics(train_labels, train_preds, split="Train")

    test_labels, test_preds, _ = evaluate(model, test_loader, DEVICE)
    print_metrics(test_labels, test_preds, split="Test")

    # --- Save the model ---
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {"hidden_channels": 64, "dropout": 0.5},
        },
        "deadlock_gnn_best.pt",
    )
    print("\nModel saved to deadlock_gnn_best.pt")
