import torch
import networkx as nx

from deadlock_gnn.algorithms.bankers import bankers_check, rag_to_banker_matrices
from deadlock_gnn.algorithms.wfg import build_wfg, detect_cycle_dfs
from deadlock_gnn.data.converter import convert_to_pyg_data

def hybrid_detect(G: nx.DiGraph, gnn_model: torch.nn.Module, is_rgcn: bool = True, threshold: float = 0.5):
    """
    Combines Banker's safety check with GNN probability (DRIP hybrid architecture).
    """
    resources = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'resource']
    
    # Classical path
    allocation, max_demand, available = rag_to_banker_matrices(G, resources)
    banker_safe, safe_seq = bankers_check(allocation, max_demand, available)
    
    wfg = build_wfg(G, resources)
    wfg_deadlocked, cycle_nodes = detect_cycle_dfs(wfg)
    
    # GNN path
    pyg_data = convert_to_pyg_data(G, label=0)
    device = next(gnn_model.parameters()).device
    pyg_data = pyg_data.to(device)
    
    gnn_model.eval()
    with torch.no_grad():
        batch = torch.zeros(pyg_data.x.size(0), dtype=torch.long).to(device)
        if is_rgcn:
            out = gnn_model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_type, batch).view(-1)
        else:
            out = gnn_model(pyg_data.x, pyg_data.edge_index, batch).view(-1)
            
        gnn_prob = torch.sigmoid(out).item()
        
    # Fusion logic (DRIP-inspired)
    if wfg_deadlocked:
        return "DEADLOCK", "HIGH_CONFIDENCE", cycle_nodes, gnn_prob
    elif not banker_safe and gnn_prob > threshold:
        return "DEADLOCK", "MEDIUM_CONFIDENCE", [], gnn_prob
    elif banker_safe and gnn_prob < threshold:
        return "SAFE", "HIGH_CONFIDENCE", [], gnn_prob
    else:
        return "UNCERTAIN", "LOW_CONFIDENCE", [], gnn_prob
