import torch
from torch_geometric.data import Data
import networkx as nx
from deadlock_gnn.algorithms.wfg import build_wfg, detect_cycle_dfs

def convert_to_pyg_data(nx_graph: nx.DiGraph, label: float) -> Data:
    """
    Converts a NetworkX RAG to a PyG Data object.

    Node features (8-dim):
      [0] is_process (binary)
      [1] is_resource (binary)
      [2] normalized degree
      [3] in-degree / (out-degree + 1) ratio
      [4] number of request edges incident
      [5] number of assignment edges incident
      [6] resource utilisation ratio = holders / capacity
      [7] is_in_cycle

    Edge types:
      0: Request edge (P -> R)
      1: Assignment edge (R -> P)
    """
    nodes = list(nx_graph.nodes)
    mapping = {node: i for i, node in enumerate(nodes)}
    
    max_deg = max(dict(nx_graph.degree()).values(), default=1)
    
    # Identify cycle nodes for ground-truth WFG feature
    resources = [n for n, d in nx_graph.nodes(data=True) if d.get('node_type') == 'resource']
    wfg = build_wfg(nx_graph, resources)
    is_deadlocked, cycle_nodes = detect_cycle_dfs(wfg)
    cycle_set = set(cycle_nodes)

    x = []
    for node in nodes:
        ntype = nx_graph.nodes[node].get("node_type")
        deg = nx_graph.degree(node)
        norm_deg = deg / max_deg
        
        in_deg = nx_graph.in_degree(node)
        out_deg = nx_graph.out_degree(node)
        in_out_ratio = in_deg / (out_deg + 1.0)
        
        in_cycle = 1.0 if node in cycle_set else 0.0
        
        if ntype == "process":
            # P -> R are requests (out_edges for P)
            # R -> P are assignments (in_edges for P)
            num_request = out_deg
            num_assign = in_deg
            util_ratio = 0.0
            x.append([1.0, 0.0, norm_deg, in_out_ratio, float(num_request), float(num_assign), util_ratio, in_cycle])
        else:
            # P -> R are requests (in_edges for R)
            # R -> P are assignments (out_edges for R)
            num_request = in_deg
            num_assign = out_deg
            capacity = float(nx_graph.nodes[node].get("capacity", 1.0))
            util_ratio = num_assign / capacity if capacity > 0 else 0.0
            x.append([0.0, 1.0, norm_deg, in_out_ratio, float(num_request), float(num_assign), util_ratio, in_cycle])
            
    x = torch.tensor(x, dtype=torch.float)

    edge_list = []
    edge_type_list = []
    for u, v in nx_graph.edges():
        u_type = nx_graph.nodes[u].get('node_type')
        if u_type == 'process':
            # P -> R is request (0)
            edge_type_list.append(0)
        else:
            # R -> P is assignment (1)
            edge_type_list.append(1)
            
        edge_list.append([mapping[u], mapping[v]])

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type_list, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=torch.tensor([label], dtype=torch.float))
