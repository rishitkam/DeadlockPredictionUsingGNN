# dataset_generator/converter/pyg_converter.py
import torch
from torch_geometric.data import Data
import networkx as nx
from dataset_generator.algorithms.wait_for_graph import detect_deadlock_wfg

class PyGConverter:
    @staticmethod
    def convert(nx_graph: nx.DiGraph) -> Data:
        """
        Converts the snapshot RAG into a PyG Data tensor.
        7-dim Node Vector:
          0: is_process
          1: is_resource
          2: normalized_degree
          3: in_out_ratio
          4: requests_connected
          5: allocations_connected
          6: resource_utilisation (holders/capacity)
        """
        is_deadlock = 1.0 if detect_deadlock_wfg(nx_graph) else 0.0
        
        nodes = list(nx_graph.nodes)
        mapping = {n: i for i, n in enumerate(nodes)}
        
        max_deg = max(dict(nx_graph.degree()).values(), default=1)
        
        x = []
        for n in nodes:
            attr = nx_graph.nodes[n]
            ntype = attr.get("node_type")
            in_deg = nx_graph.in_degree(n)
            out_deg = nx_graph.out_degree(n)
            
            norm_deg = (in_deg + out_deg) / max_deg if max_deg > 0 else 0
            in_out_ratio = in_deg / (out_deg + 1.0)
            
            if ntype == "process":
                reqs = out_deg
                allocs = in_deg
                util = 0.0
                x.append([1.0, 0.0, float(norm_deg), float(in_out_ratio), float(reqs), float(allocs), float(util)])
            else:
                reqs = in_deg
                allocs = out_deg
                cap = attr.get("capacity", 1.0)
                util = allocs / cap if cap > 0 else 0.0
                x.append([0.0, 1.0, float(norm_deg), float(in_out_ratio), float(reqs), float(allocs), float(util)])
                
        x_tensor = torch.tensor(x, dtype=torch.float)
        
        edge_idx = []
        edge_attr = []
        
        for u, v, dat in nx_graph.edges(data=True):
            edge_idx.append([mapping[u], mapping[v]])
            # 0 for Request, 1 for Assignment
            edge_attr.append(0 if dat.get("edge_type") == "request" else 1)
            
        if edge_idx:
            ei_tensor = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
            ea_tensor = torch.tensor(edge_attr, dtype=torch.long)
        else:
            ei_tensor = torch.empty((2, 0), dtype=torch.long)
            ea_tensor = torch.empty((0,), dtype=torch.long)
            
        y = torch.tensor([is_deadlock], dtype=torch.float)
        
        return Data(x=x_tensor, edge_index=ei_tensor, edge_type=ea_tensor, y=y)
