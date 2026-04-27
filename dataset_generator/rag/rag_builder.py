# dataset_generator/rag/rag_builder.py
import networkx as nx

class RAGBuilder:
    @staticmethod
    def build_from_state(nodes: list, edges: list) -> nx.DiGraph:
        """
        Constructs a NetworkX DiGraph from the OS state dump.
        Nodes contain feature dictionaries. Edge types are marked.
        """
        G = nx.DiGraph()
        
        for n in nodes:
            nid = n["id"]
            if n["type"] == "process":
                G.add_node(
                    nid, 
                    node_type="process",
                    state=n["state"],
                    ticks_waiting=n["ticks_waiting"],
                    # Fill dummy features for downstream pyg compatibility
                    capacity=1.0,
                    allocated_count=0
                )
            else:
                G.add_node(
                    nid, 
                    node_type="resource",
                    state="running",
                    ticks_waiting=0,
                    capacity=float(n["capacity"]),
                    allocated_count=float(n["allocated_count"])
                )
                
        for u, v, etype in edges:
            G.add_edge(u, v, edge_type=etype)
            
        return G
