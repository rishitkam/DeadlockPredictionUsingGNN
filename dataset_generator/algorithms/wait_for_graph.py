# dataset_generator/algorithms/wait_for_graph.py
import networkx as nx

def detect_deadlock_wfg(G: nx.DiGraph) -> bool:
    """
    Detects if a deadlock exists in a PyG compatible RAG by reducing it 
    to a pure Wait-For-Graph (WFG) and detecting directed cycles.
    """
    wfg = nx.DiGraph()
    
    # Isolate Process nodes
    processes = [n for n, attr in G.nodes(data=True) if attr.get("node_type") == "process"]
    wfg.add_nodes_from(processes)
    
    # WFG Edges: P1 -> R -> P2 becomes P1 -> P2 (meaning P1 waits on P2)
    # Note: RAG Request is P1 -> R. Assignment is R -> P2.
    resources = [n for n, attr in G.nodes(data=True) if attr.get("node_type") == "resource"]
    
    for r in resources:
        # Who is requesting this resource?
        requesting_procs = [u for u, v, d in G.in_edges(r, data=True) if d.get('edge_type') == 'request']
        # Who is currently holding this resource?
        holding_procs = [v for u, v, d in G.out_edges(r, data=True) if d.get('edge_type') == 'assign']
        
        # If capacity is maxed out, everyone requesting is mathematically waiting on EVERYONE holding it
        # Actually in classical OS, if a multi-instance resource R is fully held, 
        # a requesting process is waiting on ANY of the holders to release.
        # But for conservative WFG bounds, we add edges to all holders.
        if len(holding_procs) >= G.nodes[r].get("capacity", 1.0) and requesting_procs:
            for p_req in requesting_procs:
                for p_hold in holding_procs:
                    wfg.add_edge(p_req, p_hold)
                    
    try:
        # Find ANY cycle
        cycle = nx.find_cycle(wfg, orientation="original")
        return True
    except nx.NetworkXNoCycle:
        return False
