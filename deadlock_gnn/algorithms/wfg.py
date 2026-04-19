import networkx as nx

def build_wfg(G: nx.DiGraph, resources: list[str]) -> nx.DiGraph:
    """
    Builds the Wait-For Graph (WFG) from the Resource Allocation Graph (RAG).
    A process P1 waits for P2 if P1 requests a resource R, and R is held by P2.
    For multi-instance resources (capacity k), processes only wait if all slots are full.
    """
    wfg = nx.DiGraph()
    processes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'process']
    wfg.add_nodes_from(processes)
    
    for r in resources:
        # Holders: R -> P
        holders = [v for _, v in G.out_edges(r)]
        # Waiters: P -> R
        waiters = [u for u, _ in G.in_edges(r) if u not in holders]
        
        capacity = G.nodes[r].get('capacity', 1)
        
        # If resource is not fully saturated, waiters don't strictly wait on holders
        # (they wait on the resource, but not forming a hard wait-for edge to a holder)
        if len(holders) < capacity:
            continue
            
        for waiter in waiters:
            for holder in holders:
                if waiter != holder:
                    wfg.add_edge(waiter, holder)
                    
    return wfg

def detect_cycle_dfs(wfg: nx.DiGraph) -> tuple[bool, list[str]]:
    """
    Detects deadlocks by finding a cycle in the WFG.
    Returns:
        (is_deadlocked, cycle_nodes)
    """
    try:
        cycle_nodes = next(nx.simple_cycles(wfg))
        return True, cycle_nodes
    except StopIteration:
        return False, []
