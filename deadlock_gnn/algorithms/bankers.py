import networkx as nx

def bankers_check(allocation: list[list[int]], max_demand: list[list[int]], available: list[int]) -> tuple[bool, list[int]]:
    """
    Standard Banker's Algorithm for safe-state detection.
    
    Args:
        allocation: n x m matrix of currently allocated resources
        max_demand: n x m matrix of maximum required resources
        available: m vector of currently available resources
        
    Returns:
        (is_safe, safe_sequence)
    """
    n = len(allocation)
    if n == 0:
        return True, []
    m = len(available)
    
    # Calculate need = max_demand - allocation
    need = []
    for i in range(n):
        need.append([max_demand[i][j] - allocation[i][j] for j in range(m)])
        
    finish = [False] * n
    work = list(available)
    safe_seq = []
    
    while True:
        progress = False
        for i in range(n):
            if not finish[i]:
                # Check if need[i] <= work
                if all(need[i][j] <= work[j] for j in range(m)):
                    # Fake allocation release
                    for j in range(m):
                        work[j] += allocation[i][j]
                    finish[i] = True
                    safe_seq.append(i)
                    progress = True
                    
        if not progress:
            break
            
    if all(finish):
        return True, safe_seq
    else:
        return False, []

def rag_to_banker_matrices(G: nx.DiGraph, resources: list[str]) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """
    Converts a RAG and resource list to allocation, max_demand, and available matrices.
    Assumes max_demand = current allocation + current requests.
    """
    processes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'process']
    p_idx = {p: i for i, p in enumerate(processes)}
    r_idx = {r: j for j, r in enumerate(resources)}
    
    n = len(processes)
    m = len(resources)
    
    allocation = [[0]*m for _ in range(n)]
    request = [[0]*m for _ in range(n)]
    
    for r in resources:
        capacity = G.nodes[r].get('capacity', 1)
        # Assignment edges: R -> P (P holds R)
        holders = [v for _, v in G.out_edges(r)]
        for h in holders:
            if h in p_idx:
                allocation[p_idx[h]][r_idx[r]] += 1
                
        # Request edges: P -> R (P requests R)
        waiters = [u for u, _ in G.in_edges(r) if u not in holders]
        for w in waiters:
            if w in p_idx:
                request[p_idx[w]][r_idx[r]] += 1
                
    max_demand = [[allocation[i][j] + request[i][j] for j in range(m)] for i in range(n)]
    
    available = [0]*m
    for j, r in enumerate(resources):
        capacity = G.nodes[r].get('capacity', 1)
        allocated_count = sum(allocation[i][j] for i in range(n))
        available[j] = capacity - allocated_count
        
    return allocation, max_demand, available
