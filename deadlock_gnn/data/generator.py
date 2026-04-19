import random
import networkx as nx

def generate_rag(num_processes: int, num_resources: int, p_request: float = 0.25, p_assign: float = 0.25, max_capacity: int = 3) -> nx.DiGraph:
    """
    Generates a single directed Resource Allocation Graph (RAG) with multi-instance support.
    """
    G = nx.DiGraph()
    processes = [f"P{i}" for i in range(num_processes)]
    resources = [f"R{i}" for i in range(num_resources)]

    G.add_nodes_from(processes, node_type="process")
    for r in resources:
        capacity = random.randint(1, max_capacity)
        G.add_node(r, node_type="resource", capacity=capacity)

    for r in resources:
        capacity = G.nodes[r]['capacity']
        holders = 0
        shuffled_procs = processes[:]
        random.shuffle(shuffled_procs)

        for p in shuffled_procs:
            rand = random.random()
            if rand < p_assign and holders < capacity:
                G.add_edge(r, p)  # Assignment edge: R -> P
                holders += 1
            elif rand < p_assign + p_request:
                if not G.has_edge(r, p):  # Don't request what you already hold
                    G.add_edge(p, r)  # Request edge: P -> R

    return G
