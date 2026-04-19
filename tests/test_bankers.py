import pytest
import networkx as nx
from deadlock_gnn.algorithms.bankers import bankers_check, rag_to_banker_matrices

def test_bankers_check_safe():
    # Example from Silberschatz OS textbook
    # 5 processes, 3 resources
    allocation = [
        [0, 1, 0],
        [2, 0, 0],
        [3, 0, 2],
        [2, 1, 1],
        [0, 0, 2]
    ]
    max_demand = [
        [7, 5, 3],
        [3, 2, 2],
        [9, 0, 2],
        [2, 2, 2],
        [4, 3, 3]
    ]
    available = [3, 3, 2]
    
    is_safe, seq = bankers_check(allocation, max_demand, available)
    assert is_safe
    assert len(seq) == 5

def test_bankers_check_unsafe():
    # Modify above to be unsafe
    allocation = [
        [0, 1, 0],
        [2, 0, 0],
        [3, 0, 2],
        [2, 1, 1],
        [0, 0, 2]
    ]
    max_demand = [
        [7, 5, 3],
        [3, 2, 2],
        [9, 0, 2],
        [2, 2, 2],
        [4, 3, 3]
    ]
    available = [0, 1, 0] # Not enough to satisfy anyone fully initially
    
    is_safe, seq = bankers_check(allocation, max_demand, available)
    assert not is_safe

def test_rag_to_bankers():
    G = nx.DiGraph()
    G.add_node("P1", node_type="process")
    G.add_node("P2", node_type="process")
    G.add_node("R1", node_type="resource", capacity=2)
    
    # P1 holds 1 R1
    G.add_edge("R1", "P1")
    # P2 requests R1
    G.add_edge("P2", "R1")
    
    alloc, max_d, avail = rag_to_banker_matrices(G, ["R1"])
    # P1 holds 1
    assert alloc[0][0] == 1 or alloc[1][0] == 1
    # total alloc = 1, capacity 2, so available = 1
    assert avail[0] == 1
