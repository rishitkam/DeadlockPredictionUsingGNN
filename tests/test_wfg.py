import pytest
import networkx as nx
from deadlock_gnn.algorithms.wfg import build_wfg, detect_cycle_dfs

def test_wfg_cycle():
    # 2 processes, 2 resources, circular wait
    G = nx.DiGraph()
    G.add_node("P1", node_type="process")
    G.add_node("P2", node_type="process")
    G.add_node("R1", node_type="resource", capacity=1)
    G.add_node("R2", node_type="resource", capacity=1)
    
    G.add_edge("R1", "P1") # P1 holds R1
    G.add_edge("R2", "P2") # P2 holds R2
    G.add_edge("P1", "R2") # P1 requests R2
    G.add_edge("P2", "R1") # P2 requests R1
    
    resources = ["R1", "R2"]
    wfg = build_wfg(G, resources)
    
    # Needs to wait for each other
    assert wfg.has_edge("P1", "P2")
    assert wfg.has_edge("P2", "P1")
    
    is_deadlocked, cycle = detect_cycle_dfs(wfg)
    assert is_deadlocked
    assert set(cycle) == {"P1", "P2"}

def test_wfg_safe():
    # 2 processes, 2 resources, safe
    G = nx.DiGraph()
    G.add_node("P1", node_type="process")
    G.add_node("P2", node_type="process")
    G.add_node("R1", node_type="resource", capacity=1)
    G.add_node("R2", node_type="resource", capacity=1)
    
    G.add_edge("R1", "P1") # P1 holds R1
    G.add_edge("R2", "P2") # P2 holds R2
    G.add_edge("P2", "R1") # P2 requests R1, waits on P1
    # P1 requests nothing
    
    resources = ["R1", "R2"]
    wfg = build_wfg(G, resources)
    
    assert wfg.has_edge("P2", "P1")
    assert not wfg.has_edge("P1", "P2")
    
    is_deadlocked, cycle = detect_cycle_dfs(wfg)
    assert not is_deadlocked
    assert len(cycle) == 0

def test_wfg_multi_instance():
    G = nx.DiGraph()
    G.add_node("P1", node_type="process")
    G.add_node("P2", node_type="process")
    G.add_node("P3", node_type="process")
    G.add_node("R1", node_type="resource", capacity=2)
    
    G.add_edge("R1", "P1")
    # Capacity is 2, length is 1, so P2 requesting R1 won't form a strict WFG edge yet
    G.add_edge("P2", "R1")
    
    resources = ["R1"]
    wfg = build_wfg(G, resources)
    is_deadlocked, cycle = detect_cycle_dfs(wfg)
    assert not is_deadlocked
    assert len(wfg.edges) == 0
    
    # Fill capacity
    G.add_edge("R1", "P3")
    wfg = build_wfg(G, resources)
    # Now P2 should wait for P1 and P3
    assert wfg.has_edge("P2", "P1")
    assert wfg.has_edge("P2", "P3")
