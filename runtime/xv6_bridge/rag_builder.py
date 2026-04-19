# runtime/xv6_bridge/rag_builder.py
import networkx as nx
from typing import Dict, List, Any

class XV6RAGBuilder:
    """
    Maintains a Graph representation of the live xv6 OS state.
    Converts events into node and edge mutations.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        # Track which resources are locks vs sleep channels
        self.lock_owners = {} # Resource -> PID
        
    def update(self, event: Dict[str, Any]):
        etype = event["type"]
        pid = f"P{event['pid']}"
        res = f"R_{event['resource']}"
        
        if etype == "PROCESS_CREATE":
            self.graph.add_node(pid, node_type="process")
            
        elif etype == "PROCESS_EXIT":
            if pid in self.graph:
                self.graph.remove_node(pid)
                
        elif etype == "LOCK_ACQUIRE":
            # Ensure nodes exist
            self._ensure_nodes(pid, res)
            # A lock acquisition implies the assignment edge: Resource -> Process
            self.graph.add_edge(res, pid, type="assignment")
            # If there was a request edge, remove it (assuming process now has it)
            if self.graph.has_edge(pid, res):
                self.graph.remove_edge(pid, res)
                
        elif etype == "LOCK_RELEASE":
            if self.graph.has_edge(res, pid):
                self.graph.remove_edge(res, pid)
                
        elif etype == "PROCESS_SLEEP":
            self._ensure_nodes(pid, res)
            # Sleeping on a channel/lock is a request: Process -> Resource
            self.graph.add_edge(pid, res, type="request")
            
        elif etype == "PROCESS_WAKE":
            if self.graph.has_edge(pid, res):
                self.graph.remove_edge(pid, res)

    def _ensure_nodes(self, pid, res):
        if pid not in self.graph:
            self.graph.add_node(pid, node_type="process")
        if res not in self.graph:
            self.graph.add_node(res, node_type="resource", capacity=1.0)

    def get_graph(self) -> nx.DiGraph:
        return self.graph
