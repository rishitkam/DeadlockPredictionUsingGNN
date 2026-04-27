# dataset_generator/resources/resource_manager.py
from typing import Dict, List, Set
import random
from .resource import Resource

class ResourceManager:
    def __init__(self, num_resources: int, max_capacity: int):
        self.resources: Dict[str, Resource] = {}
        for i in range(num_resources):
            rid = f"R{i}"
            # Randomize block capacity between 1 and max_capacity
            cap = random.randint(1, max_capacity)
            self.resources[rid] = Resource(res_id=rid, capacity=cap)
            
    def get_resource(self, res_id: str) -> Resource:
        return self.resources.get(res_id)

    def attempt_allocation(self, process) -> bool:
        """
        Attempts to allocate the process's current requested resource.
        Returns True if successful.
        """
        req_id = process.current_request()
        if not req_id:
            return False
            
        res = self.resources[req_id]
        success = res.allocate(process.pid)
        
        if success:
            process.acquire(req_id)
            return True
        else:
            process.block(req_id)
            return False

    def release_from_process(self, process):
        """Releases all resources held by the process"""
        woken_pids = set()
        for res_id in list(process.held_resources):
            res = self.resources[res_id]
            next_pid = res.release(process.pid)
            if next_pid is not None:
                woken_pids.add(next_pid)
                
        process.held_resources.clear()
        # Note: Waking up queued processes happens in the Engine tick loop automatically
        # as blocked processes retry their allocation.
        return woken_pids
    
    def get_allocation_state(self):
        """Returns adjacency lists for assignments and requests"""
        assignments = [] # (R, P)
        requests = []    # (P, R)
        
        for r_id, res in self.resources.items():
            for p_id in res.allocated_to:
                assignments.append((r_id, p_id))
            for p_id in res.waiting_queue:
                requests.append((p_id, r_id))
                
        return assignments, requests
