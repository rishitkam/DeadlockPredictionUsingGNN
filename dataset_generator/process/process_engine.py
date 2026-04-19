# dataset_generator/process/process_engine.py
from typing import Dict, List
import random
from .process import Process
from dataset_generator.workload.workload_generator import PoissonWorkloadGenerator
from dataset_generator.resources.resource_manager import ResourceManager

class OSEngine:
    def __init__(self, config: dict):
        self.config = config
        self.tick = 0
        self.next_pid = 1
        
        self.processes: Dict[int, Process] = {}
        self.completed_processes = []
        
        self.workload = PoissonWorkloadGenerator(
            config['simulation']['poisson_lambda'], 
            config['simulation'].get('seed', None)
        )
        
        self.res_manager = ResourceManager(
            num_resources=config['os_environment']['number_of_resources'],
            max_capacity=config['os_environment']['max_resource_capacity']
        )
        
    def spawn_processes(self):
        """Invokes workload generator to simulate arriving processes"""
        # Normal Poisson arrivals
        arrivals = self.workload.next_arrivals()
        
        # Burst probability
        if random.random() < self.config['os_environment']['burst_probability']:
            arrivals += random.randint(1, self.config['os_environment']['max_process_burst'])
            
        for _ in range(arrivals):
            seq = self.workload.generate_random_sequence(
                self.config['os_environment']['number_of_resources'],
                self.config['process_behavior']['min_requests_per_process'],
                self.config['process_behavior']['max_requests_per_process']
            )
            hold_time = random.randint(
                self.config['process_behavior']['hold_duration_min'],
                self.config['process_behavior']['hold_duration_max']
            )
            p = Process(self.next_pid, self.tick, seq, hold_time)
            self.processes[p.pid] = p
            self.next_pid += 1

    def step(self):
        """Advances the OS by 1 clock tick."""
        self.tick += 1
        self.spawn_processes()
        
        active_pids = list(self.processes.keys())
        # Randomize schedule order to simulate threading concurrency
        random.shuffle(active_pids)
        
        for pid in active_pids:
            p = self.processes[pid]
            p.tick()
            
            if p.is_finished():
                self.res_manager.release_from_process(p)
                self.completed_processes.append(p)
                del self.processes[pid]
                continue
                
            # If ready or waiting, attempt to acquire next resource
            if p.state in ["ready", "waiting", "running"]:
                req = p.current_request()
                if req and req not in p.held_resources:
                     self.res_manager.attempt_allocation(p)
                     
    def get_system_state(self):
        """Extracts the global OS graph topology for RAG conversion"""
        assignments, requests = self.res_manager.get_allocation_state()
        
        nodes = []
        # Export all active processes
        for p in self.processes.values():
            nodes.append({
                "id": f"P{p.pid}",
                "type": "process",
                "state": p.state,
                "ticks_waiting": p.ticks_waiting
            })
            
        # Export all resources
        for rid, res in self.res_manager.resources.items():
            nodes.append({
                "id": rid,
                "type": "resource",
                "capacity": res.capacity,
                "allocated_count": len(res.allocated_to)
            })
        
        edges = []
        for r, p in assignments:
            edges.append((r, f"P{p}", "assign"))
        for p, r in requests:
            edges.append((f"P{p}", r, "request"))
            
        return nodes, edges
