# dataset_generator/workload/workload_generator.py
import random
import math

class PoissonWorkloadGenerator:
    def __init__(self, lambda_obj: float, seed: int = None):
        if seed is not None:
            random.seed(seed)
        self.lam = lambda_obj
        
    def next_arrivals(self) -> int:
        """Generates random process arrival bursts from Poisson distribution."""
        L = math.exp(-self.lam)
        k = 0
        p = 1.0
        while True:
            p = p * random.random()
            if p <= L:
                break
            k += 1
        return k

    def generate_random_sequence(self, num_resources: int, min_req: int, max_req: int) -> list:
        """Generates a random sequence of completely unordered resource requests."""
        if num_resources == 0:
            return []
        count = random.randint(min_req, min(max_req, num_resources))
        # Unique resources to avoid self-deadlocking immediately logically
        res_pool = [f"R{i}" for i in range(num_resources)]
        safe_sample = random.sample(res_pool, count) 
        
        # We can add duplication logic for multi-instances if we wanted, 
        # but unique resource IDs per sequence is safer for OS deadlocks
        return safe_sample
