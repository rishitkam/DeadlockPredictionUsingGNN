# dataset_generator/resources/resource.py
from collections import deque

class Resource:
    def __init__(self, res_id: str, capacity: int = 1):
        self.res_id = res_id
        self.capacity = capacity
        
        self.allocated_to = set()  # Set of Process IDs holding this resource
        self.waiting_queue = deque()  # FIFO queue for waiting processes (PID)

    def is_available(self):
        return len(self.allocated_to) < self.capacity
        
    def allocate(self, pid: int) -> bool:
        """Attempts to allocate a slot to process PID. Returns True if successful."""
        if self.is_available():
            self.allocated_to.add(pid)
            if pid in self.waiting_queue:
                self.waiting_queue.remove(pid)
            return True
        else:
            if pid not in self.waiting_queue:
                self.waiting_queue.append(pid)
            return False

    def release(self, pid: int) -> int:
        """
        Releases the resource from the process PID.
        Returns the PID of the next process in queue that should wake up, or None.
        """
        if pid in self.allocated_to:
            self.allocated_to.remove(pid)
            
        if self.waiting_queue and self.is_available():
            next_pid = self.waiting_queue[0]
            # Don't pop it here, resource manager handles the re-allocation attempt
            return next_pid
            
        return None
