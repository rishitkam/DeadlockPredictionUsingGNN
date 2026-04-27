# dataset_generator/process/process.py

class Process:
    def __init__(self, pid: int, arrival_time: int, requested_resources: list, hold_duration: int):
        self.pid = pid
        self.arrival_time = arrival_time
        
        # OS Tracking Properties
        self.resource_sequence = requested_resources  # List of resource IDs process wants in order
        self.held_resources = set()
        self.waiting_for = None                       # Resource ID process is currently blocked waiting on
        
        self.state = "ready"  # ready, running, waiting, finished
        
        self.hold_duration = hold_duration
        self.ticks_held = 0
        self.ticks_waiting = 0

    def current_request(self):
        """Returns the next resource in sequence if not finished."""
        if not self.resource_sequence:
            return None
        return self.resource_sequence[0]
        
    def acquire(self, res_id):
        """Called by Resource Manager when allocation succeeds."""
        if self.resource_sequence and self.resource_sequence[0] == res_id:
            self.resource_sequence.pop(0)
            self.held_resources.add(res_id)
            self.waiting_for = None
            self.state = "running"
            self.ticks_waiting = 0

    def block(self, res_id):
        """Process is starved."""
        self.waiting_for = res_id
        self.state = "waiting"
        
    def tick(self):
        """Advances internal process state by one OS cycle."""
        if self.state == "waiting":
            self.ticks_waiting += 1
        elif self.state == "running":
            self.ticks_held += 1
            if self.ticks_held >= self.hold_duration:
                # Finished holding completely
                if len(self.resource_sequence) == 0:
                    self.state = "finished"

    def is_finished(self):
        return self.state == "finished"
