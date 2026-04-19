# runtime/xv6_bridge/xv6_stream_listener.py
import subprocess
import threading
import time
from .event_parser import XV6EventParser
from .rag_builder import XV6RAGBuilder

class XV6StreamListener:
    """
    Spawns xv6 via QEMU and listens to its stdout stream.
    Updates a live RAG model in real-time.
    """
    def __init__(self, xv6_dir: str, callback=None):
        self.xv6_dir = xv6_dir
        self.callback = callback
        self.parser = XV6EventParser()
        self.builder = XV6RAGBuilder()
        self.running = False
        self.process = None

    def start(self):
        """Starts the QEMU process and the listener thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        # make qemu-nox (no graphic window) is usually better for console scraping
        cmd = ["make", "qemu-nox"]
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=self.xv6_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in self.process.stdout:
                if not self.running:
                    break
                
                event = self.parser.parse_line(line)
                if event:
                    self.builder.update(event)
                    if self.callback:
                        self.callback(self.builder.get_graph(), event)
                
                # Optional: log line for debugging if it starts with [GNN_TRACE]
                if "[GNN_TRACE]" in line:
                    with open("xv6_bridge_debug.log", "a") as f:
                        f.write(line)

        except Exception as e:
            print(f"Error running xv6: {e}")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
            self.process = None

    def get_current_graph(self):
        return self.builder.get_graph()
