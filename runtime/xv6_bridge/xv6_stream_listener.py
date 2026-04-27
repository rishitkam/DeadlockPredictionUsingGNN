# runtime/xv6_bridge/xv6_stream_listener.py
import subprocess
import threading
import time
import os
import sys
from .event_parser import XV6EventParser
from .rag_builder import XV6RAGBuilder

class XV6StreamListener:
    """
    Launches xv6 in a DEDICATED macOS Terminal window and listens to its output
    via a UNIX FIFO (Named Pipe) for real-time neural processing.
    """
    def __init__(self, xv6_dir: str, callback=None):
        self.xv6_dir = os.path.abspath(xv6_dir)
        self.callback = callback
        self.parser = XV6EventParser()
        self.builder = XV6RAGBuilder()
        self.running = False
        
        # FIFO Configuration
        self.fifo_path = "/tmp/xv6_gnn_pipe"
        self.cleanup_pipe()

    def cleanup_pipe(self):
        if os.path.exists(self.fifo_path):
            try:
                os.remove(self.fifo_path)
            except:
                pass

    def start(self):
        """Starts the dedicated window and the listener thread."""
        self.running = True
        
        # Create FIFO
        try:
            os.mkfifo(self.fifo_path)
        except OSError as e:
            print(f"Error creating FIFO: {e}")
            return

        # Start background reader thread
        self.thread = threading.Thread(target=self._run_reader, daemon=True)
        self.thread.start()

        # Launch dedicated terminal via AppleScript
        self._launch_terminal()

    def _launch_terminal(self):
        # Command to run in the new window
        # 1. cd to xv6 dir
        # 2. Set title
        # 3. run make qemu and pipe output to FIFO using tee (to keep console interactive)
        inner_cmd = f"cd \\\"{self.xv6_dir}\\\"; clear; printf \\\"\\\\033]0;xv6 GNN Monitor\\\\007\\\"; make qemu | tee {self.fifo_path}"
        
        applescript = f'tell application "Terminal" to do script "{inner_cmd}" activate'
        subprocess.run(["osascript", "-e", applescript])

    def _run_reader(self):
        """Reads from the FIFO and updates the RAG."""
        try:
            # Note: opening a FIFO for reading blocks until a writer opens it
            with open(self.fifo_path, 'r') as fifo:
                while self.running:
                    line = fifo.readline()
                    if not line:
                        # EOF - pipe closed
                        time.sleep(0.1)
                        continue
                    
                    event = self.parser.parse_line(line)
                    if event:
                        self.builder.update(event)
                        if self.callback:
                            self.callback(self.builder.get_graph(), event)
                    
                    # Debug log
                    if "[GNN_TRACE]" in line:
                        with open("xv6_bridge_debug.log", "a") as f:
                            f.write(line)

        except Exception as e:
            print(f"FIFO Reader Error: {e}")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        # We don't terminate the terminal window (user might want to keep playing),
        # but the AI stops listening.
        self.cleanup_pipe()

    def get_current_graph(self):
        return self.builder.get_graph()
