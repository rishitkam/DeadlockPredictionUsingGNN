# runtime/xv6_bridge/event_parser.py
import re
from typing import Optional, Dict, Any

class XV6EventParser:
    """
    Parses structured [GNN_TRACE] logs from xv6 console output.
    """
    TRACE_PATTERN = re.compile(r"\[GNN_TRACE\]\s+(\w+)\s+P(\d+)\s+(\S+)\s+(\d+)")

    @staticmethod
    def parse_line(line: str) -> Optional[Dict[str, Any]]:
        """
        Extracts event data from a raw string line.
        Returns a dict if a trace is found, else None.
        """
        match = XV6EventParser.TRACE_PATTERN.search(line)
        if not match:
            return None
        
        event_type = match.group(1)
        pid = int(match.group(2))
        resource = match.group(3)
        timestamp = int(match.group(4))

        return {
            "type": event_type,
            "pid": pid,
            "resource": resource,
            "timestamp": timestamp
        }
