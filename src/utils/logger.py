import os
import json
import uuid
from datetime import datetime

class PipelineLogger:
    def __init__(self, base_dir="logs"):
        self.run_id = str(uuid.uuid4())
        self.base_path = os.path.join(base_dir, self.run_id)
        os.makedirs(self.base_path, exist_ok=True)
        
        # Initialize telemetry file
        self.telemetry_path = os.path.join(self.base_path, "telemetry.jsonl")
        self.log_event("pipeline_boot", {"run_id": self.run_id})

    def log_state(self, name: str, data: dict):
        """Used for single-write state dumps (e.g., master_edf.json)"""
        path = os.path.join(self.base_path, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def log_event(self, phase: str, meta: dict = None):
        """Used for high-frequency telemetry. Append-only (O(1) I/O)."""
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "phase": phase,
            "meta": meta or {}
        }
        with open(self.telemetry_path, "a") as f:
            f.write(json.dumps(event) + "\n")
            
    def get_run_dir(self):
        return self.base_path