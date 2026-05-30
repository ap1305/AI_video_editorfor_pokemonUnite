import os
import json
from datetime import datetime

class PipelineRecoveryManager:
    def __init__(self, video_path: str, base_dir: str = "logs"):
        """
        Uses deterministic folder naming based on the video file. 
        This allows the pipeline to 'remember' previous crashes.
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.state_dir = os.path.join(base_dir, f"{video_name}_state")
        os.makedirs(self.state_dir, exist_ok=True)
        
        self.telemetry_path = os.path.join(self.state_dir, "telemetry.jsonl")
        self.log_event("recovery_manager_boot", {"video": video_path})

    def log_event(self, phase: str, meta: dict = None):
        """Append-only telemetry (O(1) I/O)."""
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "phase": phase,
            "meta": meta or {}
        }
        with open(self.telemetry_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def save_state(self, name: str, data: dict):
        """Saves a pipeline artifact (e.g., Qwen JSON, EDF)."""
        path = os.path.join(self.state_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def load_state(self, name: str) -> dict:
        """
        THE RECOVERY BRAIN: Checks if a phase was already completed.
        Returns the JSON if it exists, otherwise returns None.
        """
        path = os.path.join(self.state_dir, f"{name}.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                print(f"♻️ [Recovery] Found cached state for '{name}'. Skipping heavy execution!")
                self.log_event("state_recovered", {"artifact": name})
                return data
            except json.JSONDecodeError:
                print(f"⚠️ [Recovery] Corrupted state file found: {name}. Overwriting.")
                return None
        return None