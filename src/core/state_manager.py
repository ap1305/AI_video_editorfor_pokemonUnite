import sqlite3
import os
from datetime import datetime

class FactoryStateManager:
    def __init__(self, db_path: str = "data/factory_queue.db"):
        """Initializes the SQLite database to track video processing states."""
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize_db()

    def _initialize_db(self):
        """Creates the queue table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_video_path TEXT UNIQUE,
                    status TEXT,
                    added_at TIMESTAMP
                )
            """)
            conn.commit()

    def add_video_to_queue(self, video_path: str):
        """Registers a new video into the system if it isn't already tracked."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO job_queue (raw_video_path, status, added_at) VALUES (?, ?, ?)",
                (video_path, 'PENDING', datetime.now())
            )
            conn.commit()

    def get_pending_jobs(self) -> list[str]:
        """Fetches all videos that haven't been successfully completed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Grab anything that isn't finished (PENDING, FAILED, or stuck in PROCESSING from a crash)
            cursor.execute("SELECT raw_video_path FROM job_queue WHERE status != 'COMPLETED'")
            rows = cursor.fetchall()
            return [row[0] for row in rows]

    def update_job_status(self, video_path: str, new_status: str):
        """Updates the state of a video (PENDING -> PROCESSING -> COMPLETED/FAILED)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE job_queue SET status = ? WHERE raw_video_path = ?",
                (new_status, video_path)
            )
            conn.commit()

    def trigger_garbage_collection(self, video_path: str):
        """
        The Auto-Cleanup function. 
        Once a video is marked COMPLETED, we delete the massive raw source file.
        """
        print(f"[Garbage Collection] Attempting to clean up raw source: {video_path}")
        if os.path.exists(video_path):
            try:
                # Get file size before deleting to log how much space we saved
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                os.remove(video_path)
                print(f"[Garbage Collection] ✅ Deleted {video_path}. Freed {size_mb:.2f} MB of disk space.")
            except Exception as e:
                print(f"[Garbage Collection] ⚠️ Failed to delete {video_path}: {e}")
        else:
            print(f"[Garbage Collection] File already removed or not found: {video_path}")

if __name__ == "__main__":
    # Quick test of the DB logic
    db = FactoryStateManager(db_path="data/test_queue.db")
    db.add_video_to_queue("data/raw_gameplay_1.mp4")
    print("Pending jobs:", db.get_pending_jobs())
    db.update_job_status("data/raw_gameplay_1.mp4", "COMPLETED")
    print("Pending jobs after completion:", db.get_pending_jobs())