import os
import json
import subprocess

class VerifiedClipCutter:
    def __init__(self):
        self.output_dir = "data/vod_verified_candidates"
        os.makedirs(self.output_dir, exist_ok=True)

    def cut_approved_clips(self, verified_json_path: str):
        print(f"\n✂️ [Precise Cutter] Reading {os.path.basename(verified_json_path)}...")

        with open(verified_json_path, 'r') as f:
            candidates = json.load(f)

        approved_clips = [c for c in candidates if c.get("approved", False)]
        if not approved_clips:
            print("⚠️ No approved clips to cut in this file.")
            return

        for clip in approved_clips:
            vod_path = clip["source_vod"]
            vod_name = os.path.splitext(os.path.basename(vod_path))[0]
            w_id = clip["window_id"].replace("RAW_", "VERIFIED_")
            
            peak_t = float(clip.get("peak_time", (clip["start_time"] + clip["end_time"]) / 2.0))

            # Dynamic 32-second window anchored precisely on the action peak
            start_t = max(0.0, peak_t - 14.0)
            end_t = peak_t + 18.0
            duration = end_t - start_t
            
            if not (12.0 <= duration <= 45.0):
                print(f"   ⚠️ Skipping {w_id}: Duration {duration}s out of bounds.")
                continue

            out_file = os.path.join(self.output_dir, f"{vod_name}_{w_id}_{int(start_t)}s.mp4")
            if os.path.exists(out_file):
                print(f"   ⏩ Skipping {w_id} (already cut)")
                continue

            print(f"   🎬 Re-encoding {w_id} centered at peak {peak_t}s...")
            
            # Frame-accurate FFmpeg re-encoding (-ss after -i)
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", vod_path,
                "-ss", str(start_t),
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "192k",
                out_file
            ]
            subprocess.run(cmd)

        print(f"✅ Successfully cut verified clips to {self.output_dir}/")