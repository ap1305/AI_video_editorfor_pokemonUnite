import os
import cv2
import json
import subprocess
import numpy as np
from typing import List, Dict, Tuple

class MatchExtractor:
    def __init__(self):
        self.templates_dir = "assets/templates/pokemon_unite"
        self.output_dir = "data/gameplay_segments"
        self.debug_dir = "data/debug"
        
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # ✅ CHECKLIST: Detect gameplay viewport/crop first & Ignore stream overlay
        # Adjust this if the stream layout changes! (Currently Top 52% of screen)
        self.crop_box = [0.00, 0.00, 1.00, 0.52] 
        
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, np.ndarray]:
        templates = {}
        files = {
            "timer": "gameplay_timer.png",
            "minimap": "minimap.png"
        }
        for key, filename in files.items():
            path = os.path.join(self.templates_dir, filename)
            if os.path.exists(path):
                templates[key] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            else:
                print(f"⚠️ [Match Extractor] Missing template: {path}")
        return templates

    def _apply_viewport_crop(self, frame: np.ndarray) -> np.ndarray:
        """Crops the frame to ignore stream overlays, webcams, and chat."""
        h, w = frame.shape[:2]
        x1 = int(w * self.crop_box[0])
        y1 = int(h * self.crop_box[1])
        x2 = int(w * self.crop_box[2])
        y2 = int(h * self.crop_box[3])
        return frame[y1:y2, x1:x2]

    def _save_debug_crop(self, frame: np.ndarray, vod_name: str):
        """Saves a visual representation of what the Extractor is actually looking at."""
        cropped = self._apply_viewport_crop(frame)
        debug_path = os.path.join(self.debug_dir, f"{vod_name}_viewport_debug.jpg")
        cv2.imwrite(debug_path, cropped)
        print(f"   🖼️ Saved viewport debug image to: {debug_path}")

    def _check_in_game_state(self, cropped_gray: np.ndarray) -> bool:
        if not self.templates: return False

        h, w = cropped_gray.shape[:2]
        
        # We only care about the Timer now
        timer_region = cropped_gray[0:int(h*0.2), int(w*0.35):int(w*0.65)]
        timer_score = 0.0

        if "timer" in self.templates:
            res = cv2.matchTemplate(timer_region, self.templates["timer"], cv2.TM_CCOEFF_NORMED)
            _, timer_score, _, _ = cv2.minMaxLoc(res)

        # Print what it sees, so we know it's working
        print(f"      [Debug] Timer Confidence: {timer_score:.2f}")

        # ✅ Pure Timer logic with a safe 0.75 threshold
        return timer_score >= 0.75

    def extract_matches(self, vod_path: str, sample_interval_sec: int = 3):
        # ✅ CHECKLIST: Sample every 3-5 seconds
        print(f"\n🎮 [Match Extractor] Analyzing VOD: {vod_path}")
        if not self.templates:
            print("❌ Cannot extract matches without HUD templates (timer/minimap).")
            return

        vod_name = os.path.splitext(os.path.basename(vod_path))[0]
        cap = cv2.VideoCapture(vod_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_sec = int(total_frames / fps) if fps > 0 else 0

        # Save debug frame of the crop box from the first frame
        ret, first_frame = cap.read()
        if ret: self._save_debug_crop(first_frame, vod_name)

        in_game = False
        consecutive_in_game = 0
        consecutive_out_game = 0
        
        current_match_start = 0.0
        matches = []

        print(f"   [State Machine] Scanning at 1 frame every {sample_interval_sec}s...")
        
        for sec_index in range(0, duration_sec, sample_interval_sec):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec_index * 1000)
            ret, frame = cap.read()
            if not ret: continue

            # Standardize resolution for template matching, then crop
            small_frame = cv2.resize(frame, (1920, 1080))
            cropped_frame = self._apply_viewport_crop(small_frame)
            gray_crop = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

            is_gameplay_frame = self._check_in_game_state(gray_crop)

            if is_gameplay_frame:
                consecutive_in_game += 1
                consecutive_out_game = 0

                # ✅ CHECKLIST: Start segment only after IN_GAME is stable
                # START RULE: 3 consecutive samples (9 seconds) of HUD presence
                if not in_game and consecutive_in_game >= 3:
                    in_game = True
                    # Pad start backward by 5 seconds to catch the exact "GO" drop-in animation
                    current_match_start = max(0.0, float(sec_index - (sample_interval_sec * 3) - 5.0))
                    print(f"   ⚔️ Match STARTED around {current_match_start:.0f}s")
            else:
                consecutive_out_game += 1
                consecutive_in_game = 0

                # ✅ CHECKLIST: End segment when RESULT/Time’s Up appears or HUD disappears
                # END RULE: 6 consecutive samples (18 seconds) of missing HUD (Result/Lobby)
                if in_game and consecutive_out_game >= 6:
                    in_game = False
                    # Pad end forward by 5 seconds to catch the "Time's UP" screen
                    match_end = float(sec_index - (sample_interval_sec * 6) + 5.0)
                    match_duration = match_end - current_match_start

                    # ✅ CHECKLIST: Reject < 8 mins, Reject > 15 mins
                    if 480 <= match_duration <= 900:
                        matches.append((current_match_start, match_end))
                        print(f"   🛑 Match ENDED at {match_end:.0f}s. Validated Duration: {match_duration/60:.1f}m")
                    else:
                        print(f"   ⚠️ Rejected Segment: {match_duration/60:.1f}m (Out of 8-15m bounds)")

            if sec_index % max(1, (duration_sec // 10)) == 0:
                print(f"      ... {(sec_index / duration_sec) * 100:.0f}% scanned")

        cap.release()

        # ✅ CHECKLIST: Save only full matches to data/gameplay_segments/
        print(f"\n✂️ [Match Extractor] Found {len(matches)} complete matches. Slicing...")
        for i, (start_t, end_t) in enumerate(matches, 1):
            dur = end_t - start_t
            
            # Format time for clean filenames (e.g., 01h22m15s)
            sh, sm, ss = int(start_t // 3600), int((start_t % 3600) // 60), int(start_t % 60)
            eh, em, es = int(end_t // 3600), int((end_t % 3600) // 60), int(end_t % 60)
            time_str = f"{sh:02d}h{sm:02d}m{ss:02d}s_to_{eh:02d}h{em:02d}m{es:02d}s"
            
            out_file = os.path.join(self.output_dir, f"{vod_name}_match_{str(i).zfill(3)}_{time_str}.mp4")
            
            if os.path.exists(out_file):
                print(f"   ⏩ Skipping Match {i} (already extracted)")
                continue
                
            print(f"   🎬 Extracting Match {i} ({dur/60:.1f} mins)...")
            
            # Fast, lossless extraction since these are massive 10-minute files
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-ss", str(start_t),
                "-t", str(dur),
                "-i", vod_path,
                "-c", "copy",
                out_file
            ]
            subprocess.run(cmd)

        print(f"🎉 Complete! All full matches saved to {self.output_dir}/")

if __name__ == "__main__":
    extractor = MatchExtractor()
    # You can test it directly or run via run_vod_pipeline.py