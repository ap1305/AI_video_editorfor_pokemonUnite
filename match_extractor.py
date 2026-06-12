import os
import cv2
import json
import base64
import subprocess
import numpy as np
from dotenv import load_dotenv

# Use your existing Colab/Hosted LLM handler
from src.utils.llm_client import execute_with_colab_fallback

load_dotenv()

class MatchExtractor:
    def __init__(self):
        self.output_dir = "data/gameplay_segments"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.api_url = os.getenv("QWEN_PACING_URL")
        self.api_key = os.getenv("QWEN_API_KEY", "ollama")
        
        # We still crop the top 52% so Qwen doesn't get distracted by your chat/webcam
        self.crop_box = [0.00, 0.00, 1.00, 0.52] 

    def _apply_viewport_crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = int(w * self.crop_box[0])
        y1 = int(h * self.crop_box[1])
        x2 = int(w * self.crop_box[2])
        y2 = int(h * self.crop_box[3])
        return frame[y1:y2, x1:x2]

    def _ask_qwen_state(self, frame: np.ndarray) -> bool:
        """Sends a compressed frame to Qwen to classify as Gameplay (True) or Lobby (False)."""
        # Compress heavily to save API bandwidth
        h, w, _ = frame.shape
        final_h = int(h * (426 / w))
        resized = cv2.resize(frame, (426, final_h))
        _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        b64_img = base64.b64encode(buffer).decode('utf-8')

        prompt = """Look at this image from a Pokémon UNITE stream.
        Is this active, in-progress gameplay?
        - Reply TRUE if you see the in-game timer, minimap, and active map movement.
        - Reply FALSE if it is the main menu, lobby, character selection screen, loading screen, or final scoreboard.
        
        Respond ONLY in valid JSON format:
        {
            "is_gameplay": true_or_false
        }"""

        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
        ]
        messages = [{"role": "user", "content": content}]

        try:
            raw_response = execute_with_colab_fallback(self.api_key, self.api_url, messages)
            
            # JSON Sanitization Block
            cleaned = raw_response.strip()
            md_ticks = chr(96) * 3 
            if cleaned.startswith(md_ticks + "json"): cleaned = cleaned[7:]
            elif cleaned.startswith(md_ticks): cleaned = cleaned[3:]
            if cleaned.endswith(md_ticks): cleaned = cleaned[:-3]
            
            cleaned = cleaned.strip()
            start_idx = cleaned.find('{')
            end_idx = cleaned.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned = cleaned[start_idx:end_idx+1]
                
            result = json.loads(cleaned)
            return result.get("is_gameplay", False)
            
        except Exception as e:
            print(f"      ⚠️ API Error: {e}")
            return False # Default to false on error to prevent fake matches

    def extract_matches(self, vod_path: str, sample_interval_sec: int = 15):
        print(f"\n🧠 [Match Extractor] Booting Qwen Vision API...")
        print(f"🎮 Analyzing VOD: {vod_path}")
        
        vod_name = os.path.splitext(os.path.basename(vod_path))[0]
        cap = cv2.VideoCapture(vod_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) if fps > 0 else 0

        in_game = False
        consecutive_in_game = 0
        consecutive_out_game = 0
        current_match_start = 0.0
        matches = []

        print(f"   [State Machine] Scanning via API every {sample_interval_sec} seconds. This will take ~15-20 mins...")
        
        for sec_index in range(0, duration_sec, sample_interval_sec):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec_index * 1000)
            ret, frame = cap.read()
            if not ret: continue

            cropped_frame = self._apply_viewport_crop(frame)
            is_gameplay = self._ask_qwen_state(cropped_frame)

            # --- THE STATE MACHINE LOGIC ---
            if is_gameplay:
                consecutive_in_game += 1
                consecutive_out_game = 0
                
                # START RULE: 2 consecutive 'True' hits (30 seconds of confirmed gameplay)
                if not in_game and consecutive_in_game >= 2:
                    in_game = True
                    # Roll back 30 seconds (the samples) + 5 seconds buffer to catch "GO!"
                    current_match_start = max(0.0, float(sec_index - 35.0))
                    print(f"   ✅ [{sec_index//60:02d}m {sec_index%60:02d}s] Match STARTED! Locked in at {current_match_start/60:.1f}m")
                else:
                    print(f"   👁️ [{sec_index//60:02d}m {sec_index%60:02d}s] Qwen sees Gameplay.")
            else:
                consecutive_out_game += 1
                consecutive_in_game = 0
                
                # END RULE: 3 consecutive 'False' hits (45 seconds of lobby/scoreboard to bypass long respawn screens)
                if in_game and consecutive_out_game >= 3:
                    in_game = False
                    # Roll back 45 seconds (the samples) - 5 seconds buffer to catch "Time's UP!"
                    match_end = float(sec_index - 40.0)
                    match_duration = match_end - current_match_start

                    # Duration Rule: 8 to 15 minutes
                    if 480 <= match_duration <= 900:
                        matches.append((current_match_start, match_end))
                        print(f"   🛑 [{sec_index//60:02d}m {sec_index%60:02d}s] Match ENDED. Saved {match_duration/60:.1f}m game.")
                    else:
                        print(f"   ⚠️ [{sec_index//60:02d}m {sec_index%60:02d}s] Rejected game chunk: {match_duration/60:.1f}m (Out of bounds)")
                else:
                    if not in_game:
                        print(f"   💤 [{sec_index//60:02d}m {sec_index%60:02d}s] Qwen sees Lobby/Menu.")

        # Catch match if VOD ends during gameplay
        if in_game:
            match_duration = duration_sec - current_match_start
            if 480 <= match_duration <= 900:
                matches.append((current_match_start, duration_sec))

        cap.release()

        print(f"\n✂️ [Match Extractor] Qwen found {len(matches)} valid 10-minute games. Slicing now...")
        extracted_files = []
        for i, (start_t, end_t) in enumerate(matches, 1):
            dur = end_t - start_t
            
            sh, sm, ss = int(start_t // 3600), int((start_t % 3600) // 60), int(start_t % 60)
            eh, em, es = int(end_t // 3600), int((end_t % 3600) // 60), int(end_t % 60)
            time_str = f"{sh:02d}h{sm:02d}m{ss:02d}s_to_{eh:02d}h{em:02d}m{es:02d}s"
            
            out_file = os.path.join(self.output_dir, f"{vod_name}_match_{str(i).zfill(3)}_{time_str}.mp4")
            extracted_files.append(out_file)
            
            if os.path.exists(out_file):
                print(f"   ⏩ Skipping Match {i} (already extracted)")
                continue
                
            print(f"   🎬 FFmpeg Extracting Match {i} ({dur/60:.1f} mins)...")
            cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(start_t), "-t", str(dur), "-i", vod_path, "-c", "copy", out_file]
            subprocess.run(cmd)

        print(f"🎉 Complete! Saved to {self.output_dir}/")
        return extracted_files

# --- ISOLATED TEST BLOCK ---
if __name__ == "__main__":
    extractor = MatchExtractor()
    test_vod = "data/raw_vods/LIVE_RANKED_GRIND_Pokemon_UNITE_Shorts.mkv.mp4"
    if os.path.exists(test_vod):
        extractor.extract_matches(test_vod)
    else:
        print(f"❌ Could not find test video at: {test_vod}")