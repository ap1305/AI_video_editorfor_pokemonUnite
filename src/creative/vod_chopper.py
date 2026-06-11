import os
import cv2
import subprocess
import numpy as np
import shutil
import uuid
from typing import List, Dict, Tuple

class VODChopper:
    def __init__(self):
        self.templates_dir = "assets/templates/pokemon_unite"
        self.output_dir = "data/gameplay_segments" 
        
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, np.ndarray]:
        templates = {}
        files = {
            "timer": "gameplay_timer.png",
            "skills": "skill_buttons.png",
            "minimap": "minimap.png",
            "result": "result_screen.png",
            "time_up": "time_up.png",
            "lobby": "lobby_marker.png"
        }
        
        for key, filename in files.items():
            path = os.path.join(self.templates_dir, filename)
            if os.path.exists(path):
                templates[key] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            else:
                print(f"⚠️ [VOD Chopper] Missing template: {path}")
                
        return templates

    def _template_confidence(
        self,
        frame_gray: np.ndarray,
        template: np.ndarray
    ) -> float:
        if template is None:
            return 0.0

        if (
            template.shape[0] > frame_gray.shape[0]
            or template.shape[1] > frame_gray.shape[1]
        ):
            return 0.0

        result = cv2.matchTemplate(
            frame_gray,
            template,
            cv2.TM_CCOEFF_NORMED
        )

        _, max_score, _, _ = cv2.minMaxLoc(result)
        return float(max_score)

    def _extract_hud_regions(
        self,
        frame_gray: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extracts expected Pokémon Unite HUD areas from a landscape gameplay frame.
        """
        h, w = frame_gray.shape[:2]

        return {
            "timer": frame_gray[
                int(h * 0.00):int(h * 0.18),
                int(w * 0.35):int(w * 0.65)
            ],
            "minimap": frame_gray[
                int(h * 0.00):int(h * 0.40),
                int(w * 0.00):int(w * 0.30)
            ],
            "skills": frame_gray[
                int(h * 0.55):int(h * 1.00),
                int(w * 0.58):int(w * 1.00)
            ],
            "center": frame_gray[
                int(h * 0.12):int(h * 0.90),
                int(w * 0.10):int(w * 0.90)
            ],
            "full": frame_gray,
        }

    def _score_frame(
        self,
        frame_gray: np.ndarray
    ) -> Tuple[bool, bool, Dict[str, float]]:
        if not self.templates:
            raise RuntimeError("Cannot score frame: No templates loaded.")

        regions = self._extract_hud_regions(frame_gray)

        scores = {
            "timer": self._template_confidence(regions["timer"], self.templates.get("timer")),
            "skills": self._template_confidence(regions["skills"], self.templates.get("skills")),
            "minimap": self._template_confidence(regions["minimap"], self.templates.get("minimap")),
            "result": self._template_confidence(regions["center"], self.templates.get("result")),
            "time_up": self._template_confidence(regions["center"], self.templates.get("time_up")),
            "lobby": self._template_confidence(regions["full"], self.templates.get("lobby")),
        }

        timer_detected = scores["timer"] >= 0.88
        skills_detected = scores["skills"] >= 0.60
        minimap_detected = scores["minimap"] >= 0.55

        explicit_end = (
            scores["result"] >= 0.78
            or scores["time_up"] >= 0.78
            or scores["lobby"] >= 0.82
        )

        # Timer is never allowed to confirm gameplay alone.
        gameplay_confirmed = (
            timer_detected
            and (skills_detected or minimap_detected)
            and not explicit_end
        )

        return gameplay_confirmed, explicit_end, scores

    def scan_vod(self, vod_path: str, scan_interval_sec: int = 5) -> List[Tuple[float, float]]:
        print(f"\n🔍 [VOD Chopper] Scanning {vod_path} at 1 frame every {scan_interval_sec}s...")
        
        if not self.templates:
            print("\n❌ FATAL ERROR: No templates loaded!")
            print("Please take screenshots of the game UI and save them as PNGs in:")
            print(f"👉 {self.templates_dir}")
            print("Required files: gameplay_timer.png, skill_buttons.png, minimap.png, result_screen.png")
            return []

        if not os.path.exists(vod_path):
            raise FileNotFoundError(f"VOD not found: {vod_path}")

        cap = cv2.VideoCapture(vod_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_duration = total_frames / fps if fps > 0 else 0
        
        frame_step = max(1, int(fps * scan_interval_sec))
        current_frame = 0
        
        matches = []
        in_match = False
        current_match_start = 0.0
        last_confirmed_gameplay_time = 0.0

        start_confirmation_samples = 3
        end_confirmation_samples = 6
        
        min_match_duration_sec = 180.0
        max_match_duration_sec = 900.0  # 15 minutes safety limit

        consecutive_gameplay = 0
        consecutive_non_gameplay = 0
        
        progress_step = max(1, int(total_frames / 10))

        while current_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret: break
            
            timestamp_sec = current_frame / fps
            
            h, w = frame.shape[:2]
            scale = 1080 / h
            new_w = int(w * scale)
            frame_resized = cv2.resize(frame, (new_w, 1080))
            frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            
            gameplay_detected, explicit_end, scores = self._score_frame(frame_gray)

            if (
                gameplay_detected
                or explicit_end
                or max(scores.values(), default=0.0) >= 0.60
            ):
                print(
                    f"   [DEBUG] {timestamp_sec:.0f}s | "
                    f"gameplay={gameplay_detected} | "
                    f"end={explicit_end} | "
                    f"timer={scores['timer']:.2f} | "
                    f"skills={scores['skills']:.2f} | "
                    f"minimap={scores['minimap']:.2f} | "
                    f"result={scores['result']:.2f} | "
                    f"time_up={scores['time_up']:.2f} | "
                    f"lobby={scores['lobby']:.2f}"
                )

            if gameplay_detected:
                consecutive_gameplay += 1
                consecutive_non_gameplay = 0
                last_confirmed_gameplay_time = timestamp_sec

                if not in_match and consecutive_gameplay >= start_confirmation_samples:
                    in_match = True
                    confirmation_delay = (start_confirmation_samples - 1) * scan_interval_sec
                    current_match_start = max(0.0, timestamp_sec - confirmation_delay - 5.0)
                    print(f"   ⚔️ Confirmed match start around {current_match_start:.0f}s")
            else:
                consecutive_gameplay = 0

                if in_match:
                    consecutive_non_gameplay += 1
                    should_end = (explicit_end or consecutive_non_gameplay >= end_confirmation_samples)

                    if should_end:
                        match_end = min(total_duration, last_confirmed_gameplay_time + 10.0)
                        match_duration = match_end - current_match_start

                        if min_match_duration_sec <= match_duration <= max_match_duration_sec:
                            matches.append((current_match_start, match_end))
                            print(f"   🛑 Confirmed match end at {match_end:.0f}s (Duration: {match_duration / 60:.1f} mins)")
                        else:
                            print(f"   ⚠️ Discarded short/invalid detection ({match_duration:.0f}s)")

                        in_match = False
                        consecutive_non_gameplay = 0

            if in_match and (timestamp_sec - current_match_start) > max_match_duration_sec:
                print(
                    f"   ⚠️ Detection exceeded {max_match_duration_sec / 60:.0f} minutes. "
                    "Rejecting it as a false-positive match."
                )
                in_match = False
                consecutive_gameplay = 0
                consecutive_non_gameplay = 0
            
            if current_frame % progress_step < frame_step:
                print(f"   ... {(current_frame/total_frames)*100:.0f}% scanned")
                
            current_frame += frame_step

        if in_match:
            match_end = min(total_duration, last_confirmed_gameplay_time + 10.0)
            match_duration = match_end - current_match_start
            
            if min_match_duration_sec <= match_duration <= max_match_duration_sec:
                matches.append((current_match_start, match_end))
                print(f"   🛑 Confirmed match end at {match_end:.0f}s (Duration: {match_duration / 60:.1f} mins)")

        validated_matches = []
        for start_t, end_t in matches:
            duration = end_t - start_t
            coverage_ratio = (duration / total_duration if total_duration > 0 else 0.0)

            if total_duration > 1800 and coverage_ratio > 0.80:
                print(f"⚠️ Rejected suspicious full-VOD match detection: {coverage_ratio * 100:.1f}% coverage.")
                continue

            validated_matches.append((start_t, end_t))

        matches = validated_matches
        cap.release()
        print(f"✅ [VOD Chopper] Scan complete! Found {len(matches)} valid matches.")
        return matches

    def slice_vod(self, vod_path: str, matches: List[Tuple[float, float]]):
        if not matches:
            print("⚠️ No matches to slice.")
            return

        vod_name = os.path.splitext(os.path.basename(vod_path))[0]
        
        print(f"\n🔪 [VOD Chopper] Slicing {len(matches)} matches (Lossless Copy)...")
        
        for i, (start_t, end_t) in enumerate(matches, 1):
            output_file = os.path.join(self.output_dir, f"{vod_name}_match_{str(i).zfill(2)}.mp4")
            duration = end_t - start_t
            
            cmd = [
                "ffmpeg", "-y", "-v", "error", 
                "-ss", str(start_t), 
                "-t", str(duration), 
                "-i", vod_path, 
                "-c", "copy", 
                output_file
            ]
            
            subprocess.run(cmd)
            print(f"   💾 Saved: {output_file}")
            
        print("🎉 [VOD Chopper] All matches safely extracted to data/gameplay_segments!")

if __name__ == "__main__":
    chopper = VODChopper()
    raw_dir = "data/raw_vods"
    os.makedirs(raw_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("🎬 VOD CHOPPER & DOWNLOADER")
    print("="*50)
    print("1. Scan & Slice an existing local video")
    print("2. Download from YouTube/Twitch, then Scan & Slice")
    print("="*50)
    
    choice = input("\nEnter choice [1/2]: ").strip()
    target_video = None
    
    if choice == "1":
        valid_exts = (".mp4", ".mkv", ".webm", ".mov")
        files = [f for f in os.listdir(raw_dir) if f.lower().endswith(valid_exts)]
        
        if not files:
            print(f"\n❌ No videos found in '{raw_dir}'. Please add some or use Option 2.")
        else:
            print("\n📂 Available Videos:")
            for idx, f in enumerate(files, 1):
                print(f"  {idx}. {f}")
                
            vid_choice = input(f"\nSelect video [1-{len(files)}]: ").strip()
            try:
                target_video = os.path.join(raw_dir, files[int(vid_choice) - 1])
            except (ValueError, IndexError):
                print("❌ Invalid selection.")
                
    elif choice == "2":
        if not shutil.which("yt-dlp"):
            print("\n❌ FATAL: 'yt-dlp' is not installed or not in PATH.")
            print("Please install it by running: pip install yt-dlp")
        else:
            url = input("\n🔗 Enter YouTube/Twitch URL: ").strip()
            if url:
                print("\n📥 Starting Download (Highest Quality MP4)...")
                
                safe_name = f"vod_download_{uuid.uuid4().hex[:6]}.mp4"
                expected_filename = os.path.join(raw_dir, safe_name)
                
                download_cmd = [
                    "yt-dlp", 
                    "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "--merge-output-format", "mp4", 
                    "-o", expected_filename,        
                    url
                ]
                
                try:
                    subprocess.run(download_cmd, check=True)
                    
                    if os.path.exists(expected_filename):
                        print(f"\n✅ Download complete: {safe_name}")
                        target_video = expected_filename
                    else:
                        print("\n❌ Download failed or file not found.")
                except subprocess.CalledProcessError as e:
                    print(f"\n❌ yt-dlp encountered an error: {e}")
            else:
                print("❌ No URL provided.")
    else:
        print("❌ Invalid choice.")

    if target_video and os.path.exists(target_video):
        print("\n🚀 Initiating Chopper Sequence...")
        matches = chopper.scan_vod(target_video)
        chopper.slice_vod(target_video, matches)
    else:
        print("\n🛑 Exiting without processing.")