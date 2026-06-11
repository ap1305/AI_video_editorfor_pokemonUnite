import os
import json
import random
import subprocess

class MusicDirector:
    def __init__(self, base_bgm_dir="assets/audio/bgm", history_file="data/creative/music_usage_history.json"):
        self.base_bgm_dir = base_bgm_dir
        self.history_file = history_file
        self.valid_extensions = {".mp3", ".m4a", ".wav", ".aac"}
        self.history_limit = 50

        # Ensure directory structure exists
        for mood in ["hype", "funny", "sad", "tense", "neutral"]:
            os.makedirs(os.path.join(self.base_bgm_dir, mood), exist_ok=True)
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)

    def _get_mood(self, story: dict) -> str:
        """Determines the music mood based on the story contract."""
        if float(story.get("confidence", 1.0)) < 0.6:
            return "neutral"

        mech = str(story.get("comedy_mechanism", "")).lower()
        outcome = str(story.get("actual_outcome", "")).lower()
        desc = str(story.get("scene_description", "")).lower()

        combined_text = f"{mech} {outcome} {desc}"

        # Safe Rule Mapping (Avoids false "hype" on player death)
        if any(word in combined_text for word in ["rayquaza", "zapdos", "objective", "close combat", "final fight", "tense"]):
            return "tense"
        if any(word in combined_text for word in ["fail", "mistake", "whiff", "backfire", "panic", "enemy_overconfidence"]):
            return "funny"
        if any(word in combined_text for word in ["player is knocked out", "ko'd by", "defeated by", "destroyed", "loses", "tragic"]):
            return "sad"
        if any(word in combined_text for word in ["clutch", "survives", "scores", "comeback", "exciting", "win", "kos enemy", "multi-ko"]):
            return "hype"

        return "neutral"

    def _get_history(self) -> list:
        """Reads recent tracks history."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                    return data.get("recent_tracks", [])
            except Exception:
                return []
        return []

    def save_history(self, track_path: str):
        """Updates history. Called by pipeline ONLY after successful render."""
        history = self._get_history()
        history.append(track_path)
        if len(history) > self.history_limit:
            history = history[-self.history_limit:]
        with open(self.history_file, "w") as f:
            json.dump({"recent_tracks": history}, f, indent=4)

    def _validate_audio(self, filepath: str) -> bool:
        """Uses ffprobe to confirm the file has an audio stream AND duration > 0."""
        if not os.path.exists(filepath): 
            return False
            
        # 1. Check that an actual audio stream exists (Prevents broken media files)
        cmd_stream = [
            "ffprobe", "-v", "error", "-select_streams", "a:0", 
            "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1", filepath
        ]
        try:
            res_stream = subprocess.run(cmd_stream, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            if "audio" not in res_stream.stdout.strip().lower():
                return False
                
            # 2. Check duration
            cmd_dur = [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filepath
            ]
            res_dur = subprocess.run(cmd_dur, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            duration = float(res_dur.stdout.strip())
            return duration > 0.0
            
        except Exception as e:
            print(f"⚠️ [Music Director] FFprobe validation failed for {filepath}: {e}")
            return False

    def _pick_track(self, target_mood: str) -> tuple:
        """Selects a valid track, applying fallbacks and avoiding history."""
        moods_to_try = [target_mood] if target_mood == "neutral" else [target_mood, "neutral"]
        history = self._get_history()

        for current_mood in moods_to_try:
            folder_path = os.path.join(self.base_bgm_dir, current_mood)
            if not os.path.exists(folder_path):
                continue

            all_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in self.valid_extensions
            ]

            if not all_files:
                continue

            unused_files = [f for f in all_files if f not in history]
            candidates = unused_files if unused_files else all_files

            random.shuffle(candidates)
            
            for candidate in candidates:
                if self._validate_audio(candidate):
                    return candidate, current_mood

        return None, "neutral"

    def generate_plan(self, story_contract: dict) -> dict:
        """Returns the BGM contract."""
        print(f"🎵 [Music Director] Analyzing story for BGM mood...")
        
        target_mood = self._get_mood(story_contract)
        track_path, final_mood = self._pick_track(target_mood)

        if track_path:
            clean_path = track_path.replace("\\", "/")
            print(f"   Selected: {clean_path} (Mood: {final_mood})")
            return {
                "enabled": True,
                "music_mood": final_mood,
                "music_path": clean_path,
                "gain": 0.18,      
                "fade_in": 0.5,    
                "fade_out": 1.0,   
                "loop_to_duration": True 
            }
        else:
            print(f"⚠️ [Music Director] No valid tracks found. Disabling BGM.")
            return {
                "enabled": False,
                "music_mood": "neutral",
                "music_path": None,
                "skip_reason": "No valid music found"
            }