import os
import random

class MemeManager:
    def __init__(self, meme_dir: str = "assets/memes"):
        self.meme_dir = meme_dir
        os.makedirs(self.meme_dir, exist_ok=True)
        
        # 🧠 The Meme Brain: Map your 20+ memes to specific vibes here!
        self.library = {
            "funny_fail": ["willem_dafoe.mp4", "disappointed_fan.mp4", "windows_error.mp4"],
            "hype": ["lets_go_reaction.mp4", "mind_blown.mp4", "crowd_cheer.mp4"],
            "chaotic": ["spongebob_fire.mp4", "everything_is_fine.mp4"],
            "sneaky": ["metal_gear_alert.mp4", "hiding_meme.mp4"],
            "boring": [] # No memes for boring clips (they should be trashed anyway!)
        }

    def fetch_matching_meme(self, vibe_tag: str) -> str:
        """Pulls a random meme that perfectly matches the detected vibe."""
        available_memes = self.library.get(vibe_tag, [])
        
        if not available_memes:
            print(f"[Meme Manager] No memes configured for vibe: '{vibe_tag}'")
            return None
            
        # Pick a random meme from the correct category so the videos don't get repetitive
        selected_meme = random.choice(available_memes)
        meme_path = os.path.join(self.meme_dir, selected_meme)
        
        if os.path.exists(meme_path):
            return meme_path
        else:
            print(f"[Meme Manager] Error: File {selected_meme} not found in {self.meme_dir}!")
            return None