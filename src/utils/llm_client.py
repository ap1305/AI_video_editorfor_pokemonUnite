import time
import sys
from openai import OpenAI, APIConnectionError, APIError
from typing import List, Dict, Any
import base64
from PIL import Image
import io

def encode_image_for_qwen(image_path: str) -> str:
    """Resizes and converts a local image into a web-safe Base64 string for the API."""
    try:
        with Image.open(image_path) as img:
            img.thumbnail((512, 512)) 
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{base64_str}"
    except Exception as e:
        print(f"Failed to encode image: {e}")
        return None

class JobAbortedException(Exception):
    """Custom exception raised when the user explicitly aborts a job."""
    pass

# 👇 We changed the default model name right here! 👇
def execute_with_colab_fallback(api_key: str, base_url: str, messages: list, model: str = "qwen-director"):
    """
    Executes an LLM call. If the connection fails, it pauses execution and 
    presents an interactive recovery menu to the user.
    """
    current_url = base_url
    
    while True:
        try:
            client = OpenAI(api_key=api_key, base_url=current_url)
            response = client.chat.completions.create(
                model=model, 
                messages=messages,
                temperature=0.1,
                max_tokens=1500 # 👈 Bump this up so it doesn't cut off mid-sentence!
            )
            return response.choices[0].message.content
            
        except (APIConnectionError, APIError) as e:
# ... [the rest of your code stays exactly the same] ...
            print("\n" + "!"*50)
            print(f"[CRITICAL ERROR] AI Provider Connection Lost")
            print(f"Details: {e}")
            print("!"*50)
            print("1. Enter new Ngrok/Colab URL (Resume immediately)")
            print("2. Pause and close factory (Saves state in DB to do later)")
            print("3. Abort this specific video (Marks as FAILED and skips)")
            print("!"*50)
            
            choice = input("\nEnter choice (1, 2, or 3): ").strip()
            
            if choice == "1":
                current_url = input("Paste the new URL here: ").strip()
                print(f"\n[Recovery] Retrying with {current_url}...\n")
                time.sleep(1)
                
            elif choice == "2":
                print("\n[Recovery] Factory shutting down. Job remains PENDING in database.")
                sys.exit(0) # Gracefully kills the entire script
                
            elif choice == "3":
                print("\n[Recovery] Aborting current job...")
                raise JobAbortedException("User manually aborted the job due to API failure.")
                
            else:
                print("\n[Error] Invalid choice. Retrying current URL...")
                time.sleep(1)



def build_advanced_qwen_payload(base64_frames: list) -> list:
    """
    Advanced prompt optimized for Qwen2.5-VL esports highlight and viral moment detection.
    Designed specifically for Pokémon Unite short-form content extraction.
    """

    # 👇 UPGRADED: The Bodyguard is now ruthless! No more dying or boring walking. 👇
    system_prompt = """
    You are an extremely ruthless and strict AI Video Judge for Pokémon Unite gameplay.

    You are analyzing 6 SEQUENTIAL FRAMES sampled across a 15-second gameplay clip. 

    Your job is to look at the CENTER of the screen (the main player) and determine whether this clip contains a HIGH-ENGAGEMENT, VIRAL gameplay moment.
    DEFAULT TO REJECTION. You must actively look for reasons to score this clip poorly.

    --------------------------------------------------
    PRIMARY OBJECTIVE & TRACKING
    --------------------------------------------------
    CENTER PLAYER PERSISTENCE: The center player in the EARLY frames is the primary subject. Track this SAME player across all subsequent frames.

    Detect whether the clip contains:
    - HYPE: PvP (Player vs Player) combat escalation, contested objective fights, multi-KOs.

    --------------------------------------------------
    THE ANTI-HYPE RULES (CRITICAL - PENALIZE HEAVILY)
    --------------------------------------------------
    1. THE DEATH RULE: If the main center player dies or gets knocked out, this is a BAD clip. The MAXIMUM score you are allowed to give is 35.
    2. THE ZERO-ENEMY RULE: If the main player is just walking around, or if `max_enemies_visible_in_any_frame` is 0, this is boring downtime. The MAXIMUM score you are allowed to give is 35.
    3. THE FARMING RULE: If the player is just attacking wild Pokémon (like Rotom, Baltoy, or Swablu) WITHOUT an enemy player fighting them for it, it is boring. The MAXIMUM score is 45.
    4. THE COUNTDOWN TRAP: If `is_match_countdown_active` is true (giant yellow numbers counting down 10, 9, 8... or "Time's up!" screen), the action is over. The MAXIMUM score is 35.

    --------------------------------------------------
    CRITICAL SCORING RULES
    --------------------------------------------------
    AUTO LOW SCORE (<45):
    - The main player dies.
    - Walking around doing nothing.
    - MATCH COUNTDOWN or "Time's up!" screens.
    - UNCONTESTED objective farming.

    MID SCORE (45-75):
    - small PvP skirmish or single PvP KO.
    - average lane fight with sparse combat.

    HIGH SCORE (76-89):
    - intense PvP teamfight with multi-enemy engagement.
    - CONTESTED objective fights where the main player wins.

    ELITE VIRAL SCORE (90-100):
    - `total_enemy_kos_observed` is 2 or more in rapid succession by the main player.
    - 1v3 / 1v4 PvP outplay.
    - massive Unite combo that secures a team wipe.

    --------------------------------------------------
    OUTPUT FORMAT (STRICT JSON)
    --------------------------------------------------
    You must return ONLY a raw, valid JSON object. 
    The JSON must be syntactically valid and directly parsable by Python's json.loads().
    DO NOT wrap the response in markdown blocks (e.g., do not use ```json).

    To ensure proper internal reasoning, you MUST output the fields in this EXACT order:

    {
        "max_enemies_visible_in_any_frame": 2,
        "total_enemy_kos_observed": 3,
        "is_match_countdown_active": false,
        "reason": "1. [Describe player state]. 2. [Describe how the PvP action escalates]. 3. [Describe the climax]. 4. [Describe the final outcome].",
        "score": 92,
        "confidence": 0.95,
        "tier": "S-TIER",
        "climax_frame_index": 4,
        "tags": {
            "primary_emotion": "[choose from: hype, funny, clutch, chaotic, satisfying, shocking, meme, cinematic, boring]",
            "secondary_emotion": "descriptive_word",
            "action_type": "teamfight_or_event_name"
        }
    }
    """

    content = [{"type": "text", "text": system_prompt}]

    for idx, b64 in enumerate(base64_frames):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}"
            }
        })

    return [{
        "role": "user",
        "content": content
    }]