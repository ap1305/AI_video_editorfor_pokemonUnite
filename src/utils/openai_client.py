import os
import json
import re
from openai import OpenAI

def clean_ocr_text(raw_text: str) -> str:
    """
    The 'Event Extractor'. 
    Strips out static stream text and keeps only high-value action lines.
    """
    static_noise = [
        r"^\s*🎯\s*\[BULLSEYE\] Action found at", 
        "LIVE NOW POKEMON UNITE ROAD TO MASTER",
        r"FPS[;:]?\s*\d*", 
        "BATTLE DATA MOVESET PERFORMANCE",
        "VIEW BATTLE REPORT", 
        "DETAILS BATTLE DATA",
        "DMG", "DEALT", "TAKEN", "RECOVERY",
        r"UN[LI]TE\s*MOV[IE]?\w*",  
        "POTION",
        r"ENERGY\s*HELD"
    ]

    # 👇 FIX 1: Removed 'DEFEATED' so we stop rendering clips of you dying!
    other_keywords = ["STREAK", "STOLEN", "GOAL", "DEFENDING", "ATTACKING", "VICTORIOUS"]
    ko_pattern = re.compile(r'\b\d?-?K[O0]\b', re.IGNORECASE)

    cleaned_lines = []
    lines = raw_text.split('\n')

    for line in lines:
        if not line.strip():
            continue
            
        if any(end_word in line for end_word in ["BATTLE REPORT", "DMG", "DETAILS BATTLE DATA"]):
            continue

        clean_line = line
        
        for noise in static_noise:
            clean_line = re.sub(noise, "", clean_line, flags=re.IGNORECASE)

        clean_line = re.sub(r'\s+', ' ', clean_line).strip()

        has_ko = bool(ko_pattern.search(clean_line))
        has_other_action = any(kw in clean_line.upper() for kw in other_keywords)

        if has_ko or has_other_action:
            cleaned_lines.append(clean_line)

    return "\n".join(cleaned_lines)


def get_best_timestamps_from_ocr(ocr_file_path: str, api_key: str) -> list:
    """
    Acts as 'The Director'. Reads the scrubbed log, evaluates narrative arcs,
    and returns a highly structured JSON of the best viral moments.
    """
    if not os.path.exists(ocr_file_path):
        print(f"[Director] Error: Could not find {ocr_file_path}")
        return []
        
    with open(ocr_file_path, 'r', encoding='utf-8') as file:
        raw_ocr_log = file.read()

    print("[Event Extractor] Scrubbing raw Colab text...")
    clean_log_content = clean_ocr_text(raw_ocr_log)
    
    clean_log_path = "data/inputs/cleaned_timeline.txt"
    os.makedirs("data/inputs", exist_ok=True)
    with open(clean_log_path, 'w', encoding='utf-8') as file:
        file.write(clean_log_content)

    client = OpenAI(api_key=api_key)

    # ---------------------------------------------------------
    # THE ULTIMATE EDITOR PROMPT (UPGRADED FOR TIMING & QUALITY)
    # ---------------------------------------------------------
    system_prompt = """
    You are an elite video editor for Pokémon Unite shorts. 
    I will provide a structured OCR timeline of action events from a match.
    
    Your job is to extract the 5 to 8 absolute best moments.
    
    CRITICAL TIMING RULES (DO NOT IGNORE):
    1. MINIMUM DURATION: A clip MUST be between 12 and 18 seconds long. Never pick a 2-second or 5-second window.
    2. THE BUILD-UP: Set the `start_time` at least 5 to 7 seconds BEFORE the actual KO/Action happens so the viewer understands the context.
    3. THE PAYOFF: Set the `end_time` at least 3 to 4 seconds AFTER the climax so the video doesn't end abruptly. Let the clip breathe!
    
    CRITICAL CONTENT RULES:
    1. NO DEATHS: Do not select clips where the main player simply dies or gets defeated.
    2. NO BORING WALKING: Ignore isolated events where nothing happens (e.g., just scoring 5 points alone).
    3. STRICT JSON NUMBERS: Do NOT use leading zeros for timestamps (e.g., use 7.0, NEVER 07.0).
    
    PRIORITY ORDER (Highest to Lowest):
    - 4KO / 3KO streaks
    - Final Stretch teamfights
    - Objective steals or defenses
    - Multi-event momentum swings
    
    You MUST output ONLY a valid JSON array. No markdown, no conversational text.
    
    Format EXACTLY like this example:
    [
      {
        "start_time": 540.0,
        "climax_time": 547.0,
        "end_time": 554.0,
        "priority_score": 98,
        "confidence": 0.97,
        "event_type": "quadra_kill",
        "emotion": "hype",
        "title": "4KO Final Stretch Domination",
        "reasoning": "Fight escalates from 2KO to 4KO streak."
      }
    ]
    """

    print("[Director] Analyzing narrative arcs and calculating priority scores...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": clean_log_content} 
            ],
            temperature=0.1 
        )

        raw_output = response.choices[0].message.content.strip()
        
        # --- BUG-FREE JSON SANITIZATION BLOCK ---
        md_ticks = chr(96) * 3 
        md_json = md_ticks + "json"
        
        if raw_output.startswith(md_json): raw_output = raw_output[7:]
        elif raw_output.startswith(md_ticks): raw_output = raw_output[3:]
        if raw_output.endswith(md_ticks): raw_output = raw_output[:-3]
        
        raw_output = raw_output.strip()
        
        start_idx = raw_output.find('[')
        end_idx = raw_output.rfind(']')
        if start_idx != -1 and end_idx != -1:
            raw_output = raw_output[start_idx:end_idx+1]
            
        # REGEX FIX: Automatically delete leading zeros on integers/floats
        raw_output = re.sub(r'(:\s*)0+(\d+)', r'\1\2', raw_output)
            
        clip_data = json.loads(raw_output)
        
        # 👇 FIX 2: Hard-coding the minimum duration just in case ChatGPT hallucinates
        for clip in clip_data:
            duration = clip["end_time"] - clip["start_time"]
            if duration < 10.0:
                clip["start_time"] = max(0.0, clip["climax_time"] - 6.0)
                clip["end_time"] = clip["climax_time"] + 4.0

        report_path = "data/director_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(clip_data, f, indent=4)
            
        print(f"✅ [Director] Master report saved to {report_path}!")
        
        clip_data.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
        return clip_data
        
    except json.JSONDecodeError as e:
        print(f"[Director] Failed to parse JSON timestamps: {e}. Raw Output was:\n{raw_output}")
        return []
    except Exception as e:
        print(f"[Director] Unexpected error: {e}")
        return []
def run_macro_scout(candidate_windows_path: str, api_key: str) -> dict:
    """
    The 'Macro Scout'. Takes mathematically pre-filtered candidate windows,
    evaluates their narrative arcs locally, and picks the absolute best clips.
    """
    if not os.path.exists(candidate_windows_path):
        print(f"❌ [Macro Scout] Error: Could not find {candidate_windows_path}")
        return {}

    with open(candidate_windows_path, 'r', encoding='utf-8') as file:
        candidate_windows = json.load(file)

    # To save tokens and focus on top-tier plays, we only evaluate the top 8 candidates
    top_candidates = candidate_windows[:8]
    candidates_string = json.dumps(top_candidates, indent=2)

    client = OpenAI(api_key=api_key)

    system_prompt = """
You are the 'Macro Scout' for an autonomous esports video editor.
Evaluate the following structured candidate windows from a MOBA match.
Your job is to act as the executive producer and select the absolute best, most highly-retainable moments to send to the final rendering bay.

CRITICAL RULES:

1. OVERLAP PROTECTION & TIE-BREAKING:
NEVER select multiple clips that share more than 30% of the same timeframe.
If two candidate windows overlap significantly, choose ONLY the stronger one.
If two clips are similarly strong across the board, use this strict tie-breaker hierarchy:
  1st: Higher `importance_score`
  2nd: `is_late_game` is true
  3rd: Higher `audio_density`
  4th: Presence of `ocr_events`

2. QUALITY GATING:
Select up to 3 clips.
ONLY select clips that are genuinely viral-worthy. Base your judgment strictly on the provided data: high `importance_score`, high `audio_density`, presence of `ocr_events`, and `is_late_game`.
If only 1 clip meets a high quality threshold (confidence > 0.85), return exactly 1. Do not force 3.

3. CLIP DIVERSITY:
Prefer varied emotional pacing and gameplay structure across selected clips when possible.
Avoid selecting clips that feel narratively identical.

4. NARRATIVE AWARENESS (STRICT ENUMS):
Assign one of the following narrative types ONLY:
[FINAL_PUSH, TEAMFIGHT, OBJECTIVE_SECURE, OBJECTIVE_STEAL, CLUTCH_ESCAPE, SOLO_OUTPLAY, COMEBACK, EARLY_SNOWBALL, CHAOTIC_FIGHT]

5. RETENTION PRIORITY (STRICT ENUMS):
Rank clips based on estimated viewer retention. You must use ONLY one of these values:
[HIGH, MEDIUM, LOW]

6. PRIMARY SIGNAL (STRICT ENUMS & DEFINITIONS):
Identify the core driver of the hype using ONLY one of these values:
[AUDIO_DRIVEN, VISUAL_CHAOS, OCR_CONFIRMED, MULTI_MODAL].
*Use MULTI_MODAL only when at least two major signal categories strongly contribute to the clip selection.*

7. ANTI-HALLUCINATION & CONSTRAINTS:
DO NOT invent gameplay details not supported by the metadata.
The `selection_confidence` field MUST be a float between 0.0 and 1.0.

8. STRICT SCHEMA:
Do not add extra fields, explanations, markdown, comments, or additional keys outside the required JSON structure. You MUST output ONLY a valid JSON object.

Format EXACTLY like this:
{
  "selected_clips": [
    {
      "clip_number": 1,
      "window_id": "CANDIDATE_13",
      "start_time": 675.0,
      "end_time": 707.0,
      "narrative_type": "FINAL_PUSH",
      "primary_signal": "MULTI_MODAL",
      "retention_priority": "HIGH",
      "selection_confidence": 0.94,
      "narrative_reasoning": "Highest tension sequence of the match. Won tie-breaker over Candidate 12 due to higher importance score and late-game positioning."
    }
  ]
}
"""

    print("🧠 Routing Candidate Windows through OpenAI API (GPT-4o-mini)...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here are the top candidates:\n{candidates_string}"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"} # Forces a valid JSON object return
        )

        raw_output = response.choices[0].message.content.strip()
        final_blueprints = json.loads(raw_output)
        
        report_path = "data/inputs/final_blueprints.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(final_blueprints, f, indent=4)
            
        print(f"✅ [Macro Scout] Final blueprints saved to {report_path}!")
        return final_blueprints

    except Exception as e:
        print(f"❌ [Macro Scout] Unexpected error: {e}")
        return {}