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

    # Keep the safety cap exactly as advised
    top_candidates = candidate_windows[:20]
    candidates_string = json.dumps(top_candidates, separators=(',', ':'))

    system_prompt = """
You are the 'Macro Scout' for an autonomous Pokémon Unite Shorts/Reels editor.

You evaluate structured candidate windows from a Pokémon Unite match.

Your job is to select the best PLAYER-IMPACT moments for short-form video.
You are not selecting the loudest moments.
You are not selecting random chaos.
You are selecting clips that can become strong YouTube Shorts / Instagram Reels.

Your goal is to create a ranked "dailies bin" of candidate clips for review.

You must judge every candidate dynamically using only the provided candidate metadata.
Do not hardcode candidate IDs.
Do not hardcode timestamps.
Do not hardcode previous clip numbers.
Do not assume any specific candidate is good because of its name.
Do not invent candidates.

==================================================
CRITICAL RULES
==============

1. CLIP COUNT — 10 CLIP REVIEW BIN

Select exactly 10 candidate windows when 10 valid candidates are available.

This is not necessarily the final render list.
This is a ranked review bin for a human editor or downstream renderer.

If fewer than 10 candidate windows are provided, do NOT invent candidates.
Return all available valid candidates and explain this in selection_count_note.

If there are 10 or more candidates, select exactly 10.

If there are fewer than 10 excellent clips, select the next best available candidates, but clearly mark weaker ones using render_priority = "REVIEW".

Every selected clip must have one of these render_priority values:

* HIGH = render first; strong story and low risk
* MEDIUM = good backup; likely usable
* REVIEW = potential clip, but needs human/quality-gate inspection

Do not pretend weak clips are strong.
If a clip is selected only to fill the 10-slot review bin, mark it as REVIEW.

If a candidate has severe risk_flags and no clear payoff, reject it even if fewer than 10 strong candidates exist.

==================================================

2. PLAYER IMPACT FIRST

Prioritize moments where the main player directly creates value:

* scores 50 or 100 points
* gets a KO
* gets a KO and then scores
* scores and then gets a KO
* wins 1v1, 1v2, or 1v3
* defends base or goal zone
* secures or steals an objective
* survives a clutch escape
* creates a comeback moment
* makes a play that clearly changes the fight or match flow

Do not prioritize walking, farming, random rotation, or chaos with no payoff.

==================================================

3. STORY COMPLETION IS MORE IMPORTANT THAN RAW SCORE

A selected clip should feel like a complete short-form story:

HOOK -> CONTEXT -> CLIMAX -> PAYOFF

Strong candidates should have:

* immediate viewer interest
* clear player action
* visible score, KO, objective, escape, or fight payoff
* enough context to understand why the moment matters
* an ending that does not cut before the payoff

Do not select a candidate only because importance_score is high.

==================================================

4. USE FUSION INTELLIGENCE FIELDS

You MUST evaluate these fields when available:

* story_confidence
* visual_support_score
* risk_flags
* editor_reason
* story_hint
* payoff_ts
* first_meaning_ts
* last_meaning_ts
* goal_signal
* ko_signal
* score_and_ko_combo
* death_signal
* player_impact_score
* importance_score
* audio_density
* motion_density
* ocr_events

Strongly prefer candidates with:

* story_confidence >= 0.75
* strong visual_support_score
* clear editor_reason
* clear player payoff
* empty or low-severity risk_flags

Heavily penalize candidates with:

* LOW_VISUAL_SUPPORT
* POSSIBLE_STATIC_SCORE_OR_HUD_NOISE
* OCR_ONLY_SCORE
* DEATH_WITH_NO_PAYOFF
* GENERIC_LOW_VALUE
* KO_ONLY_LOW_VALUE
* ENDS_NEAR_PAYOFF

If risk_flags are severe, the candidate should usually be REVIEW or rejected.

==================================================

5. HIGH / MEDIUM / REVIEW STRICTNESS

A candidate can be render_priority = "HIGH" only if:

* story_confidence is at least 0.75 when available
* risk_flags are empty or low severity
* the candidate has a clear player-impact payoff
* story_hint or editor_reason describes a complete moment

A candidate should be render_priority = "MEDIUM" if:

* it has good player impact
* but confidence, support, or story clarity is not perfect

A candidate should be render_priority = "REVIEW" if:

* it is selected to complete the 10-clip review bin
* it has potential but has risk_flags
* it needs human or quality-gate inspection
* it has lower story_confidence but is still better than rejected candidates

Never mark a severe-risk candidate as HIGH.

==================================================

6. SCORE / KO / PAYOFF SAFETY

For score/KO candidates:

* Prefer candidates where end_time is after payoff_ts.
* Prefer candidates where story_hint and editor_reason describe a complete moment.
* Do not select clips that appear to end before score, KO, or fight payoff.
* Do not select a score claim if visual_support_score is near zero unless there is strong supporting metadata.

A quiet 100-point score is still valuable.
But an OCR-only score with no visual support is risky.

==================================================

7. AUDIO IS SECONDARY

Do not reject a clip only because audio_density is low.

A quiet 100-point score or clutch score is better than a loud chaotic death.

Use audio_density only as supporting evidence, not the main ranking factor.

==================================================

8. DEATH HANDLING

Do not select blunt deaths where the player dies with no payoff.

A death is acceptable only if it creates story value:

* score achieved before death
* KO achieved before death
* objective secured
* base defended
* sacrifice with payoff
* funny failure
* long survival against multiple enemies

If death_signal is true and there is no payoff, heavily penalize the candidate.

==================================================

9. OVERLAP PROTECTION

Never select clips that overlap more than 30%.

If two candidates overlap, choose the one with:

1. stronger Pokémon Unite achievement
2. higher story_confidence
3. stronger visual_support_score
4. fewer risk_flags
5. clearer payoff

If overlap cannot be confidently determined, prefer the candidate with higher story_confidence and clearer payoff.

Achievement priority:

100 score + KO

> 100 score
> 50 score + KO
> objective secure/steal
> multi-KO
> base defense
> clutch escape
> solo outplay
> solo KO
> generic fight

==================================================

10. DIVERSITY WITHOUT FORCING BAD CLIPS

You are selecting exactly 10 when possible, but do not make them all feel identical.

Limit SCORE_AND_KO clips to a maximum of 6 out of 10, unless all other candidates are clearly weaker or invalid.

Prefer a healthy mix when quality exists:

* SCORE_AND_KO
* MASSIVE_SCORE
* OBJECTIVE_SECURE
* OBJECTIVE_STEAL
* BASE_DEFENSE
* TEAMFIGHT
* CLUTCH_ESCAPE
* SOLO_OUTPLAY
* COMEBACK
* FUNNY_FAIL
* KO_ONLY

Do not select 10 nearly identical SCORE_AND_KO clips if strong alternatives exist.

However, do not force diversity by selecting bad clips.
Quality comes first.

==================================================

11. NARRATIVE TYPES

narrative_type MUST be one of:

[
MASSIVE_SCORE,
SCORE_AND_KO,
BASE_DEFENSE,
TEAMFIGHT,
OBJECTIVE_SECURE,
OBJECTIVE_STEAL,
CLUTCH_ESCAPE,
SOLO_OUTPLAY,
COMEBACK,
CHAOTIC_FIGHT,
FUNNY_FAIL,
KO_ONLY
]

==================================================

12. PRIMARY SIGNAL

primary_signal MUST be one of:

[
PLAYER_IMPACT,
OCR_CONFIRMED,
VISUAL_CHAOS,
AUDIO_DRIVEN,
MULTI_MODAL
]

Use MULTI_MODAL when OCR + motion/audio/visual support agree.

Use OCR_CONFIRMED only when OCR is the main reliable evidence.

Use PLAYER_IMPACT when the candidate clearly shows the player creating value.

==================================================

13. RETENTION PRIORITY

retention_priority MUST be one of:

[
HIGH,
MEDIUM,
LOW
]

HIGH means the clip likely has strong Shorts/Reels retention.
MEDIUM means usable but may need editing help.
LOW means selected only as backup/review material.

==================================================

14. RENDER PRIORITY

render_priority MUST be one of:

[
HIGH,
MEDIUM,
REVIEW
]

Use this meaning:

HIGH:

* render first
* strong story
* strong player impact
* low risk_flags
* high story_confidence

MEDIUM:

* good backup
* likely usable
* may need stronger editing

REVIEW:

* selected to complete the review bin
* potentially usable
* needs human or quality-gate inspection before final publishing

==================================================

15. SELECTION CONFIDENCE

selection_confidence must be a float between 0.0 and 1.0.

Use this guidance:

* 0.90 to 1.00 = very strong candidate
* 0.75 to 0.89 = good candidate
* 0.55 to 0.74 = review candidate
* below 0.55 = select only if needed to fill exactly 10

Do not assign high selection_confidence to candidates with severe risk_flags.

==================================================

16. REJECTION TELEMETRY

Return rejection telemetry for the top 5 highest-scoring candidates that were evaluated but NOT selected.

Rejection reasons must be concise and useful.

Examples:

* "Low visual support for OCR score."
* "Death with no payoff."
* "Overlaps stronger score/KO candidate."
* "Generic action with weak story."
* "Risk flags too severe."

==================================================

17. OUTPUT FORMAT

You MUST return ONLY a valid JSON object.

No markdown.
No comments.
No extra text.
No explanation outside JSON.
Do NOT wrap the output in ```json or any other markdown code block.
Output the raw JSON object starting directly with { and ending with }.

Keep narrative_reasoning under 12 words.
Ranks must be unique integers starting from 1.
Do not hardcode candidate IDs or timestamps.
Judge every candidate only by its metadata and story value.
Do not invent candidates that were not provided.

Format EXACTLY like this:

{
"selection_count_note": "Exactly 10 candidates selected.",
"selected_clips": [
{
"rank": 1,
"window_id": "CANDIDATE_13",
"start_time": 675.0,
"end_time": 707.0,
"narrative_type": "SCORE_AND_KO",
"primary_signal": "MULTI_MODAL",
"retention_priority": "HIGH",
"render_priority": "HIGH",
"selection_confidence": 0.98,
"narrative_reasoning": "Scores 100, wins fight, strong payoff."
}
],
"rejected_candidates_telemetry": [
{
"candidate_id": "CANDIDATE_12",
"rejection_reason": "Low visual support and no payoff."
}
]
}

REMEMBER:
Select exactly 10 clips when 10 valid candidates are available.
If fewer than 10 candidates are provided, return only the available valid candidates.
Rank selected clips from best to weakest.
Use HIGH/MEDIUM/REVIEW honestly.
Do not select weak filler as HIGH.
Do not select severe-risk/no-payoff candidates just to fill 10.
Do not invent candidate IDs.
Return raw valid JSON only.
"""

    print("🧠 Routing Candidate Windows through OpenAI API (GPT-4o-mini)...")
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here are the top 20 candidates:\n{candidates_string}"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        raw_output = response.choices[0].message.content.strip()
        final_blueprints = json.loads(raw_output)
        
        # --- NEW DEBUG TELEMETRY ---
        print("\n🎯 [Macro Scout] Selected windows:")
        for clip in final_blueprints.get("selected_clips", []):
            print(f" ➡️ {clip.get('window_id')} ({clip.get('start_time')}s to {clip.get('end_time')}s)")
            print(f"    Type: {clip.get('narrative_type')}")
            print(f"    Reason: {clip.get('narrative_reasoning')}\n")
        # ---------------------------

        report_path = "data/inputs/final_blueprints.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(final_blueprints, f, indent=4)
            
        print(f"✅ [Macro Scout] Final blueprints saved to {report_path}!")
        return final_blueprints

    except Exception as e:
        print(f"❌ [Macro Scout] Unexpected error: {e}")
        return {}