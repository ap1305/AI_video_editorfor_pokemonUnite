import json
import os
from typing import Dict, Any

def generate_edit_decision_file(qwen_pacing_data: dict, output_path: str = "data/temp/master_edf.json") -> Dict[str, Any]:
    """
    The Deterministic EDF Director. 
    Replaces the GPT-4o call. Maps Qwen's semantic labels directly to FFmpeg instructions using hardcoded rule engines.
    """
    print("🎬 [Phase 4] Executing Deterministic EDF Translation Matrix...")
    
    if not qwen_pacing_data:
        print("❌ [EDF Director] Error: Missing Qwen pacing data.")
        return {}

    # ---------------------------------------------------------
    # 1. THE CINEMATIC RULE ENGINE (The Brains)
    # ---------------------------------------------------------
    BEAT_RULES = {
        "SETUP": {"risk": 0.8, "speed": 1.5, "cam": "none", "audio": "none"},
        "PRESSURE": {"risk": 0.5, "speed": 1.0, "cam": "none", "audio": "low_rumble"},
        "CHAOS": {"risk": 0.2, "speed": 1.0, "cam": "camera_shake", "audio": "none"},
        "CLIMAX": {"risk": 0.05, "speed": 0.8, "cam": "zoom_in_center", "audio": "bass_impact"}, # 0.8 = slight slow-mo
        "RELEASE": {"risk": 0.3, "speed": 1.0, "cam": "none", "audio": "swoosh"},
        "AFTERMATH": {"risk": 0.9, "speed": 2.0, "cam": "none", "audio": "fade_out"}
    }

    EVENT_MEME_MAP = {
        "high_impact_ko": "hype",
        "clutch_escape": "sweating_survival",
        "objective_steal": "troll_face",
        "player_defeated": "sad_violin"
    }

    editing_plan = []
    meme_candidates = []
    
    # ---------------------------------------------------------
    # 2. TRANSLATE PACING BEATS INTO FFMPEG CUTS
    # ---------------------------------------------------------
    narrative_beats = qwen_pacing_data.get("narrative_beats", [])
    
    for beat in narrative_beats:
        beat_type = beat.get("beat", "SETUP")
        rules = BEAT_RULES.get(beat_type, BEAT_RULES["SETUP"])
        
        # Calculate safe speedup (prevent clips from becoming less than 1 second long)
        duration = beat["end_timestamp"] - beat["start_timestamp"]
        final_speed = rules["speed"]
        if final_speed > 1.0 and (duration / final_speed) < 1.0:
            final_speed = max(1.0, duration) # Cap the speedup so it doesn't break the edit

        playback_action = {"type": "normal"}
        if final_speed > 1.0: playback_action = {"type": "speedup", "speed": round(final_speed, 1)}
        elif final_speed < 1.0: playback_action = {"type": "slowmo", "speed": round(final_speed, 1)}

        editing_plan.append({
            "start": beat["start_timestamp"],
            "end": beat["end_timestamp"],
            "segment_type": beat_type.lower(),
            "intensity": 1.0 - rules["risk"], # Inverse of risk
            "retention_risk": rules["risk"],
            "playback_action": playback_action,
            "camera_fx": {"type": rules["cam"], "strength": 0.5 if rules["cam"] != "none" else 0.0},
            "audio_fx": {"type": rules["audio"]}
        })

    # ---------------------------------------------------------
    # 3. TRANSLATE EVENTS INTO MEMES
    # ---------------------------------------------------------
    semantic_events = qwen_pacing_data.get("semantic_events", [])
    for event in semantic_events:
        e_type = event.get("event_type")
        if e_type in EVENT_MEME_MAP and event.get("confidence", 0) > 0.85:
            meme_candidates.append({
                "type": EVENT_MEME_MAP[e_type],
                "trigger_timestamp": event["timestamp"],
                "confidence": event["confidence"]
            })

    # ---------------------------------------------------------
    # 4. ASSEMBLE FINAL MASTER EDF
    # ---------------------------------------------------------
    master_edf = {
        "visual_summary": "Deterministically mapped from Qwen semantic timeline.",
        "editing_plan": editing_plan,
        "meme_candidates": meme_candidates,
        "music_profile": {
            "genre": "high_energy_edm" if any(b.get("beat") in ["CHAOS", "CLIMAX"] for b in narrative_beats) else "chill_lofi",
            # Dynamically build the energy curve based on the beats
            "energy_curve": [round(1.0 - BEAT_RULES.get(b.get("beat", "SETUP"))["risk"], 2) for b in narrative_beats]
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(master_edf, f, indent=4)

    print(f"✅ [Phase 4] Deterministic EDF created with {len(editing_plan)} cuts and {len(meme_candidates)} meme triggers.")
    return master_edf