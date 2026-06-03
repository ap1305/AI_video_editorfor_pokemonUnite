import yaml
import os

# Load config once
CONFIG_PATH = "config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {
        "editor": {
            "safety_confidence_threshold": 0.60,
            "ultra_climax_threshold": 0.95,
            "high_setup_threshold": 0.75,
            "dead_air_fast_threshold": 0.10,
            "dead_air_med_threshold": 0.30
        }
    }

def generate_story_profile(qwen_json: dict) -> dict:
    """
    Macro-level decisions: Extracts the true story type based on the climax event.
    """
    primary_climax_id = qwen_json.get("primary_climax_id")
    hook_timestamp = qwen_json.get("the_hook_timestamp")
    
    viral_potential = 0.0
    story_type = "unknown_event"
    
    # Hunt for the climax data to define the story
    for hl in qwen_json.get("highlight_segments", []):
        if hl.get("segment_id") == primary_climax_id:
            viral_potential = float(hl.get("importance", 0.0))
            story_type = hl.get("event_type", "unspecified_climax")
            break
            
    return {
        "primary_climax_id": primary_climax_id,
        "hook_timestamp": hook_timestamp,
        "viral_potential": viral_potential,
        "story_type": story_type
    }

def generate_segment_execution_plan(segment: dict, is_primary_climax: bool = False) -> dict:
    """
    Generates the strict Intent, the agnostic Render Hints, and the Explanations.
    """
    role = segment.get("narrative_role", "Dead_Air")
    importance = float(segment.get("importance", 0.0))
    confidence = float(segment.get("confidence", 0.0))
    thresholds = config.get("editor", {})

    # Shells
    editing_intent = {
        "pacing": "normal",
        "audio_policy": "normal",
        "effects": []
    }
    
    render_hints = {
        "requires_optical_flow": False,
        "requires_audio_boost": False,
        "hard_cut_allowed": True
    }
    
    explanation = {
        "intent_confidence": confidence,
        "why_decision_made": "default_fallback"
    }

    # 🛑 "DO NO HARM" OVERRIDE
    if confidence < thresholds.get("safety_confidence_threshold", 0.60):
        explanation["why_decision_made"] = "Safety fallback triggered due to low AI confidence."
        return {"editing_intent": editing_intent, "render_hints": render_hints, "explanation": explanation}

    # 🎬 PACING MATRIX
    if role == "Dead_Air":
        editing_intent["audio_policy"] = "mute"
        render_hints["hard_cut_allowed"] = True
        
        if importance <= thresholds.get("dead_air_fast_threshold", 0.10):
            editing_intent["pacing"] = "fast_forward"
            explanation["why_decision_made"] = "Extreme dead air identified. Safe to jump/skip."
        elif importance <= thresholds.get("dead_air_med_threshold", 0.30):
            editing_intent["pacing"] = "accelerate_heavy"
            explanation["why_decision_made"] = "Moderate dead air. Heavy acceleration applied."
        else:
            editing_intent["pacing"] = "accelerate_light"
            explanation["why_decision_made"] = "Light dead air. Minor acceleration applied to bridge action."
            
    elif role == "Setup":
        editing_intent["audio_policy"] = "pitch_correct"
        editing_intent["pacing"] = "accelerate_light"
        render_hints["hard_cut_allowed"] = False # Don't cut, we need the context
        
        if importance >= thresholds.get("high_setup_threshold", 0.75):
            explanation["why_decision_made"] = "High tension setup. Preserving visual context to build anticipation."
        else:
            explanation["why_decision_made"] = "Standard setup rotation."
            
    elif role == "Conflict":
        editing_intent["pacing"] = "normal"
        render_hints["hard_cut_allowed"] = False
        explanation["why_decision_made"] = "Active conflict. Maintaining standard pacing to preserve gameplay readability."

    elif role == "Climax":
        render_hints["hard_cut_allowed"] = False
        if is_primary_climax:
            editing_intent["effects"].append({"name": "primary_hook_flash", "priority": 100})
            
        if importance >= thresholds.get("ultra_climax_threshold", 0.95):
            editing_intent["pacing"] = "slow_mo"
            editing_intent["audio_policy"] = "boost"
            editing_intent["effects"].append({"name": "cinematic_zoom", "priority": 90})
            
            render_hints["requires_optical_flow"] = True
            render_hints["requires_audio_boost"] = True
            explanation["why_decision_made"] = "Ultra climax detected (>0.95 importance). Applying slow-mo, audio boost, and focus zoom."
        else:
            editing_intent["pacing"] = "normal"
            explanation["why_decision_made"] = "Standard climax. Leaving unmodified to let the play speak for itself."

    return {
        "editing_intent": editing_intent,
        "render_hints": render_hints,
        "explanation": explanation
    }