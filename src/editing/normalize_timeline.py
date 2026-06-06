import os
import json
from typing import List, Dict, Any
import copy

def fill_timeline_gaps(editing_plan: list, clip_start: float, clip_end: float) -> list:
    """
    Ensures the normalized EDF covers the full selected clip window.
    Missing unlabeled gaps/tails are filled with safe normal-speed gameplay.
    Handles overlapping AI hallucinations safely.
    """
    def create_default_segment(start, end, reason):
        return {
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "playback_action": {
                "type": "speed_control",
                "speed": 1.0
            },
            "_audit": {
                "decision": reason,
                "reason": "No Qwen label covered this interval"
            }
        }

    if not editing_plan:
        return [create_default_segment(clip_start, clip_end, "full_clip_default_fill")]

    sorted_plan = sorted(editing_plan, key=lambda x: float(x["start"]))
    filled_plan = []
    current_time = float(clip_start)

    for original_seg in sorted_plan:
        seg = copy.deepcopy(original_seg)

        seg_start = max(float(seg["start"]), float(clip_start))
        seg_end = min(float(seg["end"]), float(clip_end))

        if seg_end <= seg_start:
            continue

        # Fill gap before segment
        if seg_start > current_time:
            filled_plan.append(create_default_segment(current_time, seg_start, "unclassified_gap_fill"))

        # If AI hallucinated an overlap, trim the start forward
        if seg_start < current_time:
            seg_start = current_time

        if seg_end <= seg_start:
            continue

        seg["start"] = round(seg_start, 3)
        seg["end"] = round(seg_end, 3)

        if "playback_action" not in seg:
            seg["playback_action"] = {"type": "speed_control", "speed": 1.0}

        filled_plan.append(seg)
        current_time = max(current_time, seg_end)

    # Fill final tail
    if current_time < float(clip_end):
        filled_plan.append(create_default_segment(current_time, clip_end, "unclassified_tail_fill"))

    return filled_plan
# In a full run, this would be loaded via a yaml.safe_load("config.yaml")
CONFIG = {
    "editor": {
        "min_confidence": 0.70,
        "max_speed": 10.0,
        "min_speed": 0.25
    },
    "event_priority": {
        "objective_steal": 100,
        "teamfight": 90,
        "clutch_escape": 80,
        "high_impact_ko": 70,
        "dead_air": 10,
        "default": 1
    }
}

def clamp_speed(speed: float) -> float:
    """Ensure FFmpeg doesn't receive fatal speed math."""
    return max(CONFIG["editor"]["min_speed"], min(CONFIG["editor"]["max_speed"], speed))

def normalize_timeline(qwen_analysis: Dict[str, Any], clip_start: float, clip_end: float, run_dir: str) -> List[Dict[str, Any]]:
    """
    Translates overlapping AI semantic segments into a deterministic, 
    FFmpeg-safe contiguous timeline.
    """
    min_conf = CONFIG["editor"]["min_confidence"]
    
    # 1. Filter out hallucinated or low-confidence segments
    valid_dead_air = [
        da for da in qwen_analysis.get("dead_air_segments", []) 
        if da.get("confidence", 1.0) >= min_conf
    ]
    valid_highlights = [
        hl for hl in qwen_analysis.get("highlight_segments", []) 
        if hl.get("confidence", 1.0) >= min_conf
    ]

    # 2. Collect all unique timeline boundaries
    boundaries = {clip_start, clip_end}
    for seg in valid_dead_air + valid_highlights:
        boundaries.update([seg["start_timestamp"], seg["end_timestamp"]])
        
    sorted_boundaries = sorted(list(boundaries))
    
    # 3. Generate Atomic Segments
    normalized_plan = []
    
    for i in range(len(sorted_boundaries) - 1):
        seg_start = sorted_boundaries[i]
        seg_end = sorted_boundaries[i+1]
        midpoint = (seg_start + seg_end) / 2.0
        
        # Determine highest priority state for this midpoint
        winning_event = None
        highest_priority = CONFIG["event_priority"]["default"]
        speed = 1.0
        decision_reason = "baseline_gameplay"
        conf_score = 1.0

        # Check Highlights
        for hl in valid_highlights:
            if hl["start_timestamp"] <= midpoint <= hl["end_timestamp"]:
                event_type = hl.get("event_type", "high_impact_ko")
                priority = CONFIG["event_priority"].get(event_type, 50)
                if priority > highest_priority:
                    highest_priority = priority
                    speed = 1.0 # Highlights play at normal speed
                    winning_event = hl
                    decision_reason = event_type
                    conf_score = hl.get("confidence", 1.0)

        # Check Dead Air
        for da in valid_dead_air:
            if da["start_timestamp"] <= midpoint <= da["end_timestamp"]:
                priority = CONFIG["event_priority"]["dead_air"]
                if priority > highest_priority:
                    highest_priority = priority
                    speed = 4.0 # Compress dead air
                    winning_event = da
                    decision_reason = "dead_air_compression"
                    conf_score = da.get("confidence", 1.0)

        safe_speed = clamp_speed(speed)

        # 4. Build the atomic segment with the audit trail
        segment_data = {
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "playback_action": {
                "type": "speed_control",
                "speed": safe_speed
            },
            "_audit": {
                "decision": decision_reason,
                "priority_score": highest_priority,
                "confidence": conf_score
            }
        }
        normalized_plan.append(segment_data)

    # 5. Save the Compiler Audit Log
    normalized_plan = fill_timeline_gaps(normalized_plan, clip_start, clip_end)

    # 5. Save the Compiler Audit Log
    audit_output = {
        "edf_version": "2.0",
        "clip_bounds": {"start": clip_start, "end": clip_end},
        "normalized_timeline": normalized_plan
    }
    
    audit_path = os.path.join(run_dir, "normalized_timeline.json")
    with open(audit_path, 'w') as f:
        json.dump(audit_output, f, indent=4)
        
    print(f"✅ [EDF Compiler] Normalized {len(normalized_plan)} atomic segments. Audit saved.")
    
    return normalized_plan