import os
import json
from src.editing.normalize_timeline import normalize_timeline

# ==========================================
# 1. THE HARDENED VALIDATOR (The Bouncer)
# ==========================================
def validate_edf(editing_plan: list, clip_start: float, clip_end: float):
    EPSILON = 0.05  # 50ms tolerance for floating point safety
    
    if not editing_plan:
        raise ValueError("EDF Validation Failed: Editing plan is empty.")

    # Outer boundary checks with tolerance
    if abs(editing_plan[0]["start"] - clip_start) > EPSILON:
        raise ValueError(f"EDF Validation Failed: Timeline must start at {clip_start}")
    if abs(editing_plan[-1]["end"] - clip_end) > EPSILON:
        raise ValueError(f"EDF Validation Failed: Timeline must end at {clip_end}")

    for i in range(len(editing_plan)):
        segment = editing_plan[i]
        
        # Time moves forward
        if segment["start"] >= segment["end"]:
            raise ValueError(f"EDF Validation Failed: Segment {i} has invalid timeline (start >= end).")
            
        # Speed is valid
        speed = segment.get("playback_action", {}).get("speed", 1.0)
        if speed <= 0:
            raise ValueError(f"EDF Validation Failed: Segment {i} has impossible speed ({speed}).")

        if i > 0:
            prev_segment = editing_plan[i-1]
            
            # Sort order check
            if segment["start"] < prev_segment["start"]:
                raise ValueError(f"EDF Validation Failed: Segments are out of chronological order at index {i}.")
                
            # Contiguous check (No gaps, no overlaps)
            if abs(segment["start"] - prev_segment["end"]) > EPSILON:
                raise ValueError(f"EDF Validation Failed: Discontinuity > 50ms between {prev_segment['end']}s and {segment['start']}s.")
                
    print("✅ [Validator] EDF Timeline is mathematically contiguous, sorted, and safe for FFmpeg.")

# ==========================================
# 2. RETENTION ANALYTICS (The Scoreboard)
# ==========================================
def generate_retention_report(qwen_analysis: dict, editing_plan: list, run_dir: str, validator_warnings: int = 0):
    raw_duration = 0.0
    rendered_duration = 0.0
    ramps_applied = 0
    
    for seg in editing_plan:
        seg_raw = seg["end"] - seg["start"]
        speed = seg.get("playback_action", {}).get("speed", 1.0)
        
        raw_duration += seg_raw
        rendered_duration += (seg_raw / speed) # ACTUAL viewing time
        
        if speed > 1.0:
            ramps_applied += 1

    compressed_seconds = raw_duration - rendered_duration
    engagement = qwen_analysis.get("engagement_score", 0.0)
    
    # Dynamic Pipeline Health
    health = "STABLE"
    if engagement < 0.4 or validator_warnings > 0 or rendered_duration == 0:
        health = "WARNING"

    report = {
        "edf_version": "2.0",
        "pipeline_health": health,
        "metrics": {
            "total_raw_duration": round(raw_duration, 2),
            "final_rendered_duration": round(rendered_duration, 2),
            "dead_air_compressed_seconds": round(compressed_seconds, 2),
            "speed_ramps_applied": ramps_applied,
            "qwen_engagement_score": engagement,
            "hook_timestamp_detected": qwen_analysis.get("the_hook_timestamp", None)
        }
    }
    
    report_path = os.path.join(run_dir, "retention_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"📈 [Telemetry] Retention report saved. Compressed {round(compressed_seconds, 1)}s of dead air.")

# ==========================================
# 3. THE MASTER ORCHESTRATOR
# ==========================================
def generate_master_edf(qwen_analysis: dict, clip_start: float, clip_end: float, run_dir: str) -> dict:
    """
    The main entry point for EDF V2. 
    Flow: Normalize -> Validate -> Telemetry -> Dispatch
    """
    print("\n[EDF Compiler V2] Booting pipeline...")
    
    # Step 1: Normalize the raw AI semantics into FFmpeg math
    normalized_plan = normalize_timeline(qwen_analysis, clip_start, clip_end, run_dir)
    
    # Step 2: Validate the math (The Bouncer)
    warnings = 0
    try:
        validate_edf(normalized_plan, clip_start, clip_end)
    except ValueError as e:
        print(f"❌ [CRITICAL] EDF Validation Failed: {e}")
        raise e
        
    # Step 3: Generate the analytics scoreboard
    generate_retention_report(qwen_analysis, normalized_plan, run_dir, validator_warnings=warnings)
    
    # Step 4: Package the final blueprint for FFmpeg
    master_blueprint = {
        "edf_version": "2.0",
        "clip_bounds": {"start": clip_start, "end": clip_end},
        "editing_plan": normalized_plan,
        "meme_candidates": qwen_analysis.get("meme_candidates", []) 
    }
    
    # Save the final master blueprint
    master_path = os.path.join(run_dir, "master_edf.json")
    with open(master_path, 'w') as f:
        json.dump(master_blueprint, f, indent=4)
        
    print("✅ [EDF Compiler V2] Master Blueprint locked and ready for rendering.")
    return master_blueprint