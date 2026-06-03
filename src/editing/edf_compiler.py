import os
import json
from src.editing.normalize_timeline import normalize_timeline

# ==========================================
# GLOBAL CONSTANTS
# ==========================================
MICRO_DRIFT_TOLERANCE = 0.05  # 50ms tolerance for floating point safety

# ==========================================
# 1. THE HARDENED VALIDATOR (Safe Auto-Healer V1.0)
# ==========================================
def validate_edf(editing_plan: list, clip_start: float, clip_end: float):
    if not editing_plan:
        raise ValueError("EDF Validation Failed: Editing plan is empty.")

    # ==========================================
    # 🩹 THE SAFE AUTO-HEALER (Floating point & micro-drift only)
    # ==========================================
    # 1. Fix Outer Boundaries (Sub-50ms drift only)
    start_delta = editing_plan[0]["start"] - clip_start
    if 0 < abs(start_delta) <= MICRO_DRIFT_TOLERANCE:
        direction = "gap" if start_delta > 0 else "overlap"
        print(f"🩹 [Auto-Heal] Snapping first segment start (micro-{direction} {abs(start_delta):.3f}s) to {clip_start}")
        editing_plan[0]["start"] = float(clip_start)
        
    end_delta = editing_plan[-1]["end"] - clip_end
    if 0 < abs(end_delta) <= MICRO_DRIFT_TOLERANCE:
        direction = "overshoot" if end_delta > 0 else "gap"
        print(f"🩹 [Auto-Heal] Snapping last segment end (micro-{direction} {abs(end_delta):.3f}s) to {clip_end}")
        editing_plan[-1]["end"] = float(clip_end)
        
    # 2. Fix Internal Discontinuities (Micro-gaps and overlaps only)
    for i in range(1, len(editing_plan)):
        prev_end = editing_plan[i-1]["end"]
        delta = editing_plan[i]["start"] - prev_end
        
        if 0 < abs(delta) <= MICRO_DRIFT_TOLERANCE:
            direction = "gap" if delta > 0 else "overlap"
            print(f"🩹 [Auto-Heal] Snapping internal micro-{direction} of {abs(delta):.3f}s at index {i}")
            editing_plan[i]["start"] = float(prev_end)
    # ==========================================

    # Outer boundary checks (Will catch anything > MICRO_DRIFT_TOLERANCE)
    if abs(editing_plan[0]["start"] - clip_start) > MICRO_DRIFT_TOLERANCE:
        raise ValueError(f"EDF Validation Failed: Timeline must start at {clip_start}. Detected large drift: {abs(editing_plan[0]['start'] - clip_start):.2f}s")
    if abs(editing_plan[-1]["end"] - clip_end) > MICRO_DRIFT_TOLERANCE:
        raise ValueError(f"EDF Validation Failed: Timeline must end at {clip_end}. Detected large drift: {abs(editing_plan[-1]['end'] - clip_end):.2f}s")

    for i in range(len(editing_plan)):
        segment = editing_plan[i]
        
        # Time moves forward
        if segment["start"] >= segment["end"]:
            raise ValueError(f"EDF Validation Failed: Segment {i} has invalid timeline (start >= end).")
            
        # ⚠️ Quality Warning (Product Rule, purely for visibility in logs)
        duration = segment["end"] - segment["start"]
        if duration > 60.0:
            print(f"⚠️ [Quality Warning] Segment {i} is unusually long ({duration:.2f}s) for short-form pacing.")
            
        # Speed is valid
        speed = segment.get("playback_action", {}).get("speed", 1.0)
        if speed <= 0:
            raise ValueError(f"EDF Validation Failed: Segment {i} has impossible speed ({speed}).")

        if i > 0:
            prev_segment = editing_plan[i-1]
            
            # Sort order check (Maintained for rapid developer UX/debugging)
            if segment["start"] < prev_segment["start"]:
                raise ValueError(f"EDF Validation Failed: Segments are out of chronological order at index {i}.")
                
            # Contiguous check (Will catch any gap/overlap > MICRO_DRIFT_TOLERANCE)
            delta = segment["start"] - prev_segment["end"]
            if abs(delta) > MICRO_DRIFT_TOLERANCE:
                raise ValueError(f"EDF Validation Failed: Large discontinuity detected ({abs(delta):.2f}s) between {prev_segment['end']}s and {segment['start']}s.")
                
    print("✅ [Validator] EDF Timeline is mathematically contiguous, sorted, and safe for FFmpeg.")

# ==========================================
# 2. RETENTION ANALYTICS (The Scoreboard)
# ==========================================
def generate_retention_report(qwen_analysis: dict, editing_plan: list, run_dir: str):
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
    
    # Dynamic Pipeline Health (Removed validator_warnings dependency)
    health = "STABLE"
    if engagement < 0.4 or rendered_duration == 0:
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
    try:
        validate_edf(normalized_plan, clip_start, clip_end)
    except ValueError as e:
        print(f"❌ [CRITICAL] EDF Validation Failed: {e}")
        raise e
        
    # Step 3: Generate the analytics scoreboard
    generate_retention_report(qwen_analysis, normalized_plan, run_dir)
    
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