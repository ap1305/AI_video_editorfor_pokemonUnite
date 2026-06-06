import os
import cv2
import base64
import json
from typing import Dict, Any
from src.utils.llm_client import execute_with_colab_fallback

def extract_dense_cinematic_frames(video_path: str, start_time: float, end_time: float, fps_target: int = 1) -> list:
    clip_duration = end_time - start_time
    num_frames = int(clip_duration * fps_target)
    
    print(f"📸 [Micro Director] Snapping frames (1 FPS) between {start_time}s and {end_time}s...")
    
    if not os.path.exists(video_path):
        print(f"❌ [Error] Video file not found: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * original_fps)
    end_frame = int(end_time * original_fps)
    
    frame_interval = int(original_fps / fps_target)
    
    base64_frames = []
    current_frame = start_frame
    
    debug_dir = "data/temp_frames"
    os.makedirs(debug_dir, exist_ok=True)
    frame_count = 0
    
    while current_frame < end_frame: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, frame = cap.read()
        if success:
            h, w, _ = frame.shape
            
            top_edge = int(h * 0.12) 
            bottom_edge = int(h * 0.57) 
            roi_cropped_frame = frame[top_edge:bottom_edge, 0:w]
            
            # 👇 FIX 1: Aggressive Token Compression (Resize to 426px wide instead of 854px)
            final_h = int(roi_cropped_frame.shape[0] * (426 / w))
            resized_frame = cv2.resize(roi_cropped_frame, (426, final_h))
            
            file_name = f"frame_{str(frame_count).zfill(3)}.jpg"
            file_path = os.path.join(debug_dir, file_name)
            # Drop JPEG quality from 70 to 50 to save payload size
            cv2.imwrite(file_path, resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            
            _, buffer = cv2.imencode('.jpg', resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            
            calculated_timestamp = round(start_time + (frame_count / fps_target), 1)
            base64_frames.append({
                "timestamp": calculated_timestamp,
                "b64": base64.b64encode(buffer).decode('utf-8')
            })
            
            frame_count += 1
            current_frame += frame_interval
        else:
            break
            
    cap.release()
    print(f"✅ [Micro Director] Extracted {len(base64_frames)} sequential frames.")
    return base64_frames


# 👇 UPDATE 1: Added strict_mode parameter to signature
# 👇 Updated signature with strict_mode=False
def analyze_clip_pacing(video_path: str, start_time: float, end_time: float, api_key: str, base_url: str, strict_mode: bool = False) -> Dict[str, Any]:
    base_frames_data = extract_dense_cinematic_frames(video_path, start_time, end_time)
    
    if not base_frames_data:
        print("❌ [Phase 3] Error: No frames extracted. Skipping pacing analysis.")
        return {}
        
    # ==========================================
    # 👇 CORRECTED COMMENT: Strict 30-frame limit to reduce context overflow risk
    # ==========================================
    MAX_SAFE_FRAMES = 30
    if len(base_frames_data) > MAX_SAFE_FRAMES:
        print(f"⚠️ [VRAM Protection] Downsampling {len(base_frames_data)} frames to {MAX_SAFE_FRAMES} to prevent Context Overflow...")
        step = len(base_frames_data) / MAX_SAFE_FRAMES
        base_frames_data = [base_frames_data[int(i * step)] for i in range(MAX_SAFE_FRAMES)]
    # ==========================================
        
    clip_duration = round(end_time - start_time, 1)
    
    system_prompt = f"""
    You are an elite, high-retention Pokémon Unite video editor and coach.
    I am providing {len(base_frames_data)} sequential frames representing a {clip_duration}-second gameplay highlight.
    Each frame is preceded by its exact video timestamp. 

    Your primary job is to identify the single 'Primary Climax' of the clip and ruthlessly classify the rest of the footage to support it. 

    TIER 1 (MUST NEVER MISS - Highest Viral Potential):
    - Objective Steals (Rayquaza/Zapdos/Regi secure with low boss HP and clustered players)
    - Teamfight Climax (Multiple HP bars dropping, Unite move auras, KO popups)
    - Multi-KO Chains (Consecutive KO notifications/kill-feed activity)

    TIER 2 (HIGH VALUE):
    - Clutch Escapes (Player survives on very low HP against multiple enemies)
    - 1vX Survival (Outnumbered player trades favorably or survives)
    - Last-Second Scores (Goal animation during final countdown)

    TIER 3 (CONTEXT BUILDERS VS DEAD AIR):
    - Setup: Sneaking to a pit, flanking, or positioning before a fight. (Keep this in highlight_segments).
    - Dead Air: Aimless walking or baseline farming with no immediate payoff.
    
    CRITICAL EVALUATION RULES:
    1. PRIMARY CLIMAX: Identify the single most important segment in the clip for thumbnail/hook generation.
    2. NARRATIVE ROLE: Role is determined by IMPACT, not event type. A teamfight might be 'Conflict', or it might be the 'Climax'. Assign every segment: [Setup, Conflict, Climax, Resolution, Dead_Air].
    3. IMPORTANCE RUBRIC:
       - 0.90-1.00 = Game-defining moment (viral climax)
       - 0.75-0.89 = Major highlight
       - 0.50-0.74 = Meaningful action / Setup
       - 0.00-0.49 = Low-value content / Dead air
    4. NARRATIVE OUTCOME: Tag segments with "success", "failure", or "neutral". A sacrificial death that secures Rayquaza is a "success".
    5. BE EXTREMELY CONCISE: To save processing time, your 'visual_evidence' and 'reason' fields MUST NOT exceed 5 words each. Do not write full sentences.
    
    You MUST output strictly in this JSON schema with NO markdown:
    {{
      "engagement_score": <float between 0.0 and 1.0>,
      "the_hook_timestamp": <float>,
      "dead_air_segments": [
        {{
          "segment_id": "da_1",
          "start_timestamp": <float>, 
          "end_timestamp": <float>, 
          "narrative_role": "Dead_Air",
          "visual_evidence": ["<strict description of screen, MAX 5 WORDS>"],
          "reason": "<why this is dead air, MAX 5 WORDS>",
          "importance": <float>, 
          "confidence": <float>
        }}
      ],
      "highlight_segments": [
        {{
          "segment_id": "hl_1",
          "start_timestamp": <float>, 
          "end_timestamp": <float>, 
          "event_type": "<string>", 
          "narrative_role": "<Setup | Conflict | Climax | Resolution>",
          "visual_evidence": ["<strict description of screen, MAX 5 WORDS>"],
          "reason": "<narrative impact, MAX 5 WORDS>",
          "outcome": "<success | failure | neutral>", 
          "importance": <float>, 
          "confidence": <float>
        }}
      ],
      "primary_climax_id": "<string matching a highlight segment_id>",
      "primary_climax_confidence": <float>
    }}
    """

    # 👇 CORRECTED STRICT RETRY PROMPT 👇
    if strict_mode:
        system_prompt += """
    ==================================================
    STRICT RETRY MODE TRIGGERED:
    Your previous edit failed quality checks because it ruined pacing, classified important action as dead air, or had a boring hook.

    NEW MANDATORY RULES:
    1. The first 2 seconds must contain a hook: enemy presence, combat, scoring, danger, KO, chase, or immediate payoff.
    2. If the candidate begins with walking, rotation, farming, or dead space, classify that opening as dead_air_segments so the compiler can trim/compress it.
    3. Never classify combat, scoring, KO, objective, enemy engagement, clutch escape, or payoff as dead_air_segments.
    4. All combat, scoring, KO, objective, enemy engagement, and payoff moments must be placed in highlight_segments.
    5. Dead air means only walking, farming, map travel, rotation, or no immediate payoff.
    6. Keep visual_evidence and reason under 5 words each.

    CRITICAL SCHEMA REMINDER:
    You MUST still output the exact same semantic blueprint JSON schema requested above.
    Do not include markdown.
    Do not include conversational explanations.
    Return ONLY valid JSON.
    ==================================================
    """
    
    content = [{"type": "text", "text": system_prompt}]
    
    for frame_data in base_frames_data:
        content.append({"type": "text", "text": f"FRAME_TIMESTAMP: {frame_data['timestamp']}s"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_data['b64']}"}})
        
    messages = [{"role": "user", "content": content}]
    
    print(f"🧠 [Phase 3] Sending chronologically tagged frames to Qwen for Semantic Choreography...")
    raw_response = execute_with_colab_fallback(api_key, base_url, messages)
    
    # --- BUG-FREE JSON SANITIZATION BLOCK ---
    cleaned = raw_response.strip()
    md_ticks = chr(96) * 3 
    md_json = md_ticks + "json"
    
    if cleaned.startswith(md_json): cleaned = cleaned[7:]
    elif cleaned.startswith(md_ticks): cleaned = cleaned[3:]
    if cleaned.endswith(md_ticks): cleaned = cleaned[:-3]
    
    cleaned = cleaned.strip()
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')
    
    if start_idx != -1 and end_idx != -1:
        cleaned = cleaned[start_idx:end_idx+1]
        
    try:
        blueprint = json.loads(cleaned)
        print("✅ [Phase 3] Qwen successfully generated semantic blueprint!")
        return blueprint
    except json.JSONDecodeError:
        print(f"❌ [Error] Qwen Micro Director returned invalid JSON: {raw_response}")
        return {}