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
    
    # 👇 FIX 1: Changed <= to < to prevent the off-by-one extra frame
    while current_frame < end_frame: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, frame = cap.read()
        if success:
            h, w, _ = frame.shape
            
            # The Perfect Crop (Already implemented perfectly by you)
            top_edge = int(h * 0.12) 
            bottom_edge = int(h * 0.57) 
            roi_cropped_frame = frame[top_edge:bottom_edge, 0:w]
            
            final_h = int(roi_cropped_frame.shape[0] * (854 / w))
            resized_frame = cv2.resize(roi_cropped_frame, (854, final_h))
            
            file_name = f"frame_{str(frame_count).zfill(3)}.jpg"
            file_path = os.path.join(debug_dir, file_name)
            cv2.imwrite(file_path, resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            
            _, buffer = cv2.imencode('.jpg', resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            
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

def analyze_clip_pacing(video_path: str, start_time: float, end_time: float, api_key: str, base_url: str) -> Dict[str, Any]:
    base_frames_data = extract_dense_cinematic_frames(video_path, start_time, end_time)
    
    if not base_frames_data:
        print("❌ [Phase 3] Error: No frames extracted. Skipping pacing analysis.")
        return {}
        
    # ==========================================
    # 👇 FIX 2: VRAM CRASH PREVENTION (DYNAMIC DOWNSAMPLE)
    # ==========================================
    MAX_SAFE_FRAMES = 8
    if len(base_frames_data) > MAX_SAFE_FRAMES:
        print(f"⚠️ [VRAM Protection] Downsampling {len(base_frames_data)} frames to {MAX_SAFE_FRAMES} to prevent Colab GPU crash...")
        step = len(base_frames_data) / MAX_SAFE_FRAMES
        # Grabs evenly spaced frames across the timeline
        base_frames_data = [base_frames_data[int(i * step)] for i in range(MAX_SAFE_FRAMES)]
    # ==========================================
        
    clip_duration = round(end_time - start_time, 1)
    
    system_prompt = f"""
    You are the 'Micro Director' for an autonomous esports video editor.
    I am providing {len(base_frames_data)} sequential frames representing a {clip_duration}-second MOBA highlight.
    Each frame is preceded by its exact video timestamp. 
    
    Your job is to analyze the semantic narrative flow (HP drops, ultimates, escapes, KOs) and output a semantic choreography blueprint.
    
    CRITICAL RULES:
    1. DO NOT invent editing effects (e.g., no "screen_shake" or "zoom"). You describe the EVENT, the rendering engine decides the effect.
    2. Classify the overarching pacing into explicit Narrative Beats: [SETUP, PRESSURE, CHAOS, CLIMAX, RELEASE, AFTERMATH].
    3. Provide a confidence score (0.0 to 1.0) for every major event.

    You MUST output strictly in this JSON schema with NO markdown:
    {{
      "narrative_beats": [
        {{"start_timestamp": {start_time}, "end_timestamp": {start_time + 4.0}, "beat": "SETUP"}},
        {{"start_timestamp": {start_time + 4.0}, "end_timestamp": {start_time + 10.0}, "beat": "CHAOS"}}
      ],
      "semantic_events": [
        {{"timestamp": {start_time + 4.0}, "event_type": "high_impact_ko", "description": "Enemy team engages, massive splash damage.", "confidence": 0.95}},
        {{"timestamp": {start_time + 8.0}, "event_type": "clutch_escape", "description": "Player survives on low HP and dashes away.", "confidence": 0.88}}
      ],
      "pacing_anomalies": [
        {{"timestamp": {start_time + 5.0}, "anomaly_type": "sudden_visual_flash", "confidence": 0.90}}
      ]
    }}
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