import os
import cv2
import json
import base64
from dotenv import load_dotenv

# 👇 FIX 1: Import your existing Colab/Hosted LLM handler
from src.utils.llm_client import execute_with_colab_fallback

load_dotenv()

class GameplayVerifier:
    def __init__(self):
        self.inputs_dir = "data/inputs"
        
        # 👇 FIX 2: Route through your existing environment variables
        self.api_url = os.getenv("QWEN_PACING_URL")
        self.api_key = os.getenv("QWEN_API_KEY", "ollama")

    def _extract_frames_b64(self, vod_path: str, start_t: float, end_t: float, peak_t: float) -> list:
        cap = cv2.VideoCapture(vod_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        timestamps = [
            max(start_t, peak_t - 8.0),
            max(start_t, peak_t - 4.0),
            peak_t,
            min(end_t, peak_t + 4.0),
            min(end_t, peak_t + 8.0),
        ]
        
        b64_frames = []

        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ret, frame = cap.read()
            if ret:
                # Use the same aggressive compression from your reference code
                h, w, _ = frame.shape
                top_edge = int(h * 0.12) 
                bottom_edge = int(h * 0.57) 
                roi_cropped_frame = frame[top_edge:bottom_edge, 0:w]
                
                final_h = int(roi_cropped_frame.shape[0] * (426 / w))
                resized_frame = cv2.resize(roi_cropped_frame, (426, final_h))
                
                _, buffer = cv2.imencode('.jpg', resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                b64_frames.append(base64.b64encode(buffer).decode('utf-8'))
                
        cap.release()
        return b64_frames

    def _call_vision_api(self, b64_frames: list) -> dict:
        prompt = """Analyze these 5 sequential frames from a Pokemon Unite video game stream.
        Classify the scene strictly as ONE of the following:
        - high_action_gameplay 
        - boring_gameplay 
        - lobby_or_menu 
        - loading_or_matchmaking 
        - results_screen 
        - dead_air 

        CRITICAL RULE: Do NOT approve a clip just because there is motion. 
        Approve only if there is visible active gameplay with meaningful action: enemy fight, team fight, KO, scoring attempt, escape, objective fight, or intense combat.
        Reject menus, lobby, walking alone, farming alone, camera movement, overlays, loading, or streamer-only movement.

        Respond ONLY in valid JSON format:
        {
            "classification": "chosen_category",
            "reason": "short explanation of what is happening on screen"
        }"""

        # 👇 FIX 3: Format payload exactly like your extract_dense_cinematic_frames logic
        content = [{"type": "text", "text": prompt}]
        for b64 in b64_frames:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            
        messages = [{"role": "user", "content": content}]

        # 👇 FIX 4: Call your custom client wrapper
        raw_response = execute_with_colab_fallback(self.api_key, self.api_url, messages)
        
        # --- BUG-FREE JSON SANITIZATION BLOCK (Copied from your reference) ---
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
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"      ❌ [Error] Vision API returned invalid JSON: {raw_response}")
            return {"classification": "error", "reason": "Invalid JSON response"}

    def verify_candidates(self, raw_json_path: str):
        print(f"\n👁️ [Gameplay Verifier] Booting Vision Model via Colab Fallback...")
        
        with open(raw_json_path, 'r') as f:
            candidates = json.load(f)

        verified_data = []
        approved_count = 0

        for idx, clip in enumerate(candidates, 1):
            print(f"   ► Verifying {clip['window_id']} ({idx}/{len(candidates)})...")
            
            peak_t = clip.get("peak_time", (clip["start_time"] + clip["end_time"]) / 2.0)
            frames = self._extract_frames_b64(clip["source_vod"], clip["start_time"], clip["end_time"], peak_t)
            
            if len(frames) < 3:
                print("      ⚠️ Failed to extract frames. Skipping.")
                continue

            result = self._call_vision_api(frames)
            
            classification = result.get("classification", "unknown")
            approved = (classification == "high_action_gameplay")
            
            clip["classification"] = classification
            clip["approved"] = approved
            clip["reason"] = result.get("reason", "No reason provided")
            clip["status"] = "verified"
            
            if clip["approved"]: approved_count += 1
            print(f"      Result: {clip['classification']} -> Approved: {clip['approved']} ({clip['reason']})")
            
            verified_data.append(clip)

        vod_name = os.path.splitext(os.path.basename(candidates[0]["source_vod"]))[0]
        verified_json = os.path.join(self.inputs_dir, f"verified_candidate_windows_{vod_name}.json")
        
        with open(verified_json, 'w') as f:
            json.dump(verified_data, f, indent=2)

        print(f"✅ Verification Complete! {approved_count}/{len(candidates)} clips approved.")
        print(f"📁 Saved to {verified_json}")