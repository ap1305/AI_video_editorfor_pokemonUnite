import os
import json
import base64
import cv2
import requests

class StoryAnalyst:
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """
        Initializes the Story Analyst using a direct HTTP client for Qwen-VL.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = "qwen-vl-max"

    def _extract_frames(self, video_path: str, fps_target: int = 1) -> list:
        """
        Extracts frames from the paced clip at the target FPS.
        Returns a list of base64 encoded JPEG strings with their timestamps.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Paced clip not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / fps_target) if original_fps > 0 else 30
        
        frames_base64 = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Calculate exact timestamp
                timestamp = round(frame_count / original_fps, 2)
                
                # Resize to save token context/VRAM (720p width max)
                height, width = frame.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    frame = cv2.resize(frame, (1280, int(height * scale)))
                
                # Encode to JPEG -> Base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                b64_str = base64.b64encode(buffer).decode('utf-8')
                
                frames_base64.append({
                    "timestamp": timestamp,
                    "b64": b64_str
                })
                
            frame_count += 1
            
        cap.release()
        return frames_base64

    def _execute_with_colab_fallback(self, messages: list) -> str:
        """
        Direct HTTP request bypassing the OpenAI wrapper for robust compatibility.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 800
        }
        
        endpoint = f"{self.base_url}/chat/completions"
        response = requests.post(endpoint, json=payload, headers=headers, timeout=120)
        
        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}: {response.text}")
            
        return response.json()['choices'][0]['message']['content']
        
    def _clamp_story_timing(self, story_contract: dict, frames_data: list) -> dict:
        """
        Repairs hallucinated LLM timestamps so Director/Validator always receive
        physically valid PACED_CLIP timing.
        """
        def sf(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        # Prefer LLM clip_duration if valid, otherwise use last extracted frame timestamp
        fallback_dur = frames_data[-1]["timestamp"] if frames_data else 0.0
        clip_dur = sf(story_contract.get("clip_duration"), fallback_dur)

        if clip_dur <= 0:
            clip_dur = max(fallback_dur, 1.0)

        story_contract["clip_duration"] = round(clip_dur, 2)

        # Clamp payoff timestamp
        payoff = sf(story_contract.get("payoff_timestamp"), clip_dur * 0.75)
        payoff = max(0.0, min(payoff, clip_dur))
        story_contract["payoff_timestamp"] = round(payoff, 2)

        # Clamp reaction window
        rw = story_contract.get("reaction_window", {})
        if not isinstance(rw, dict):
            rw = {}

        raw_start = sf(rw.get("start"), payoff)
        raw_end = sf(rw.get("end"), min(clip_dur, payoff + 2.0))

        start_t = max(0.0, min(raw_start, clip_dur))
        end_t = max(0.0, min(raw_end, clip_dur))

        # If Qwen placed reaction after video end, move it before the clip ends
        if start_t >= clip_dur or end_t <= start_t:
            end_t = clip_dur
            start_t = max(0.0, clip_dur - 1.5)

        # Ensure minimum useful reaction window
        if end_t - start_t < 0.5 and clip_dur >= 1.5:
            end_t = min(clip_dur, max(end_t, start_t + 1.5))
            if end_t > clip_dur:
                end_t = clip_dur
                start_t = max(0.0, end_t - 1.5)

        story_contract["reaction_window"] = {
            "start": round(start_t, 2),
            "end": round(end_t, 2)
        }

        # Clamp protected windows
        clean_protected = []
        protected_windows = story_contract.get("protected_windows", [])

        if isinstance(protected_windows, list):
            for pw in protected_windows:
                if not isinstance(pw, dict):
                    continue

                pw_start = max(0.0, min(sf(pw.get("start")), clip_dur))
                pw_end = max(0.0, min(sf(pw.get("end")), clip_dur))

                if pw_start < pw_end:
                    clean_protected.append({
                        "start": round(pw_start, 2),
                        "end": round(pw_end, 2),
                        "reason": str(pw.get("reason", "protected gameplay"))
                    })

        story_contract["protected_windows"] = clean_protected

        return story_contract

    def analyze_clip(self, clip_id: str, paced_video_path: str, edf_metadata: dict = None) -> dict:
        """
        Passes the extracted frames to Qwen to generate the creative_story.json contract.
        """
        print(f"🎬 [Story Analyst] Extracting timeline frames for {clip_id}...")
        frames_data = self._extract_frames(paced_video_path, fps_target=1)
        
        if not frames_data:
            raise ValueError(f"Failed to extract frames from {paced_video_path}")

        print(f"🧠 [Story Analyst] Sending {len(frames_data)} frames to Vision model...")

        system_prompt = """
        You are the 'Story Analyst' for a Pokémon Unite editing pipeline.
        Your job is to watch a sequence of timestamped frames from a fast-paced gameplay clip and explain the narrative.
        
        CRITICAL RULES:
        1. Ground everything in VISIBLE REALITY. Do not invent scores, KOs, or objectives if they do not happen in the frames.
        2. Identify the 'payoff_timestamp' (the exact second the climax/score/KO resolves).
        3. Identify the 'reaction_window' (the 1 to 2 seconds immediately following the payoff where a reaction meme would naturally fit).
        4. Identify 'protected_windows' (timestamps where critical combat or scoring happens and MUST NOT be covered by a meme).
        5. Return ONLY valid JSON matching the exact schema provided. Do not use Markdown wrappers like ```json.
        
        OUTPUT SCHEMA:
        {
          "schema_version": "1.0",
          "clip_id": "<inject_clip_id>",
          "timeline_basis": "PACED_CLIP",
          "clip_duration": <float>,
          "scene_description": "A 50-100 word explanation grounded only in visible events.",
          "viewer_expectation": "Short sentence on what it looks like is about to happen.",
          "actual_outcome": "Short sentence on what actually happens.",
          "comedy_mechanism": "e.g., enemy_overconfidence_backfires, narrow_escape, whiffed_ultimate",
          "payoff_timestamp": <float>,
          "reaction_window": { "start": <float>, "end": <float> },
          "protected_windows": [ { "start": <float>, "end": <float>, "reason": "string" } ],
          "confidence": <float between 0.0 and 1.0>
        }
        """

        # Construct the multimodal message payload
        content_payload = []
        
        # Add the text prompt
        context_text = f"Analyze this clip. The clip_id is {clip_id}. "
        if edf_metadata:
            context_text += f"Additional EDF Context: {json.dumps(edf_metadata)}."
            
        content_payload.append({"type": "text", "text": context_text})
        
        # Add the frames with their timestamps
        for frame in frames_data:
            content_payload.append({"type": "text", "text": f"Timestamp: {frame['timestamp']}s"})
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame['b64']}"}
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_payload}
        ]

        try:
            print(f"🧠 [Story Analyst] Calling LLM '{self.model_name}' via Colab fallback...")
            
            # Direct HTTP Request Execution
            raw_response = self._execute_with_colab_fallback(messages)
            
            # Strict Markdown Sanitization
            cleaned = raw_response.strip()
            md_ticks = chr(96) * 3 
            md_json = md_ticks + "json"
            
            if cleaned.startswith(md_json):
                cleaned = cleaned[len(md_json):]
            elif cleaned.startswith(md_ticks):
                cleaned = cleaned[len(md_ticks):]
            if cleaned.endswith(md_ticks):
                cleaned = cleaned[:-len(md_ticks)]
                
            raw_output = cleaned.strip()

            story_contract = json.loads(raw_output)

            # Ensure clip duration is present in the output contract
            if frames_data and "clip_duration" not in story_contract:
                story_contract["clip_duration"] = frames_data[-1]["timestamp"]

            # Repair hallucinated timestamps before Director sees the story
            story_contract = self._clamp_story_timing(story_contract, frames_data)

            # Save the contract to disk
            output_dir = "data/creative/story"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{clip_id}_creative_story.json")
            
            with open(output_path, "w") as f:
                json.dump(story_contract, f, indent=4)
                
            print(f"✅ [Story Analyst] Successfully generated {output_path}")
            return story_contract

        except Exception as e:
            print(f"❌ [Story Analyst Error]: {str(e)}")
            return None

# --- Quick Test Block ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize the analyst
    analyst = StoryAnalyst(api_key=os.getenv("QWEN_API_KEY"))
    
    test_clip_path = "data/renders/viral_short_clip_1.mp4"
    
    if os.path.exists(test_clip_path):
        story = analyst.analyze_clip("clip_001", test_clip_path)
        if story:
            print(json.dumps(story, indent=2))
    else:
        print(f"Test clip not found at {test_clip_path}. Ready to integrate.")