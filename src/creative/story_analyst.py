import os
import json
import base64
import cv2
from openai import OpenAI

class StoryAnalyst:
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """
        Initializes the Story Analyst using an OpenAI-compatible client for Qwen-VL.
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        # Using the recommended multimodal model
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

    def analyze_clip(self, clip_id: str, paced_video_path: str, edf_metadata: dict = None) -> dict:
        """
        Passes the extracted frames to Qwen to generate the creative_story.json contract.
        """
        print(f"🎬 [Story Analyst] Extracting timeline frames for {clip_id}...")
        frames_data = self._extract_frames(paced_video_path, fps_target=1)
        
        if not frames_data:
            raise ValueError(f"Failed to extract frames from {paced_video_path}")

        print(f"🧠 [Story Analyst] Sending {len(frames_data)} frames to Qwen Vision...")

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
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2, # Low temperature for analytical consistency
                max_tokens=800
            )
            
            raw_output = response.choices[0].message.content.strip()
            
            # Clean potential markdown if the model disobeys the prompt
            if raw_output.startswith("```json"):
                raw_output = raw_output.replace("```json", "").replace("```", "").strip()
            elif raw_output.startswith("```"):
                raw_output = raw_output.replace("```", "").strip()

            story_contract = json.loads(raw_output)
            
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

# --- Quick Test Block (Comment out in production) ---
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize the analyst
    analyst = StoryAnalyst(api_key=os.getenv("QWEN_API_KEY"))
    
    # Test path (Replace with an actual paced clip from your data/renders folder)
    test_clip_path = "data/renders/viral_short_clip_1.mp4"
    
    if os.path.exists(test_clip_path):
        story = analyst.analyze_clip("clip_001", test_clip_path)
        print(json.dumps(story, indent=2))
    else:
        print(f"Test clip not found at {test_clip_path}. Ready to integrate.")