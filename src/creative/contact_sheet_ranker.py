import os
import json
import base64
import tempfile
import requests
import subprocess
import re
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Tuple

class ContactSheetRanker:
    def __init__(self, api_key: str, base_url: str, model_name: str, config: Dict[str, Any] = None):
        """Provider-neutral injection with direct HTTP execution."""
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.config = config or {}
        
        self.sample_positions = tuple(self.config.get("sample_positions", (0.15, 0.50, 0.85)))
        self.max_preview_bytes = int(self.config.get("max_preview_bytes", 10 * 1024 * 1024))
        self.minimum_selection_confidence = float(self.config.get("minimum_selection_confidence", 0.60))
        self.max_candidates = int(self.config.get("max_contact_sheet_candidates", 5))
        
        self.session = requests.Session()

    def _execute_with_colab_fallback(self, messages: list) -> str:
        """Direct HTTP request bypassing OpenAI wrapper for robust compatibility."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }
        
        endpoint = f"{self.base_url}/chat/completions"
        response = self.session.post(endpoint, json=payload, headers=headers, timeout=120)
        
        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}: {response.text}")
            
        return response.json()['choices'][0]['message']['content']

    def _save_decision(self, clip_id: str, decision: dict) -> dict:
        safe_clip_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(clip_id))
        os.makedirs("data/creative/decisions", exist_ok=True)
        path = f"data/creative/decisions/{safe_clip_id}_ranker_decision.json"
        with open(path, "w", encoding="utf-8") as file:
            json.dump(decision, file, indent=2)
        return decision

    def _draw_large_label(self, label: str, width: int, height: int) -> Image.Image:
        small_size = 24
        small_canvas = Image.new("RGBA", (small_size, small_size), (20, 20, 20, 255))
        draw = ImageDraw.Draw(small_canvas)
        font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(((small_size - text_w) / 2, (small_size - text_h) / 2), label, fill="white", font=font)
        return small_canvas.resize((width, height), Image.Resampling.NEAREST)

    def _resize_and_letterbox(self, img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        img_ratio = img.width / max(img.height, 1)
        target_ratio = target_w / target_h
        
        if img_ratio > target_ratio:
            new_w = target_w
            new_h = int(target_w / img_ratio)
        else:
            new_h = target_h
            new_w = int(target_h * img_ratio)
            
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        bg = Image.new("RGBA", (target_w, target_h), (40, 40, 40, 255))
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2
        bg.paste(resized, (offset_x, offset_y), resized if resized.mode == 'RGBA' else None)
        return bg

    def _safe_download(self, url: str) -> bytes:
        resp = self.session.get(url, stream=True, timeout=10)
        resp.raise_for_status()
        content_type = resp.headers.get('Content-Type', '')
        if 'image' not in content_type and 'video' not in content_type:
            raise ValueError(f"Invalid content type: {content_type}")
            
        downloaded_bytes = b""
        for chunk in resp.iter_content(chunk_size=8192):
            downloaded_bytes += chunk
            if len(downloaded_bytes) > self.max_preview_bytes:
                raise ValueError("Payload exceeded maximum safe byte bounds.")
        return downloaded_bytes

    def _get_ffprobe_duration(self, filepath: str) -> float:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filepath]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            return float(res.stdout.strip())
        except Exception:
            return 0.0

    def _extract_frames_unified(self, asset_bytes: bytes, target_w: int, target_h: int) -> list:
        tmp_path = ""
        frames = []
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(asset_bytes)
                tmp_path = tmp.name

            duration = self._get_ffprobe_duration(tmp_path)
            if duration <= 0: duration = 2.0 

            for pos in self.sample_positions:
                t = duration * pos
                cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", tmp_path, "-vframes", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                
                if res.returncode == 0 and res.stdout:
                    img = Image.open(BytesIO(res.stdout)).convert("RGBA")
                    frames.append(self._resize_and_letterbox(img, target_w, target_h))
                else:
                    break 
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            
        if len(frames) != len(self.sample_positions): return []
        return frames

    def build_contact_sheet(self, clip_id: str, candidates: list) -> Tuple[str, dict, list]:
        mapping, row_images, metadata_list = {}, [], []
        cell_w, cell_h, label_width = 200, 150, 80
        safe_clip_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(clip_id))
        
        successful_candidates = 0
        for candidate in candidates:
            if successful_candidates >= self.max_candidates: break
                
            url = candidate.get("preview_url")
            asset_format = candidate.get("preview_format", "unknown")
            if not url or not asset_format: continue
            
            try:
                if url.startswith("http"): asset_bytes = self._safe_download(url)
                else:
                    if not os.path.isfile(url) or os.path.getsize(url) > self.max_preview_bytes: continue
                    with open(url, "rb") as f: asset_bytes = f.read()
                        
                if asset_format in {"mp4", "webm", "mov", "gif", "webp"}:
                    frames = self._extract_frames_unified(asset_bytes, cell_w, cell_h)
                else:
                    continue
                
                if not frames: continue 
                    
            except Exception as e:
                continue

            label = chr(65 + successful_candidates)
            mapping[label] = candidate["candidate_id"]
            
            row_canvas = Image.new("RGBA", (label_width + (cell_w * 3), cell_h), (20, 20, 20, 255))
            row_canvas.paste(self._draw_large_label(label, label_width, cell_h), (0, 0))
            for f_idx, f_img in enumerate(frames):
                row_canvas.paste(f_img, (label_width + (f_idx * cell_w), 0))
                
            row_images.append(row_canvas)
            title = str(candidate.get("title", "Untitled"))
            dur = candidate.get("duration")
            duration_text = f"{float(dur):.2f}s" if isinstance(dur, (int, float)) else "unknown"
            metadata_list.append(f"Row {label}: {title} ({candidate.get('width', 0)}x{candidate.get('height', 0)}, {duration_text}, {asset_format})")
            successful_candidates += 1
            
        if not row_images: return "", {}, []

        contact_sheet = Image.new("RGB", (row_images[0].width, sum(img.height for img in row_images) + (10 * len(row_images))), (0, 0, 0))
        y_offset = 0
        for img in row_images:
            contact_sheet.paste(img, (0, y_offset))
            y_offset += img.height + 10
            
        os.makedirs("data/creative/contact_sheets", exist_ok=True)
        contact_sheet.save(f"data/creative/contact_sheets/{safe_clip_id}_contact_sheet.jpg", format="JPEG", quality=85)
        
        with open(f"data/creative/contact_sheets/{safe_clip_id}_mapping.json", "w") as f: json.dump(mapping, f, indent=2)
        with open(f"data/creative/contact_sheets/{safe_clip_id}_metadata.json", "w") as f: json.dump(metadata_list, f, indent=2)

        buffered = BytesIO()
        contact_sheet.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8'), mapping, metadata_list

    def rank_candidates(self, clip_id: str, b64_img: str, mapping: dict, metadata_list: list, story: dict, plan: dict) -> dict:
        fallback_chain = plan.get("creative_decision", {}).get("fallback_chain", ["TEXT_AND_SOUND", "SOUND_ONLY", "NO_MEME"])
        
        if not b64_img or not mapping:
            return self._save_decision(clip_id, {
                "selected_candidate_id": None, 
                "selection_confidence": 0.0, 
                "none_suitable": True, 
                "reason": "No valid candidates generated.", 
                "fallback_chain": fallback_chain
            })

        # --- FULLY RESTORED CONTEXT VARIABLES ---
        scene_description = story.get("scene_description", "unknown")
        story_confidence = story.get("confidence", 0.0)
        outcome = story.get("actual_outcome", "unknown")
        expectation = story.get("viewer_expectation", "unknown")
        comedy_mech = story.get("comedy_mechanism", "unknown")
        uncertainties = ", ".join(story.get("uncertainties", [])) or "None"
        
        treatment = plan.get("creative_decision", {}).get("treatment", "REACTION_OVERLAY")
        intensity = plan.get("placement", {}).get("intensity", "medium")
        reaction_window = story.get("reaction_window", {})
        window_duration = reaction_window.get("end", 0) - reaction_window.get("start", 0)
        
        avoid_str = ", ".join(plan.get("creative_decision", {}).get("avoid_concepts", [])) or "None specified"
        
        label_choices = "/".join(mapping.keys()) + "/NONE"
        meta_str = "\n".join(metadata_list)

        system_prompt = f"""
        You are a Meme Audition Judge. Evaluate the Contact Sheet.
        
        SCENE CONTEXT:
        - Scene: {scene_description}
        - Expected: {expectation}
        - Outcome: {outcome}
        - Mechanism: {comedy_mech}
        - Treatment Type: {treatment}
        - Window Duration: {window_duration}s
        - Intensity: {intensity}
        - Uncertainties: {uncertainties}
        - Qwen Story Confidence: {story_confidence}
        - AVOID: {avoid_str}
        
        CANDIDATE METADATA:
        {meta_str}
        
        TASK: Which row improves the scene, fits the duration, and avoids the banned concepts?
        If ALL rows mismatch, ruin the joke, or fail requirements, MUST select "NONE".
        
        OUTPUT SCHEMA:
        {{
            "selected_row": "{label_choices}",
            "reason": "Temporal and semantic explanation.",
            "confidence": 0.85
        }}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}]}
        ]

        try:
            # Using the new robust Colab fallback method
            raw = self._execute_with_colab_fallback(messages)
            
            cleaned = raw.strip()
            md_ticks = chr(96) * 3 
            md_json = md_ticks + "json"
            if cleaned.startswith(md_json): cleaned = cleaned[len(md_json):]
            elif cleaned.startswith(md_ticks): cleaned = cleaned[len(md_ticks):]
            if cleaned.endswith(md_ticks): cleaned = cleaned[:-len(md_ticks)]
            
            decision = json.loads(cleaned.strip())
            selected_row = str(decision.get("selected_row", "NONE")).upper()
            
            try: confidence = max(0.0, min(1.0, float(decision.get("confidence", 0.0))))
            except (ValueError, TypeError): confidence = 0.0
                
            if selected_row not in mapping and selected_row != "NONE":
                selected_row, reason = "NONE", "Hallucinated Row Selection"
            elif selected_row != "NONE" and confidence < self.minimum_selection_confidence:
                selected_row, reason = "NONE", "Below confidence threshold."
            else:
                reason = decision.get("reason", "")
            
            is_none = (selected_row == "NONE")
            final_id = mapping.get(selected_row) if not is_none else None
            
            return self._save_decision(clip_id, {
                "selected_candidate_id": final_id, 
                "selection_confidence": confidence,
                "none_suitable": is_none, 
                "reason": reason, 
                "fallback_chain": fallback_chain
            })

        except Exception as e:
            print(f"❌ [Ranker Error]: {str(e)}")
            return self._save_decision(clip_id, {
                "selected_candidate_id": None, 
                "selection_confidence": 0.0, 
                "none_suitable": True, 
                "reason": f"API Error: {str(e)}", 
                "fallback_chain": fallback_chain
            })