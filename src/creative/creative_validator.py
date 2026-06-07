import os
import json
import re
import requests
import subprocess
import shutil
from typing import Dict, Any, List, Tuple, Optional

def safe_float(val: Any, default: float = 0.0) -> float:
    try: return float(val)
    except (ValueError, TypeError): return default

def normalize_string_list(value: Any) -> List[str]:
    """Normalizes string or list inputs into a clean list of non-empty strings."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []

def intervals_overlap(start1: float, end1: float, start2: float, end2: float) -> bool:
    """True if two intervals overlap, exclusive of exact boundaries."""
    return max(start1, start2) < min(end1, end2)

class CreativeValidator:
    ALLOWED_TREATMENTS = {"REACTION_OVERLAY", "TEXT_AND_SOUND", "SOUND_ONLY", "NO_MEME"}
    ALLOWED_FULL_FORMATS = {"mp4", "webm", "mov", "gif"}

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_full_asset_bytes = int(self.config.get("max_full_asset_bytes", 50 * 1024 * 1024))
        self.minimum_confidence = float(self.config.get("minimum_selection_confidence", 0.60))
        self.max_overlay_dimension = float(self.config.get("max_overlay_dimension", 0.40))
        
        self.max_text_chars = int(self.config.get("max_text_characters", 100))
        self.max_text_lines = int(self.config.get("max_text_lines", 3))
        
        # Strictly Validate FFmpeg-compatible font path
        font_path = str(self.config.get("fallback_font_path", ""))
        if os.path.isfile(font_path) and font_path.lower().endswith((".ttf", ".otf")):
            self.fallback_font = font_path
        else:
            print("⚠️ [Validator] Configured font path is invalid. Text fallback will fail in FFmpeg unless overridden.")
            self.fallback_font = ""
        
        self.history_path = "data/creative/usage_history.json"
        self.session = requests.Session()
        self.ffmpeg_available = bool(shutil.which("ffprobe") and shutil.which("ffmpeg"))

        self.sound_catalog_path = self.config.get("sound_catalog_path", "assets/sounds/approved_catalogue.json")
        self.sound_catalog = self._load_sound_catalog()
        
        raw_protected = self.config.get("protected_regions", [
            [0.00, 0.00, 0.22, 0.28],  # Minimap
            [0.40, 0.00, 0.60, 0.12],  # Score & Timer
            [0.75, 0.65, 1.00, 1.00],  # Move Controls
            [0.35, 0.35, 0.65, 0.65],  # Core Combat
            [0.20, 0.85, 0.80, 1.00],  # Subtitles
            [0.00, 0.35, 0.20, 0.65]   # Scoring info
        ])
        self.protected_regions = self._normalize_boxes(raw_protected)

        self.anchor_points = {
            "TOP_LEFT": [0.23, 0.02], "TOP_RIGHT": [0.75, 0.02],
            "BOTTOM_LEFT": [0.02, 0.70], "BOTTOM_RIGHT": [0.75, 0.70],
            "CENTER_LEFT": [0.02, 0.40], "CENTER_RIGHT": [0.80, 0.40],
            "TOP_CENTER": [0.35, 0.15], "BOTTOM_CENTER": [0.35, 0.70]
        }

    def _load_sound_catalog(self) -> dict:
        if os.path.exists(self.sound_catalog_path):
            try:
                with open(self.sound_catalog_path, 'r') as f: return json.load(f)
            except Exception: pass
        return {} 

    def _resolve_sound_intent(self, raw_intent: Any) -> dict:
        intent_str = ""
        if isinstance(raw_intent, dict):
            intent_str = str(raw_intent.get("function", "")).strip()
        elif raw_intent and str(raw_intent).lower() != "none":
            intent_str = str(raw_intent).strip()
            
        intent_str = intent_str.lower()
        if not intent_str: return {}
        
        if intent_str in self.sound_catalog:
            entries = self.sound_catalog[intent_str]
            if isinstance(entries, list) and entries and isinstance(entries[0], dict): return entries[0]
            if isinstance(entries, dict): return entries
            
        for category, entries in self.sound_catalog.items():
            if isinstance(entries, dict): entries = [entries]
            if not isinstance(entries, list): continue
            for entry in entries:
                if isinstance(entry, dict):
                    aliases = [str(a).lower().strip() for a in entry.get("aliases", [])]
                    if intent_str in aliases: return entry
        return {}

    def _check_usage(self, asset_id: str) -> bool:
        if not asset_id or not os.path.exists(self.history_path): return False
        try:
            with open(self.history_path, 'r') as f: history = json.load(f)
            return asset_id in history[-5:]
        except Exception:
            return False

    def _sanitize_id(self, raw_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", str(raw_id))

    def _normalize_format(self, raw_fmt: str) -> str:
        """Extracts core extension from MIME types or dotted strings."""
        if not raw_fmt: return "mp4" # Default
        fmt = str(raw_fmt).lower().strip()
        fmt = fmt.replace("video/", "").replace("image/", "").replace(".", "")
        return fmt if fmt in self.ALLOWED_FULL_FORMATS else "mp4"

    def _escape_ffmpeg_text(self, text: str) -> str:
        if not text: return ""
        for char, esc in [("\\", "\\\\"), (":", "\\:"), ("%", "\\%"), ("'", "\u2019"), ("\n", " "), (",", "\\,")]:
            text = text.replace(char, esc)
        return text

    def _normalize_fallback_chain(self, raw_chain: Any, failed_treatment: Optional[str] = None) -> List[str]:
        normalized = []
        for value in normalize_string_list(raw_chain):
            treatment = value.upper()
            if treatment not in self.ALLOWED_TREATMENTS: continue
            if treatment in {failed_treatment, "NO_MEME"}: continue
            if treatment not in normalized:
                normalized.append(treatment)
        normalized.append("NO_MEME")
        return normalized

    def _normalize_windows(self, raw_windows: Any, clip_dur: float) -> List[dict]:
        if not isinstance(raw_windows, list): return []
        normalized = []
        for w in raw_windows:
            if isinstance(w, dict):
                s, e = safe_float(w.get("start")), safe_float(w.get("end"))
                if 0 <= s < e and s < clip_dur:
                    normalized.append({"start": s, "end": min(e, clip_dur)})
        return normalized

    def _normalize_boxes(self, raw_boxes: Any) -> List[List[float]]:
        valid = []
        if not isinstance(raw_boxes, list): return valid
        for b in raw_boxes:
            if isinstance(b, list) and len(b) == 4:
                try:
                    x1, y1, x2, y2 = [float(v) for v in b]
                    x1, x2 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))
                    y1, y2 = max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))
                    if x1 < x2 and y1 < y2:
                        valid.append([x1, y1, x2, y2])
                except (ValueError, TypeError): pass
        return valid

    def _validate_contract_identity(self, clip_id: str, story: dict, plan: dict) -> List[str]:
        errors = []
        if not self.ffmpeg_available: errors.append("FATAL: ffprobe or ffmpeg not found in PATH.")
        if str(story.get("clip_id", "")) != str(clip_id): errors.append("Story clip_id mismatch.")
        if str(plan.get("clip_id", "")) != str(clip_id): errors.append("Plan clip_id mismatch.")
        if story.get("timeline_basis") != "PACED_CLIP": errors.append("Story timeline_basis must be PACED_CLIP.")
        
        # Reaction window overlap check
        rw = story.get("reaction_window", {})
        rw_s, rw_e = safe_float(rw.get("start", 0.0)), safe_float(rw.get("end", 0.0))
        if rw_s >= rw_e or rw_e <= 0: 
            errors.append("FATAL: reaction_window is zero, negative, or invalid.")
        else:
            protected_windows = self._normalize_windows(story.get("protected_windows", []), float('inf'))
            for pw in protected_windows:
                # If reaction window overlaps at all with a protected window, reject upfront
                if intervals_overlap(rw_s, rw_e, pw["start"], pw["end"]):
                    errors.append("FATAL: reaction_window overlaps a protected gameplay window.")
                    break
        return errors

    def _intersects(self, box1: List[float], box2: List[float]) -> bool:
        return not (box1[2] <= box2[0] or box1[0] >= box2[2] or box1[3] <= box2[1] or box1[1] >= box2[3])

    def _probe_media(self, filepath: str, require_video: bool = False) -> dict:
        if not os.path.isfile(filepath): return {"valid": False, "error": "File does not exist."}
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,codec_type", "-of", "json", filepath]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            if res.returncode != 0: return {"valid": False, "error": "ffprobe execution failed."}
            data = json.loads(res.stdout)
            
            duration = safe_float(data.get("format", {}).get("duration", 0.0))
            width, height = 0, 0
            has_video, has_audio = False, False
            
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    width = int(stream.get("width", 0))
                    height = int(stream.get("height", 0))
                    has_video = True
                elif stream.get("codec_type") == "audio":
                    has_audio = True
                    
            if duration <= 0: return {"valid": False, "error": "Zero or unknown duration."}
            if has_video and (width <= 0 or height <= 0): return {"valid": False, "error": "Zero dimensions on visual asset."}
            if not has_video and not has_audio: return {"valid": False, "error": "No valid streams."}
            if require_video and not has_video: return {"valid": False, "error": "Missing required video stream."}
            
            return {"valid": True, "duration": duration, "width": width, "height": height, "has_video": has_video}
        except Exception as e:
            return {"valid": False, "error": f"Probe exception: {e}"}

    def _atomic_download_and_probe(self, url: str, target_path: str, format_ext: str) -> dict:
        if format_ext not in self.ALLOWED_FULL_FORMATS:
            return {"valid": False, "error": f"Format {format_ext} not allowed."}
            
        part_path = target_path + ".part"
        try:
            resp = self.session.get(url, stream=True, timeout=15)
            resp.raise_for_status()
            
            if 'image' not in resp.headers.get('Content-Type', '') and 'video' not in resp.headers.get('Content-Type', ''):
                raise ValueError("Invalid remote Content-Type.")

            downloaded = 0
            with open(part_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    downloaded += len(chunk)
                    if downloaded > self.max_full_asset_bytes:
                        raise ValueError("Byte limit exceeded.")
                    f.write(chunk)
            
            probe_result = self._probe_media(part_path, require_video=True)
            if not probe_result["valid"]:
                raise ValueError(f"Asset unreadable: {probe_result.get('error')}")
                
            os.replace(part_path, target_path) 
            probe_result["path"] = target_path
            return probe_result
            
        except Exception as e:
            if os.path.exists(part_path): os.remove(part_path)
            return {"valid": False, "error": str(e)}

    def _atomic_local_copy_and_probe(self, source_path: str, target_path: str) -> dict:
        if not os.path.isfile(source_path) or os.path.getsize(source_path) > self.max_full_asset_bytes:
            return {"valid": False, "error": "Local file missing or exceeds byte limit."}
            
        part_path = target_path + ".part"
        try:
            shutil.copy2(source_path, part_path)
            probe_result = self._probe_media(part_path, require_video=True)
            if not probe_result["valid"]:
                raise ValueError(f"Asset unreadable: {probe_result.get('error')}")
                
            os.replace(part_path, target_path)
            probe_result["path"] = target_path
            return probe_result
        except Exception as e:
            if os.path.exists(part_path): os.remove(part_path)
            return {"valid": False, "error": str(e)}

    def _resolve_timing(self, clip_dur: float, story: dict, plan: dict, asset_dur: Optional[float]) -> Tuple[float, float, bool, str]:
        reaction = story.get("reaction_window", {})
        placement = plan.get("placement", {})

        r_start = safe_float(reaction.get("start", 0.0))
        r_end = safe_float(reaction.get("end", clip_dur))

        p_start = safe_float(placement.get("suggested_trigger_timestamp", r_start))
        p_end = safe_float(placement.get("latest_end_timestamp", r_end))

        start = max(0.0, r_start, p_start)
        allowed_end = min(clip_dur, r_end, p_end)

        if start >= allowed_end: return 0.0, 0.0, False, "No valid reaction interval."

        if asset_dur and asset_dur > 0:
            end = min(start + asset_dur, allowed_end)
        else:
            end = allowed_end

        if start >= end: return 0.0, 0.0, False, "Resolved timing has zero duration."

        protected = self._normalize_windows(story.get("protected_windows", []), clip_dur)
        for pw in protected:
            if intervals_overlap(start, end, pw["start"], pw["end"]):
                return 0.0, 0.0, False, "Timing overlaps protected gameplay."

        return start, end, True, "Timing validated."

    def _find_safe_placement(self, preferred_regions: List[str], asset_w: int, asset_h: int, base_w: int, base_h: int, dynamic_exclusions: List[List[float]]) -> dict:
        if asset_w <= 0 or asset_h <= 0 or base_w <= 0 or base_h <= 0:
            return {"valid": False, "attempts": [], "reason": "Invalid dimensions."}
            
        raw_w = asset_w / base_w
        raw_h = asset_h / base_h
        
        uniform_scale = min(1.0, self.max_overlay_dimension / max(raw_w, raw_h))
        overlay_w = raw_w * uniform_scale
        overlay_h = raw_h * uniform_scale
        
        all_protected = self.protected_regions + dynamic_exclusions
        
        regions_to_try = [r for r in preferred_regions if r in self.anchor_points]
        for r in self.anchor_points.keys():
            if r not in regions_to_try: regions_to_try.append(r)
            
        attempts = []
        for scale in [1.0, 0.85, 0.70, 0.50]:
            scaled_w, scaled_h = overlay_w * scale, overlay_h * scale
            if scaled_w < 0.05 or scaled_h < 0.05: continue 
            
            for region in regions_to_try:
                anchor = self.anchor_points[region]
                x1, y1 = anchor[0], anchor[1]
                
                x1 = max(0.0, min(x1, 1.0 - scaled_w))
                y1 = max(0.0, min(y1, 1.0 - scaled_h))
                x2 = x1 + scaled_w
                y2 = y1 + scaled_h
                
                box = [x1, y1, x2, y2]
                conflict = any(self._intersects(box, pb) for pb in all_protected)
                
                attempt = {"region": region, "scale": round(uniform_scale * scale, 3), "result": "HUD_CONFLICT" if conflict else "SUCCESS"}
                attempts.append(attempt)
                
                if not conflict:
                    return {"valid": True, "box": box, "selected_region": region, "selected_scale": attempt["scale"], "attempts": attempts}
                    
        return {"valid": False, "attempts": attempts, "reason": "No safe region avoiding HUD."}

    # ==========================================
    # PUBLIC BOUNDARY (TRY/EXCEPT ENFORCED)
    # ==========================================
    def validate_clip(self, clip_id: str, paced_video_path: str, story: dict, plan: dict, ranker: dict, candidate_meta: Optional[dict] = None) -> dict:
        safe_clip_id = self._sanitize_id(clip_id)
        
        if not isinstance(story, dict): story = {}
        if not isinstance(plan, dict): plan = {}
        if not isinstance(ranker, dict): ranker = {}
        if candidate_meta is not None and not isinstance(candidate_meta, dict): candidate_meta = {}

        try:
            return self._validate_clip_internal(
                safe_clip_id=safe_clip_id, original_clip_id=clip_id, paced_video_path=paced_video_path,
                story=story, plan=plan, ranker=ranker, candidate_meta=candidate_meta
            )
        except Exception as exc:
            return self._finalize(
                clip_id=safe_clip_id, validation_passed=False, render_safe=True,
                treatment="NO_MEME", params={}, 
                validation_notes=[f"Unexpected validator error: {exc}"],
                fallback_steps=[], attempted_regions=[], plan=plan
            )

    def _validate_clip_internal(self, safe_clip_id: str, original_clip_id: str, paced_video_path: str, story: dict, plan: dict, ranker: dict, candidate_meta: Optional[dict]) -> dict:
        validation_notes = []
        fallback_steps = []
        fatal_error = False
        
        id_errors = self._validate_contract_identity(original_clip_id, story, plan)
        if id_errors:
            validation_notes.extend(id_errors)
            fatal_error = True
            
        base_vid = self._probe_media(paced_video_path, require_video=True)
        if not base_vid["valid"]:
            validation_notes.append(f"Paced clip invalid or not a video: {base_vid.get('error')}")
            fatal_error = True
            
        if fatal_error:
            return self._finalize(safe_clip_id, False, True, "NO_MEME", {}, validation_notes, [], [], plan)
            
        clip_dur, base_w, base_h = base_vid["duration"], base_vid["width"], base_vid["height"]

        creative_decision = plan.get("creative_decision", {})
        raw_exclusions = plan.get("placement", {}).get("dynamic_exclusions", [])
        dynamic_exclusions = self._normalize_boxes(raw_exclusions)
        
        current_treatment = str(creative_decision.get("treatment", "REACTION_OVERLAY")).upper()
        fallback_chain = self._normalize_fallback_chain(creative_decision.get("fallback_chain", []), failed_treatment=None)
        
        if current_treatment not in self.ALLOWED_TREATMENTS:
            current_treatment = fallback_chain.pop(0)

        preferred_regions = [r.upper() for r in normalize_string_list(plan.get("placement", {}).get("preferred_regions", ["TOP_RIGHT", "TOP_LEFT"]))]

        ranker_conf = max(0.0, min(1.0, safe_float(ranker.get("selection_confidence", 0.0))))
        selected_asset_id = ranker.get("selected_candidate_id")

        if current_treatment == "REACTION_OVERLAY":
            if ranker.get("none_suitable", False) or ranker_conf < self.minimum_confidence:
                validation_notes.append(f"Ranker aborted overlay. Confidence: {ranker_conf}")
                fallback_chain = self._normalize_fallback_chain(fallback_chain, failed_treatment="REACTION_OVERLAY")
                current_treatment = fallback_chain.pop(0)

        final_render_params = {}
        final_provenance = None
        attempted_regions = []
        usage_mark_required = False

        while current_treatment:
            valid = False
            state_notes = []
            
            if current_treatment == "REACTION_OVERLAY":
                if not selected_asset_id or not candidate_meta:
                    state_notes.append("Candidate metadata missing.")
                elif candidate_meta.get("candidate_id") != selected_asset_id:
                    state_notes.append("Candidate identity mismatch.")
                elif self._check_usage(selected_asset_id):
                    state_notes.append("Selected asset recently overused.")
                else:
                    url = str(candidate_meta.get("full_asset_url", ""))
                    fmt = self._normalize_format(candidate_meta.get("full_asset_format", ""))
                    
                    if fmt not in self.ALLOWED_FULL_FORMATS:
                        state_notes.append(f"Format '{fmt}' is not supported. Rejecting.")
                    else:
                        # CACHE FIX: Target path uses asset ID to prevent overwriting
                        safe_asset_id = self._sanitize_id(selected_asset_id)
                        target_path = f"data/creative/assets/{safe_clip_id}_{safe_asset_id}.{fmt}"
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        
                        dl_success = False
                        if url.startswith("http"):
                            probe = self._atomic_download_and_probe(url, target_path, fmt)
                        else:
                            probe = self._atomic_local_copy_and_probe(url, target_path)
                            
                        if not probe["valid"]:
                            state_notes.append(f"Asset extraction failed: {probe.get('error')}")
                            # Target path is untouched if download failed due to atomic .part
                        else:
                            start_t, end_t, time_safe, t_msg = self._resolve_timing(clip_dur, story, plan, probe["duration"])
                            if not time_safe:
                                state_notes.append(t_msg)
                            else:
                                placement = self._find_safe_placement(preferred_regions, probe["width"], probe["height"], base_w, base_h, dynamic_exclusions)
                                attempted_regions.extend(placement.get("attempts", []))
                                if not placement["valid"]:
                                    state_notes.append(placement.get("reason"))
                                else:
                                    final_render_params = {
                                        "asset_path": target_path, "start_time": start_t, "end_time": end_t,
                                        "placement": {"box": placement["box"], "region": placement["selected_region"], "scale": placement["selected_scale"]},
                                        "opacity": max(0.1, min(1.0, safe_float(plan.get("placement", {}).get("opacity", 1.0))))
                                    }
                                    final_provenance = {
                                        "candidate_id": selected_asset_id,
                                        "title": str(candidate_meta.get("title", "")),
                                        "provider": str(candidate_meta.get("provider", "")),
                                        "provider_asset_id": str(candidate_meta.get("provider_asset_id", "")),
                                        "search_query": str(candidate_meta.get("originating_search_query", "")),
                                        "retrieval_type": str(candidate_meta.get("retrieval_type", "")),
                                        "source_page": str(candidate_meta.get("source_page", "")),
                                        "source_url": url,
                                        "local_asset_path": target_path,
                                        "full_asset_format": fmt,
                                        "selection_confidence": ranker_conf,
                                        "ranker_reason": str(ranker.get("reason", ""))
                                    }
                                    usage_mark_required = True
                                    valid = True
                                    
                        if not valid and os.path.exists(target_path):
                            os.remove(target_path) # Clean only if we downloaded successfully but failed validation

            elif current_treatment == "TEXT_AND_SOUND":
                raw_text = str(creative_decision.get("fallback_text", "")).strip()
                if not raw_text:
                    state_notes.append("No fallback_text provided.")
                elif not self.fallback_font:
                    state_notes.append("No valid font path configured.")
                else:
                    if len(raw_text) > self.max_text_chars: raw_text = raw_text[:self.max_text_chars]
                    lines = raw_text.split("\n")
                    if len(lines) > self.max_text_lines: raw_text = "\n".join(lines[:self.max_text_lines])
                    
                    escaped_text = self._escape_ffmpeg_text(raw_text)
                    start_t, end_t, time_safe, t_msg = self._resolve_timing(clip_dur, story, plan, None)
                    if not time_safe:
                        state_notes.append(f"Text timing: {t_msg}")
                    else:
                        placement = self._find_safe_placement(["TOP_CENTER", "BOTTOM_CENTER"], int(base_w * 0.4), int(base_h * 0.15), base_w, base_h, dynamic_exclusions)
                        attempted_regions.extend(placement.get("attempts", []))
                        
                        if not placement["valid"]:
                            state_notes.append("Text HUD placement failed.")
                        else:
                            cat_entry = self._resolve_sound_intent(creative_decision.get("sound_intent", ""))
                            
                            snd_path = None
                            sound_end_time = None
                            snd_gain = 1.0
                            snd_ducking = -5.0
                            
                            if cat_entry:
                                pot_path = cat_entry.get("path", "")
                                probe = self._probe_media(pot_path)
                                if probe["valid"]:
                                    snd_path = pot_path
                                    sound_end_time = min(start_t + probe["duration"], end_t)
                                    snd_gain = max(0.0, min(2.0, safe_float(cat_entry.get("recommended_gain", 1.0))))
                                    snd_ducking = max(-60.0, min(0.0, safe_float(cat_entry.get("ducking_db", -5.0))))

                            final_render_params = {
                                "text": {
                                    "content": escaped_text,
                                    "font": self.fallback_font,
                                    "font_size": int(base_h * 0.05),
                                    "margin_x": int(base_w * 0.02),
                                    "margin_y": int(base_h * 0.02)
                                },
                                "start_time": start_t, "end_time": end_t,
                                "placement": {"box": placement["box"], "region": placement["selected_region"]},
                            }
                            if snd_path:
                                final_render_params["sound"] = {
                                    "path": snd_path, "start_time": start_t, "end_time": sound_end_time,
                                    "ducking": True, "ducking_db": snd_ducking, "gain": snd_gain
                                }
                            valid = True

            elif current_treatment == "SOUND_ONLY":
                cat_entry = self._resolve_sound_intent(creative_decision.get("sound_intent", ""))
                
                if not cat_entry:
                    state_notes.append("Valid sound function/alias missing or not in catalog.")
                else:
                    pot_path = cat_entry.get("path", "")
                    probe = self._probe_media(pot_path)
                    
                    if not probe["valid"]:
                        state_notes.append(f"Approved sound unreadable: {pot_path}")
                    else:
                        start_t, end_t, time_safe, t_msg = self._resolve_timing(clip_dur, story, plan, probe["duration"])
                        if time_safe:
                            snd_gain = max(0.0, min(2.0, safe_float(cat_entry.get("recommended_gain", 1.0))))
                            snd_ducking = max(-60.0, min(0.0, safe_float(cat_entry.get("ducking_db", -5.0))))
                            final_render_params = {
                                "sound": {
                                    "path": pot_path, "start_time": start_t, "end_time": end_t,
                                    "ducking": True, "ducking_db": snd_ducking, "gain": snd_gain
                                }
                            }
                            valid = True
                        else:
                            state_notes.append(f"Sound timing: {t_msg}")

            elif current_treatment == "NO_MEME":
                final_render_params = {}
                state_notes.append("Safely finalized on NO_MEME.")
                valid = True

            validation_notes.extend(state_notes)
            if valid: break
            
            fallback_chain = self._normalize_fallback_chain(fallback_chain, failed_treatment=current_treatment)
            next_treatment = fallback_chain.pop(0) if fallback_chain else "NO_MEME"
            fallback_steps.append(f"{current_treatment} -> {next_treatment}")
            current_treatment = next_treatment

        return self._finalize(
            clip_id=safe_clip_id, validation_passed=not fatal_error, render_safe=True,
            treatment=current_treatment, params=final_render_params, validation_notes=validation_notes,
            fallback_steps=fallback_steps, attempted_regions=attempted_regions, plan=plan, provenance=final_provenance,
            usage_mark=usage_mark_required
        )

    def _finalize(self, clip_id: str, validation_passed: bool, render_safe: bool, treatment: str, params: dict, validation_notes: List[str], fallback_steps: List[str], attempted_regions: List[dict], plan: dict, provenance: Optional[dict] = None, usage_mark: bool = False) -> dict:
        clean_notes = []
        for n in validation_notes:
            if n not in clean_notes: clean_notes.append(n)
            
        plan_out = {
            "clip_id": clip_id,
            "validation_passed": validation_passed,
            "render_safe": render_safe,
            "timeline_basis": "PACED_CLIP",
            "original_requested_treatment": str(plan.get("creative_decision", {}).get("treatment", "UNKNOWN")),
            "final_resolved_treatment": treatment,
            "selected_asset": provenance,
            "render_parameters": params,
            "attempted_regions": attempted_regions,
            "fallback_steps_taken": fallback_steps,
            "validation_notes": clean_notes,
            "usage_mark_required_after_render": usage_mark
        }
        os.makedirs("data/creative/render_plans", exist_ok=True)
        with open(f"data/creative/render_plans/{clip_id}_render_plan.json", "w") as f:
            json.dump(plan_out, f, indent=2)
        return plan_out