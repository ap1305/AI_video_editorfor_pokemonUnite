import os
import json
import subprocess
import shutil
import re
import tempfile
import time
import uuid
import math
from typing import Dict, Any, List, Optional, Tuple

def safe_float(val: Any, default: float = 0.0) -> float:
    try: return float(val)
    except (ValueError, TypeError): return default

class CreativeRenderer:
    ALLOWED_TREATMENTS = {"REACTION_OVERLAY", "TEXT_AND_SOUND", "SOUND_ONLY", "NO_MEME"}

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.renders_dir = "data/creative/renders"
        self.history_path = "data/creative/usage_history.json"
        self.audits_dir = "data/creative/render_audits"
        
        os.makedirs(self.renders_dir, exist_ok=True)
        os.makedirs(self.audits_dir, exist_ok=True)
        
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("FATAL: ffmpeg or ffprobe not found in system PATH.")
            
        self.has_nvenc = self._verify_nvenc_capability()
        self.timeout = int(self.config.get("render_timeout", 300))
        
        # Configurable Technical Defaults
        self.enc_nvenc_preset = self.config.get("nvenc_preset", "p4")
        self.enc_nvenc_bitrate = self.config.get("nvenc_bitrate", "8M")
        self.enc_x264_preset = self.config.get("libx264_preset", "fast")
        self.enc_x264_crf = self.config.get("libx264_crf", "23")
        
        self.text_def_color = self.config.get("text_default_color", "white")
        self.text_def_boxcolor = self.config.get("text_default_boxcolor", "black@0.6")
        self.text_def_borderw = int(self.config.get("text_default_borderw", 10))

    def _sanitize_id(self, raw_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", str(raw_id))

    def _escape_filter_path(self, path: str) -> str:
        if not path: return ""
        return os.path.abspath(path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")

    def _verify_nvenc_capability(self) -> bool:
        try:
            res = subprocess.run(["ffmpeg", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if "h264_nvenc" not in res.stdout: return False
            
            test_cmd = [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=30:d=1",
                "-c:v", "h264_nvenc", "-preset", "p1", "-f", "null", "-"
            ]
            test_res = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            return test_res.returncode == 0
        except Exception:
            return False

    def _probe_media(self, filepath: str) -> dict:
        if not os.path.isfile(filepath): return {"valid": False, "error": "Missing file."}
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height,codec_type", "-of", "json", filepath]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
            if res.returncode != 0: return {"valid": False, "error": "ffprobe failed."}
            data = json.loads(res.stdout)
            
            duration = safe_float(data.get("format", {}).get("duration", 0.0))
            if duration <= 0: return {"valid": False, "error": "Zero duration."}
            
            w, h = 0, 0
            has_v, has_a = False, False
            for s in data.get("streams", []):
                if s.get("codec_type") == "video":
                    w, h = int(s.get("width", 0)), int(s.get("height", 0))
                    has_v = True
                elif s.get("codec_type") == "audio":
                    has_a = True
                    
            return {"valid": True, "duration": duration, "width": w, "height": h, "has_video": has_v, "has_audio": has_a}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def _validate_box(self, box: Any) -> bool:
        if not isinstance(box, list) or len(box) != 4: return False
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
            if not all(math.isfinite(v) for v in [x1, y1, x2, y2]): return False
            if not (0.0 <= x1 < x2 <= 1.0) or not (0.0 <= y1 < y2 <= 1.0): return False
            return True
        except (ValueError, TypeError):
            return False

    def _validate_timestamps(self, start_t: Any, end_t: Any, clip_dur: float) -> Tuple[bool, float, float]:
        try:
            s = float(start_t)
            e = float(end_t)
            if not (math.isfinite(s) and math.isfinite(e)): return False, 0.0, 0.0
            if s < 0.0 or e > clip_dur or s >= e: return False, 0.0, 0.0
            return True, s, e
        except (ValueError, TypeError):
            return False, 0.0, 0.0

    def _atomic_history_update(self, asset_id: str) -> bool:
        if not asset_id: return False
        lock_path = self.history_path + ".lock"
        
        # Concurrency safety: wait for lock
        for _ in range(100): 
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.close(fd)
                
                history = []
                if os.path.exists(self.history_path):
                    try:
                        with open(self.history_path, 'r') as f: history = json.load(f)
                    except Exception: pass
                    
                if not isinstance(history, list): history = []
                history.append(asset_id)
                history = history[-100:] 
                
                part_path = f"{self.history_path}.{uuid.uuid4().hex[:8]}.part"
                with open(part_path, 'w') as f: json.dump(history, f, indent=2)
                os.replace(part_path, self.history_path)
                
                os.remove(lock_path)
                return True
            except FileExistsError:
                time.sleep(0.1)
            except Exception:
                if os.path.exists(lock_path): os.remove(lock_path)
                return False
        return False

    def _save_audit(self, clip_id: str, success: bool, audit_data: dict) -> dict:
        audit_data["success"] = success
        run_id = uuid.uuid4().hex[:8]
        audit_path = os.path.join(self.audits_dir, f"{self._sanitize_id(clip_id)}_{run_id}_audit.json")
        part_path = audit_path + ".part"
        
        try:
            with open(part_path, 'w') as f: json.dump(audit_data, f, indent=2)
            os.replace(part_path, audit_path)
        except Exception:
            if os.path.exists(part_path): os.remove(part_path)
        return audit_data

    def _execute_render(self, cmd: list, target_path: str, part_path: str, exp_dur: float, req_audio: bool, exp_w: int, exp_h: int) -> Tuple[bool, str, float, dict]:
        start_time = time.time()
        probe_res = {}
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=self.timeout)
            elapsed = time.time() - start_time
            
            if res.returncode != 0:
                if os.path.exists(part_path): os.remove(part_path)
                return False, f"FFmpeg error: {res.stderr[-1000:]}", elapsed, probe_res
                
            if not os.path.exists(part_path) or os.path.getsize(part_path) == 0:
                return False, "Render succeeded but output file is missing or zero bytes.", elapsed, probe_res
                
            probe = self._probe_media(part_path)
            probe_res = probe
            if not probe["valid"] or not probe["has_video"]:
                if os.path.exists(part_path): os.remove(part_path)
                return False, "Output file is invalid or missing video stream.", elapsed, probe_res
                
            if req_audio and not probe["has_audio"]:
                if os.path.exists(part_path): os.remove(part_path)
                return False, "Output file is missing required audio stream.", elapsed, probe_res
                
            if abs(probe["duration"] - exp_dur) > 1.0:
                if os.path.exists(part_path): os.remove(part_path)
                return False, f"Duration mismatch. Expected {exp_dur:.1f}s, got {probe['duration']:.1f}s", elapsed, probe_res

            # Allow for potential even-dimension rounding from source
            if abs(probe["width"] - exp_w) > 2 or abs(probe["height"] - exp_h) > 2:
                if os.path.exists(part_path): os.remove(part_path)
                return False, f"Dimension mismatch. Expected {exp_w}x{exp_h}, got {probe['width']}x{probe['height']}", elapsed, probe_res
                
            os.replace(part_path, target_path)
            return True, "Success", elapsed, probe_res
            
        except subprocess.TimeoutExpired:
            if os.path.exists(part_path): os.remove(part_path)
            return False, "FFmpeg timeout expired.", time.time() - start_time, probe_res
        except Exception as e:
            if os.path.exists(part_path): os.remove(part_path)
            return False, str(e), time.time() - start_time, probe_res

    def render_clip(self, render_plan_path: str, paced_video_path: str) -> dict:
        run_id = uuid.uuid4().hex[:8]
        
        audit = {
            "clip_id": "unknown", "treatment": "UNKNOWN", "encoder_used": "none",
            "render_safe_passed": False, "target_output": "", 
            "input_paced_video": paced_video_path,
            "render_plan_path": render_plan_path,
            "visual_asset_used": None,
            "sound_asset_used": None,
            "output_probe": {},
            "history_updated": False,
            "logs": []
        }
        
        if not os.path.exists(render_plan_path):
            audit["logs"].append("Plan not found.")
            return self._save_audit("unknown", False, audit)
        try:
            with open(render_plan_path, 'r') as f: plan = json.load(f)
        except Exception:
            audit["logs"].append("Plan unreadable.")
            return self._save_audit("unknown", False, audit)
            
        clip_id = self._sanitize_id(plan.get("clip_id", "unknown_clip"))
        audit["clip_id"] = clip_id
        
        if not plan.get("render_safe") or plan.get("timeline_basis") != "PACED_CLIP":
            audit["logs"].append("Plan marked NOT render_safe or wrong timeline basis.")
            return self._save_audit(clip_id, False, audit)
            
        audit["render_safe_passed"] = True
        treatment = str(plan.get("final_resolved_treatment", "")).upper()
        if treatment not in self.ALLOWED_TREATMENTS:
            audit["logs"].append(f"Unsupported treatment enum: {treatment}")
            return self._save_audit(clip_id, False, audit)
            
        audit["treatment"] = treatment
        params = plan.get("render_parameters", {})
        if not isinstance(params, dict): params = {}

        base_meta = self._probe_media(paced_video_path)
        if not base_meta["valid"] or not base_meta["has_video"]:
            audit["logs"].append("Base video invalid or missing video stream.")
            return self._save_audit(clip_id, False, audit)
            
        b_w, b_h, has_audio = base_meta["width"], base_meta["height"], base_meta["has_audio"]
        clip_dur = base_meta["duration"]
        audit["expected_duration"] = clip_dur

        # Run-specific output locking/concurrency
        target_output = os.path.join(self.renders_dir, f"{clip_id}_final_{run_id}.mp4")
        part_output = target_output + ".part"
        audit["target_output"] = target_output

        base_cmd = ["ffmpeg", "-y", "-i", paced_video_path]
        filter_complex = []
        inputs_count = 1
        txt_tmp_file = None
        
        # Base Normalization (Applies to NO_MEME as well)
        filter_complex.append("[0:v:0]setpts=PTS-STARTPTS,setsar=1[base_v]")
        last_v = "[base_v]"
        last_a = "0:a:0?" if has_audio else None
        if has_audio:
            filter_complex.append("[0:a:0]asetpts=PTS-STARTPTS[base_a]")
            last_a = "[base_a]"
        
        try:
            # ---------------------------------------------------------
            # A. REACTION_OVERLAY
            # ---------------------------------------------------------
            if treatment == "REACTION_OVERLAY":
                asset_path = params.get("asset_path")
                if not asset_path: raise ValueError("Overlay asset missing from params.")
                
                asset_probe = self._probe_media(asset_path)
                if not asset_probe["valid"] or not asset_probe["has_video"]:
                    raise ValueError(f"Visual asset missing or unreadable: {asset_path}")
                audit["visual_asset_used"] = asset_path
                
                valid_t, start_t, end_t = self._validate_timestamps(params.get("start_time"), params.get("end_time"), clip_dur)
                if not valid_t: raise ValueError("Invalid overlay interval.")
                
                ov_dur = end_t - start_t
                opacity = max(0.0, min(1.0, safe_float(params.get("opacity", 1.0))))
                
                box = params.get("placement", {}).get("box")
                if not self._validate_box(box): raise ValueError("Invalid placement box.")
                
                w_px = max(2, int(round((box[2]-box[0])*b_w) / 2) * 2)
                h_px = max(2, int(round((box[3]-box[1])*b_h) / 2) * 2)
                x_px = max(0, int(box[0]*b_w))
                y_px = max(0, int(box[1]*b_h))
                if x_px + w_px > b_w or y_px + h_px > b_h: raise ValueError("Overlay dimensions exceed base video bounds.")

                loop_policy = params.get("loop_policy", "NO_LOOP")
                if loop_policy == "LOOP_TO_INTERVAL": base_cmd.extend(["-stream_loop", "-1"])
                    
                base_cmd.extend(["-i", asset_path])
                ov_idx = inputs_count
                inputs_count += 1
                
                filter_complex.append(
                    f"[{ov_idx}:v]trim=duration={ov_dur},setpts=PTS-STARTPTS+{start_t}/TB,"
                    f"scale={w_px}:{h_px}:force_original_aspect_ratio=decrease,format=rgba,colorchannelmixer=aa={opacity}[ovr]"
                )
                filter_complex.append(f"{last_v}[ovr]overlay=x={x_px}:y={y_px}:enable='between(t,{start_t},{end_t})':eof_action=pass:repeatlast=0[v_out]")
                last_v = "[v_out]"

            # ---------------------------------------------------------
            # B. TEXT_AND_SOUND / SOUND_ONLY
            # ---------------------------------------------------------
            if treatment in ["TEXT_AND_SOUND", "SOUND_ONLY"]:
                valid_t, start_t, end_t = self._validate_timestamps(params.get("start_time"), params.get("end_time"), clip_dur)
                if not valid_t: raise ValueError("Invalid reaction interval.")

                if treatment == "TEXT_AND_SOUND":
                    txt_p = params.get("text", {})
                    content = str(txt_p.get("content", ""))
                    font_path = self._escape_filter_path(txt_p.get("font", ""))
                    
                    if not content or not font_path or not os.path.exists(txt_p.get("font", "")):
                        raise ValueError("Text contract missing content or real font file.")
                        
                    fd, txt_tmp_file = tempfile.mkstemp(suffix=".txt", text=True)
                    with os.fdopen(fd, 'w', encoding='utf-8') as f: f.write(content)
                    safe_txt_file = self._escape_filter_path(txt_tmp_file)
                        
                    f_size = int(txt_p.get("font_size", 50))
                    box = params.get("placement", {}).get("box")
                    if not self._validate_box(box): raise ValueError("Invalid text placement box.")
                    
                    x_px = int(box[0]*b_w) + int(txt_p.get("margin_x", 10))
                    y_px = int(box[1]*b_h) + int(txt_p.get("margin_y", 10))
                    
                    f_color = txt_p.get("fontcolor", self.text_def_color)
                    b_color = txt_p.get("boxcolor", self.text_def_boxcolor)
                    b_width = int(txt_p.get("boxborderw", self.text_def_borderw))
                    
                    filter_complex.append(
                        f"{last_v}drawtext=fontfile='{font_path}':textfile='{safe_txt_file}':"
                        f"fontsize={f_size}:fontcolor={f_color}:box=1:boxcolor={b_color}:boxborderw={b_width}:"
                        f"x={x_px}:y={y_px}:enable='between(t,{start_t},{end_t})'[v_out]"
                    )
                    last_v = "[v_out]"

                snd_p = params.get("sound", {})
                if snd_p:
                    snd_path = snd_p.get("path")
                    snd_probe = self._probe_media(snd_path) if snd_path else {"valid": False}
                    if not snd_probe["valid"] or not snd_probe["has_audio"]:
                        raise ValueError(f"Sound asset missing or unreadable: {snd_path}")
                    audit["sound_asset_used"] = snd_path
                        
                    s_end_t = min(safe_float(snd_p.get("end_time", end_t)), clip_dur)
                    if s_end_t <= start_t: raise ValueError("Invalid sound interval.")
                    s_dur = s_end_t - start_t
                    
                    base_cmd.extend(["-i", snd_path])
                    s_idx = inputs_count
                    inputs_count += 1
                    
                    duck_db = max(-60.0, min(0.0, safe_float(snd_p.get("ducking_db", -5.0))))
                    gain = max(0.0, min(2.0, safe_float(snd_p.get("gain", 1.0))))
                    delay_ms = int(start_t * 1000)
                    
                    filter_complex.append(
                        f"[{s_idx}:a]atrim=duration={s_dur},asetpts=PTS-STARTPTS,"
                        f"volume={gain},adelay={delay_ms}:all=1[sfx]"
                    )
                    
                    if has_audio:
                        filter_complex.append(f"{last_a}volume=enable='between(t,{start_t},{s_end_t})':volume={duck_db}dB[ducked_base]")
                        filter_complex.append(f"[ducked_base][sfx]amix=inputs=2:duration=first:dropout_transition=0,alimiter=level_in=1:level_out=1:limit=0.9[a_out]")
                        last_a = "[a_out]"
                    else:
                        last_a = "[sfx]"

            # Finalize command
            filter_args = []
            if filter_complex: filter_args = ["-filter_complex", ";".join(filter_complex)]
                
            map_args = ["-map", last_v]
            if last_a: map_args.extend(["-map", last_a])

            output_common = ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
            
            encoders = [("NVENC", ["-c:v", "h264_nvenc", "-preset", self.enc_nvenc_preset, "-b:v", self.enc_nvenc_bitrate])] if self.has_nvenc else []
            encoders.append(("LIBX264", ["-c:v", "libx264", "-preset", self.enc_x264_preset, "-crf", self.enc_x264_crf]))
            
            success = False
            for enc_name, enc_flags in encoders:
                audit["encoder_used"] = enc_name
                print(f"🎬 [Renderer] Encoding {clip_id} via {enc_name} to {target_output}...")
                
                test_cmd = base_cmd + filter_args + map_args + enc_flags + output_common + [part_output]
                
                success, msg, elapsed, probe_res = self._execute_render(test_cmd, target_output, part_output, exp_dur=clip_dur, req_audio=(has_audio or bool(params.get("sound"))), exp_w=b_w, exp_h=b_h)
                audit["logs"].append(f"{enc_name} Result: {msg}")
                audit["elapsed_time"] = round(elapsed, 2)
                audit["output_probe"] = probe_res
                
                if success: break
                
            if not success:
                return self._save_audit(clip_id, False, audit)

            # Success Post-Processing
            if plan.get("usage_mark_required_after_render"):
                audit["history_updated"] = self._atomic_history_update(plan.get("selected_asset", {}).get("candidate_id"))
                
            return self._save_audit(clip_id, True, audit)

        except Exception as e:
            audit["logs"].append(f"Renderer logic crashed: {e}")
            if os.path.exists(part_output): os.remove(part_output)
            return self._save_audit(clip_id, False, audit)
            
        finally:
            if txt_tmp_file and os.path.exists(txt_tmp_file):
                try: os.remove(txt_tmp_file)
                except Exception: pass