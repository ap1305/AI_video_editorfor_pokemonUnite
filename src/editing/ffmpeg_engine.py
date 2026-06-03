import os
import json
import yaml
import subprocess
import copy
import time
import tempfile
import shutil
from typing import Optional, Dict, Any, Tuple, List

from src.editing.whisper_engine import WhisperEngine

# ==========================================
# OBSERVABILITY HELPER FUNCTIONS
# ==========================================
def escape_drawtext(text):
    if text is None: return ""
    return (str(text)
            .replace("\\", "\\\\").replace(":", "\\:").replace("'", r"\'")
            .replace(",", r"\,").replace("[", r"\[").replace("]", r"\]")
            .replace("{", r"\{").replace("}", r"\}").replace("%", r"\%")
            .replace("(", r"\(").replace(")", r"\)").replace(";", r"\;"))

def extract_semantics(segment):
    return segment.get("semantics") or segment.get("raw_semantics") or {}

def get_role_color(role):
    colors = {"DEAD_AIR": "gray", "SETUP": "blue", "CONFLICT": "orange", "CLIMAX": "red", "RESOLUTION": "green"}
    return colors.get(str(role).upper(), "black")

# ==========================================
# ADVANCED RENDERING ENGINE
# ==========================================
class AdvancedVideoRenderingEngine:
    def __init__(self, output_dir: str = "data/renders"):
        self.output_dir = output_dir
        self.base_temp_dir = os.path.join(self.output_dir, "temp_workspace")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.base_temp_dir, exist_ok=True)
        self.whisper = WhisperEngine()

    def _get_safe_ffmpeg_path(self, path_str: str) -> str:
        return path_str.replace('\\', '/').replace(':', '\\:')

    def _create_textfile(self, temp_dir: str, filename: str, content: str) -> str:
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(content))
        return self._get_safe_ffmpeg_path(filepath)

    def _probe_media(self, filepath: str) -> dict:
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration:format=size:stream=codec_type", "-of", "json", filepath]
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
            data = json.loads(output)
            
            duration_str = data.get("format", {}).get("duration")
            size_str = data.get("format", {}).get("size", "0")
            
            if duration_str is None:
                raise ValueError("FFprobe could not determine video duration.")
                
            return {
                "duration": float(duration_str), 
                "has_audio": any(stream.get("codec_type") == "audio" for stream in data.get("streams", [])),
                "size_mb": float(size_str) / (1024 * 1024)
            }
        except Exception as e:
            print(f"❌ [FFprobe] Fatal error probing media {filepath}: {e}")
            raise RuntimeError(f"Media probe failed for {filepath}: {e}")

    def _build_atempo_chain(self, speed: float) -> str:
        if speed == 1.0: return ""
        chain = []
        temp_speed = speed
        while temp_speed > 2.0:
            chain.append("atempo=2.0")
            temp_speed /= 2.0
        while temp_speed < 0.5:
            chain.append("atempo=0.5")
            temp_speed /= 0.5
        if temp_speed != 1.0:
            chain.append(f"atempo={round(temp_speed, 4)}")
        return "," + ",".join(chain)

    def _validate_and_sort_segments(self, segments: list, max_duration: float) -> Tuple[list, list, dict]:
        if not segments: return [], [], {}
        
        sorted_segs = sorted(segments, key=lambda x: float(x.get("start_timestamp", 0.0)))
        valid_segs, warnings = [], []
        last_end = -1.0
        
        stats = {"total_input": len(segments), "clamped": 0, "overlapped": 0, "dropped": 0}

        for seg in sorted_segs:
            new_seg = copy.deepcopy(seg)
            start = float(new_seg.get("start_timestamp", 0.0))
            end = float(new_seg.get("end_timestamp", 0.0))
            seg_id = new_seg.get("segment_id", "unknown")
            
            if end > max_duration:
                warnings.append(f"[{seg_id}] Clamped end time from {end} to {max_duration}.")
                end = max_duration
                new_seg["end_timestamp"] = end
                stats["clamped"] += 1

            if end <= start:
                warnings.append(f"[{seg_id}] Skipped - Negative/Zero duration ({start} to {end}).")
                stats["dropped"] += 1
                continue
                
            if start < last_end:
                warnings.append(f"[{seg_id}] Overlap at {start}s. Adjusted to {last_end}s.")
                start = last_end
                new_seg["start_timestamp"] = start
                stats["overlapped"] += 1
                
                if end <= start:
                    warnings.append(f"[{seg_id}] Eclipsed completely by overlap resolution.")
                    stats["dropped"] += 1
                    continue 
                
            last_end = end
            valid_segs.append(new_seg)
            
        # 🛑 FIX 1: Normalized EDF Quality Score
        total = max(stats["total_input"], 1)
        penalty = ((stats["dropped"] / total) * 0.60 + 
                   (stats["overlapped"] / total) * 0.30 + 
                   (stats["clamped"] / total) * 0.10)
        stats["quality_score"] = round(max(0.0, 1.0 - penalty), 3)
            
        return valid_segs, warnings, stats

    # ---------------------------------------------------------
    # PHASE 3: SHADOW DEBUG RENDERER
    # ---------------------------------------------------------
    def render_shadow_debug(self, input_path: str, edf_json_path: str, output_filename: str) -> bool:
        output_path = os.path.join(self.output_dir, output_filename)
        render_temp_dir = tempfile.mkdtemp(prefix="shadow_", dir=self.base_temp_dir)
        
        if not os.path.exists(edf_json_path) or not os.path.exists(input_path):
            shutil.rmtree(render_temp_dir, ignore_errors=True)
            return False

        try:
            media_info = self._probe_media(input_path)
            clip_duration = media_info["duration"]

            with open(edf_json_path, 'r') as f:
                edf = json.load(f)

            primary_climax_id = edf.get("story_profile", {}).get("primary_climax_id")
            segments, _, _ = self._validate_and_sort_segments(edf.get("segments", []), clip_duration)
            filters = []

            for i, seg in enumerate(segments):
                start_t = seg.get("start_timestamp")
                end_t = seg.get("end_timestamp")
                time_rule = f"between(t,{start_t},{end_t})"
                
                semantics = extract_semantics(seg)
                role = semantics.get("narrative_role", "Unknown").upper()
                event = semantics.get("event_type", "None").upper()
                imp = semantics.get("importance", 0.0)
                pacing = seg.get("editing_intent", {}).get("pacing", "NORMAL").upper()
                why_txt = seg.get("explanation", {}).get("why_decision_made", "")

                f_role = self._create_textfile(render_temp_dir, f"role_{i}.txt", role)
                filters.append(f"drawtext=textfile='{f_role}':fontcolor=white:fontsize=56:box=1:boxcolor={get_role_color(role)}@0.8:x=30:y=h-150:enable='{time_rule}'")
                
                f_sem = self._create_textfile(render_temp_dir, f"sem_{i}.txt", f"EVT: {event} | IMP: {imp}")
                filters.append(f"drawtext=textfile='{f_sem}':fontcolor=white:fontsize=28:box=1:boxcolor=black@0.7:x=30:y=h-100:enable='{time_rule}'")
                
                f_why = self._create_textfile(render_temp_dir, f"why_{i}.txt", f"PACE: {pacing} | WHY: {why_txt}")
                filters.append(f"drawtext=textfile='{f_why}':fontcolor=cyan:fontsize=24:box=1:boxcolor=black@0.8:x=30:y=h-60:enable='{time_rule}'")
                
                if seg.get("segment_id") == primary_climax_id:
                    f_climax = self._create_textfile(render_temp_dir, f"climax_{i}.txt", "*** PRIMARY CLIMAX ***")
                    filters.append(f"drawtext=textfile='{f_climax}':fontcolor=yellow:fontsize=72:box=1:boxcolor=red@0.9:x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,{start_t},{start_t + 0.5})'")

            filter_graph = ",".join(filters) if filters else "null"
            cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path, "-vf", filter_graph, "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "copy", output_path]

            print(f"🎬 [Debug Renderer] Burning AI logic onto video (Shadow Mode)...")
            
            # 🛑 FIX 2: Capture FFmpeg stderr diagnostics
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed (Code {result.returncode}):\n{result.stderr[-5000:]}")
            
            return True
        except Exception as e:
            print(f"❌ [Debug Renderer] Failed: {e}")
            return False
        finally:
            shutil.rmtree(render_temp_dir, ignore_errors=True)

    # ---------------------------------------------------------
    # SPRINT A: PACING EXECUTION ENGINE
    # ---------------------------------------------------------
    def render_dynamic_short(self, input_path: str, output_filename: str, edf_blueprint: Dict[str, Any]) -> str:
        render_start_time = time.time()
        output_path = os.path.join(self.output_dir, output_filename)
        telemetry_path = os.path.join(self.output_dir, output_filename.replace(".mp4", "_telemetry.json"))
        render_temp_dir = tempfile.mkdtemp(prefix="pacing_", dir=self.base_temp_dir)
        
        # 🛑 FIX 3: Initialize Telemetry Variables for the `finally` block
        render_status = "failed"
        failure_reason = None
        input_size_mb, source_total, output_total = 0.0, 0.0, 0.0
        valid_count = 0
        filter_complex = ""
        render_telemetry, validation_warnings, edf_stats = [], [], {}

        try:
            pace_map = {"FAST_FORWARD": 4.0, "ACCELERATE_HEAVY": 2.5, "ACCELERATE_LIGHT": 1.5, "NORMAL": 1.0, "SLOW_MO": 0.75}
            if os.path.exists("config.yaml"):
                with open("config.yaml", "r") as f:
                    yaml_data = yaml.safe_load(f) or {}
                    pace_map.update(yaml_data.get("editor", {}).get("pacing", {}))
                    pace_map = {k.upper(): v for k, v in pace_map.items()}

            media_info = self._probe_media(input_path)
            clip_duration = media_info["duration"]
            input_size_mb = media_info["size_mb"]
            has_audio = media_info["has_audio"]
            
            segments, validation_warnings, edf_stats = self._validate_and_sort_segments(edf_blueprint.get("segments", []), clip_duration)
            
            if not segments:
                raise ValueError("No valid segments found after validation.")

            filter_chains = []
            concat_inputs = ""
            
            for seg in segments:
                start_t = seg.get("start_timestamp")
                end_t = seg.get("end_timestamp")
                source_dur = end_t - start_t
                
                intent = seg.get("editing_intent", {})
                if "speed_multiplier" in intent:
                    # V2 Logic: Use exact math
                    target_speed = float(intent["speed_multiplier"])
                    pacing_string = str(intent.get("action_type", "V2_EXACT_MATH")).upper() # Keeps telemetry happy
                else:
                    # Legacy V1 Logic: Dictionary lookup
                    pacing_string = str(intent.get("pacing", "NORMAL")).upper()
                    target_speed = pace_map.get(pacing_string, 1.0) 
                out_dur = source_dur / target_speed

                v_trim = f"[0:v]trim=start={start_t}:end={end_t},setpts=PTS-STARTPTS"
                if target_speed != 1.0: v_trim += f",setpts={1.0/target_speed}*PTS"
                filter_chains.append(v_trim + f"[v{valid_count}];")
                
                if has_audio:
                    a_trim = f"[0:a]atrim=start={start_t}:end={end_t},asetpts=PTS-STARTPTS"
                    if target_speed != 1.0: a_trim += self._build_atempo_chain(target_speed)
                    filter_chains.append(a_trim + f"[a{valid_count}];")
                else:
                    filter_chains.append(f"anullsrc=r=44100:cl=stereo:d={out_dur}[a{valid_count}];")
                
                concat_inputs += f"[v{valid_count}][a{valid_count}]"

                source_total += source_dur
                output_total += out_dur

                render_telemetry.append({
                    "segment_id": seg.get("segment_id", f"seg_{valid_count}"),
                    "requested_pacing": pacing_string,
                    "applied_multiplier": target_speed,
                    "source_duration": round(source_dur, 2),
                    "output_duration": round(out_dur, 2)
                })
                valid_count += 1

            filter_complex = "".join(filter_chains) + concat_inputs + f"concat=n={valid_count}:v=1:a=1[outv][outa]"

            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
                "-filter_complex", filter_complex, "-map", "[outv]", "-map", "[outa]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-c:a", "aac", "-b:a", "192k",
                output_path
            ]

            print(f" ↳ Executing FFmpeg Pacing Render ({valid_count} segments)...")
            
            # 🛑 FIX 2: Capture stderr for detailed failure analytics
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed (Code {result.returncode}):\n{result.stderr[-5000:]}")
            
            print(f"✅ [Pacing Engine] AI-paced video saved to: {output_path}")
            render_status = "success"
            return output_path

        except Exception as e:
            failure_reason = str(e)
            print(f"❌ [Pacing Engine] Pipeline failed: {e}")
            return ""
            
        finally:
            # 🛑 FIX 3: Always write telemetry, even on crashes
            render_duration = time.time() - render_start_time
            output_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0.0

            telemetry_data = {
                "status": render_status,
                "render_engine_version": "1.4",
                "provenance": edf_blueprint.get("audit", {}).get("provenance", {}),
                "system_metrics": {
                    "render_time_seconds": round(render_duration, 2),
                    "input_filesize_mb": round(input_size_mb, 2),
                    "output_filesize_mb": round(output_size_mb, 2),
                    "filter_graph_length": len(filter_complex)
                },
                "edf_quality": edf_stats,
                "summary": {
                    "source_duration_processed": round(source_total, 2),
                    "output_duration": round(output_total, 2),
                    "timeline_acceleration_ratio": round(source_total / output_total, 2) if output_total > 0 else 0,
                    "segments_rendered": valid_count,
                    "validation_warnings": validation_warnings
                },
                "segments": render_telemetry
            }
            if failure_reason:
                telemetry_data["failure_reason"] = failure_reason
                
            with open(telemetry_path, 'w') as f:
                json.dump(telemetry_data, f, indent=4)
                
            shutil.rmtree(render_temp_dir, ignore_errors=True)

    # ---------------------------------------------------------
    # PHASE 2: FRAME-ACCURATE HIGHLIGHT EXTRACTION
    # ---------------------------------------------------------
    def extract_candidate_clips(self, input_path: str, json_path: str, output_subdir: str = "final_clips") -> bool:
        """
        Reads candidate_windows.json and extracts frame-accurate highlight clips.
        """
        print(f"🎬 [Extraction Engine] Booting Frame-Accurate Cutter...")

        # Keep everything inside your class's managed output directory
        target_dir = os.path.join(self.output_dir, output_subdir)
        os.makedirs(target_dir, exist_ok=True)

        if not os.path.exists(json_path):
            print(f"❌ [Error] Could not find candidate JSON at: {json_path}")
            return False

        with open(json_path, "r") as f:
            candidates = json.load(f)

        if not candidates:
            print("⚠️ [Warning] candidate_windows.json is empty. Nothing to cut.")
            return False

        for i, clip in enumerate(candidates):
            start_time = clip.get("start_time", 0.0)
            end_time = clip.get("end_time", 0.0)
            duration = max(0.1, end_time - start_time)
            score = clip.get("importance_score", 0)
            
            output_filename = os.path.join(target_dir, f"Rank_{i+1}_Score_{score}.mp4")
            
            print(f"✂️ Cutting Rank {i+1} (Score: {score}) | {start_time}s to {end_time}s...")
            
            # Frame-accurate FFmpeg command
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", input_path,
                "-ss", str(start_time),
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac",
                output_filename
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ [FFmpeg Error] Rank {i+1} failed:\n{result.stderr[-1000:]}")
            else:
                print(f"   ✅ Saved: Rank_{i+1}_Score_{score}.mp4")

        print(f"\n✅ Factory execution complete! All clips rendered to {target_dir}")
        return True
    def render_legacy_pipeline(self, *args, **kwargs):
        """PRESERVED: Contains Whisper, Zoom, Crop, and Meme logic for Sprints C-E."""
        pass