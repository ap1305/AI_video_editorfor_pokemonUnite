import os
import ffmpeg
import subprocess
from typing import Optional, Dict, Any

from src.editing.whisper_engine import WhisperEngine

class AdvancedVideoRenderingEngine:
    def __init__(self, output_dir: str = "data/renders"):
        self.output_dir = output_dir
        self.temp_dir = os.path.join(self.output_dir, "temp_workspace")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        self.whisper = WhisperEngine()

    def _cleanup_temp_files(self):
        for file in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def _build_atempo_chain(self, speed: float) -> str:
        """
        👉 FIX 5: Bypasses FFmpeg's 0.5 - 2.0 atempo limit by chaining filters.
        """
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

    def render_dynamic_short(
        self, input_path: str, output_filename: str, start_time: float, 
        end_time: float, edf_blueprint: Dict[str, Any], 
        meme_overlay_path: Optional[str] = None, bgm_path: Optional[str] = None
    ) -> str:
        
        output_path = os.path.join(self.output_dir, output_filename)
        temp_audio_path = os.path.join(self.temp_dir, "temp_audio.wav")
        temp_subs_path = os.path.join(self.temp_dir, "temp_subs.ass")
        paced_clip_path = os.path.join(self.temp_dir, "00_paced_base.mp4")
        
        if os.path.exists(output_path): os.remove(output_path)
        self._cleanup_temp_files()

        editing_plan = edf_blueprint.get("editing_plan", [])
        meme_candidates = edf_blueprint.get("meme_candidates", [])

        # ==========================================
        # PASS 1: THE IN-MEMORY GRAPH (Speed & Cuts)
        # ==========================================
        print(f"\n[FFmpeg] PASS 1: Building in-memory filter_complex graph...")
        
        # 👉 FIX 1 & 4: No disk chunks. Extract exact base clip first.
        base_clip_path = os.path.join(self.temp_dir, "base_isolation.mp4")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", 
                        "-ss", str(start_time), "-t", str(end_time - start_time), 
                        "-i", input_path, "-c", "copy", base_clip_path], check=True)

        if not editing_plan:
            subprocess.run(["cp", base_clip_path, paced_clip_path])
        else:
            filter_chains = []
            concat_inputs = ""
            
            for i, segment in enumerate(editing_plan):
                # ==========================================
                # THE TIME WARP FIX
                # Translate AI's absolute timestamps to local relative timestamps
                # ==========================================
                raw_start = float(segment.get("start", 0.0))
                raw_end = float(segment.get("end", 0.0))
                
                seg_start = max(0.0, raw_start - start_time)
                seg_end = max(0.1, raw_end - start_time)
                
                speed = float(segment.get("playback_action", {}).get("speed", 1.0))
                
                # Video Filter (Restored the missing speed math and output labels!)
                v_trim = f"[0:v]trim=start={seg_start}:end={seg_end},setpts=PTS-STARTPTS"
                if speed != 1.0: 
                    v_trim += f",setpts={1.0/speed}*PTS"
                v_trim += f"[v{i}];"
                
                # Audio Filter (with dynamic atempo chaining)
                a_trim = f"[0:a]atrim=start={seg_start}:end={seg_end},asetpts=PTS-STARTPTS"
                a_trim += self._build_atempo_chain(speed)
                a_trim += f"[a{i}];"
                
                filter_chains.extend([v_trim, a_trim])
                concat_inputs += f"[v{i}][a{i}]"

            filter_complex = "".join(filter_chains) + concat_inputs + f"concat=n={len(editing_plan)}:v=1:a=1[outv][outa]"

            # 👉 FIX 2 & 3: Re-encode output safely to prevent audio drift (-async 1)
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", base_clip_path,
                "-filter_complex", filter_complex,
                "-map", "[outv]", "-map", "[outa]",
                "-async", "1", # Forces audio sync correction
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-c:a", "aac", "-b:a", "192k",
                paced_clip_path
            ], check=True)

        # ==========================================
        # PASS 2: COMPOSITION (Subtitles & Memes)
        # ==========================================
        print(f"[FFmpeg] PASS 2: Composition & Asset Layering...")
        
        # Audio Extraction for Whisper
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", 
                        "-i", paced_clip_path, "-acodec", "pcm_s16le", 
                        "-ac", "1", "-ar", "16k", temp_audio_path], check=True)
        
        self.whisper.generate_viral_subtitles(temp_audio_path, temp_subs_path)

        stream = ffmpeg.input(paced_clip_path)
        video = stream.video
        audio = stream.audio

        # 👉 FIX 6: Safe Cropping Logic
        camera_fx = edf_blueprint.get("editing_plan", [{}])[0].get("camera_fx", {})
        zoom = max(1.0, 1.0 + float(camera_fx.get("strength", 0.0)))
        
        # Calculate safely using ffmpeg string expressions
        crop_w = f"in_h*9/16/{zoom}"
        crop_h = f"in_h/{zoom}"
        crop_x = f"(in_w-{crop_w})/2"
        crop_y = f"(in_h-{crop_h})/2"
        
        video = ffmpeg.filter(video, 'crop', crop_w, crop_h, crop_x, crop_y)
        video = ffmpeg.filter(video, 'scale', 1080, 1920)

        # Meme Composition (if applicable)
        if meme_overlay_path and os.path.exists(meme_overlay_path):
            meme_start = float(meme_candidates[0].get("trigger_timestamp", 2.0))
            meme_end = meme_start + 3.0 
            
            meme_input = ffmpeg.input(meme_overlay_path)
            meme_video = meme_input.video.filter('colorkey', '0x00FF00', 0.3, 0.2).filter('scale', 600, -1)
            
            video = ffmpeg.overlay(
                video, meme_video, 
                x='(main_w-overlay_w)/2', y='main_h-overlay_h-300',
                enable=f'between(t,{meme_start},{meme_end})', eof_action='pass'
            )
            
            delay_ms = int(meme_start * 1000)
            meme_audio = meme_input.audio.filter('adelay', f'{delay_ms}|{delay_ms}').filter('volume', 1.2) 
            ducked_audio = audio.filter('volume', 0.2, enable=f'between(t,{meme_start},{meme_end})')
            audio = ffmpeg.filter([ducked_audio, meme_audio], 'amix', inputs=2, duration='first', dropout_transition=2)

        # Subtitles & Final Render
        safe_subs_path = os.path.relpath(temp_subs_path).replace('\\', '/')
        video = ffmpeg.filter(video, 'ass', safe_subs_path)

        print("   ↳ Rendering final .mp4 file...")
        out = ffmpeg.output(video, audio, output_path, vcodec='libx264', acodec='aac', preset='fast', loglevel='error')
        ffmpeg.run(out, overwrite_output=True)
        
        print(f"✅ [FFmpeg] Render complete! Viral Short saved to: {output_path}")
        return output_path