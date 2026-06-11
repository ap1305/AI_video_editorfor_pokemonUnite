import os
import json
import shutil
import gc
from typing import List
from dotenv import load_dotenv

try:
    import torch
except ImportError:
    torch = None

from src.utils.youtube_downloader import fetch_youtube_video
from src.memory.chroma_client import TemporalMemoryEngine, MemeMemoryIndex, AudioMemoryIndex
from src.editing.ffmpeg_engine import AdvancedVideoRenderingEngine
from src.core.state_manager import FactoryStateManager
from src.memory.meme_manager import MemeManager  
from src.utils.logger import PipelineLogger

from src.utils.openai_client import run_macro_scout
from src.utils.qwen_pacing_analyzer import analyze_clip_pacing
from src.utils.pipeline_recovery import PipelineRecoveryManager

from src.editing.edf_compiler import generate_master_edf
from src.creative.vod_chopper import VODChopper  # 👇 ADDED IMPORT

load_dotenv()
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "ollama")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QWEN_PACING_URL = os.getenv("QWEN_PACING_URL")   

# ==========================================
# QUALITY GATE & VRAM MANAGEMENT
# ==========================================
def flush_vram():
    """Aggressively clears system RAM and local GPU VRAM."""
    gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()

def enforce_safety_and_evaluate(edf_timeline):
    """
    Enforces pacing safety without blindly rejecting meaning-preserving compression.
    Includes expanded metadata search to prevent silent failures.
    """
    total_duration = 0.0
    ff_duration = 0.0
    meaning_fastforwarded = False
    starts_with_long_dead_air = False

    protected_keywords = [
        "combat", "fight", "enemy", "score", "goal", "ko",
        "objective", "rayquaza", "zapdos", "clutch",
        "escape", "low_hp", "payoff"
    ]

    for idx, segment in enumerate(edf_timeline):
        start = float(segment.get("start_time", segment.get("start", 0)))
        end = float(segment.get("end_time", segment.get("end", 0)))
        seg_duration = max(0.0, end - start)
        total_duration += seg_duration

        # Schema-safe speed extraction
        speed = 1.0
        if "speed" in segment:
            speed = float(segment.get("speed", 1.0))
        elif isinstance(segment.get("playback_action"), dict):
            speed = float(segment["playback_action"].get("speed", 1.0))

        # Hard speed cap
        if speed > 2.0:
            print(f"⚠️ [Pacing Safety] Capping speed from {speed}x to 2.0x")
            speed = 2.0
            if "speed" in segment:
                segment["speed"] = speed
            elif isinstance(segment.get("playback_action"), dict):
                segment["playback_action"]["speed"] = speed

        # Avoid glitchy speedups on tiny segments
        if seg_duration < 2.0 and speed > 1.0:
            speed = 1.0
            if "speed" in segment:
                segment["speed"] = speed
            elif isinstance(segment.get("playback_action"), dict):
                segment["playback_action"]["speed"] = speed

        # 👇 EXPANDED LABEL SEARCH: Prevents silent failure if EDF keys vary 👇
        label_text = " ".join([
            str(segment.get("segment_type", "")),
            str(segment.get("label", "")),
            str(segment.get("semantic_label", "")),
            str(segment.get("reason", "")),
            str(segment.get("visual_summary", "")),
            str(segment.get("action", "")),
            str(segment.get("emphasis", "")),
            str(segment.get("description", "")),
            str(segment.get("content_type", "")),
        ]).lower()

        is_protected = any(k in label_text for k in protected_keywords)

        if speed > 1.0:
            ff_duration += seg_duration
            if is_protected:
                meaning_fastforwarded = True
            if idx == 0 and seg_duration > 3.0:
                starts_with_long_dead_air = True

    if total_duration <= 0:
        return False, "Failed: zero duration timeline."

    ff_ratio = ff_duration / total_duration

    if meaning_fastforwarded:
        return False, "Failed: meaning-critical action was fast-forwarded."

    if starts_with_long_dead_air:
        return False, "Failed: clip starts with more than 3 seconds of fast-forward/dead-air."

    if ff_ratio > 0.60:
        print(f"⚠️ [Quality Gate Warning] {ff_ratio*100:.1f}% fast-forward, but protected action is preserved. Passing.")

    return True, "Passed"

def adapt_edf_v2_to_v1(v2_blueprint: dict) -> dict:
    if not v2_blueprint or "segments" in v2_blueprint:
        return v2_blueprint
        
    v1_segments = []
    for idx, seg in enumerate(v2_blueprint.get("editing_plan", [])):
        playback = seg.get("playback_action", {})
        v1_segments.append({
            "segment_id": f"seg_{idx}",
            "start_timestamp": float(seg.get("start", 0.0)),
            "end_timestamp": float(seg.get("end", 0.0)),
            "editing_intent": {
                "speed_multiplier": float(playback.get("speed", 1.0)),
                "action_type": playback.get("type", "standard")
            }
        })
        
    return {
        "edf_version": "1.4-adapted",
        "clip_bounds": v2_blueprint.get("clip_bounds", {}),
        "segments": v1_segments,
        "meme_candidates": v2_blueprint.get("meme_candidates", [])
    }

# ==========================================
# THE DYNAMIC ROUTER (INTAKE MENU)
# ==========================================
def setup_video_source(input_dir: str = "data/inputs") -> List[str]:
    os.makedirs(input_dir, exist_ok=True)
    print("\n" + "="*40)
    print("🎬 SELECT VIDEO SOURCE")
    print("="*40)
    print("1. Download from YouTube (Live Stream / VOD)")
    print("2. Select from Local Folder")
    # 👇 ADDED OPTION 3 👇
    print("3. Download VOD, Chop into Matches, & Process")
    print("="*40)
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        url = input("\n🔗 Enter YouTube URL: ").strip()
        downloaded_file = fetch_youtube_video(url, input_dir)
        return [downloaded_file] if downloaded_file else []
        
    elif choice == "2":
        found_videos = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.mkv', '.mov'))]
        if not found_videos:
            print(f"\n⚠️ No video files found in '{input_dir}'.")
            return []
            
        print("\n📂 AVAILABLE LOCAL VIDEOS:")
        for idx, v_file in enumerate(found_videos):
            print(f"{idx + 1}. {v_file}")
        print(f"{len(found_videos) + 1}. Process ALL videos in folder")
        
        v_choice = input(f"\nSelect a video (1-{len(found_videos) + 1}): ").strip()
        try:
            v_idx = int(v_choice) - 1
            if v_idx == len(found_videos):
                return [os.path.join(input_dir, f) for f in found_videos]
            elif 0 <= v_idx < len(found_videos):
                selected_path = os.path.join(input_dir, found_videos[v_idx])
                print(f"\n✅ Selected: {found_videos[v_idx]}")
                return [selected_path]
            else:
                return []
        except ValueError:
            return []

    # 👇 ADDED CHOPPER LOGIC 👇
    elif choice == "3":
        url = input("\n🔗 Enter 3-Hour VOD YouTube/Twitch URL: ").strip()
        downloaded_file = fetch_youtube_video(url, input_dir)
        
        if downloaded_file:
            print("\n✂️ Initializing VOD Chopper...")
            chopper = VODChopper()
            matches = chopper.scan_vod(downloaded_file)
            
            if matches:
                chopper.slice_vod(downloaded_file, matches)
                vod_name = os.path.splitext(os.path.basename(downloaded_file))[0]
                sliced_videos = []
                
                # Collect the chopped segments from the chopper's output directory
                for i in range(1, len(matches) + 1):
                    sliced_path = os.path.join(chopper.output_dir, f"{vod_name}_match_{str(i).zfill(2)}.mp4")
                    if os.path.exists(sliced_path):
                        sliced_videos.append(sliced_path)
                
                print(f"\n✅ Sending {len(sliced_videos)} extracted matches to the Factory!")
                return sliced_videos
            else:
                print("⚠️ No matches found in the downloaded VOD.")
                return []
        else:
            print("❌ Failed to download VOD.")
            return []

    return []

# ==========================================
# THE AUTONOMOUS FACTORY LOOP
# ==========================================
def run_factory_pipeline():
    logger = PipelineLogger()
    print(f"\n=== Starting Autonomous Viral Shorts Factory [Run ID: {logger.run_id}] ===")
    logger.log_event("factory_boot")
    
    db = FactoryStateManager()
    selected_videos = setup_video_source()
    if not selected_videos:
        print("Factory resting. Goodbye!")
        return
        
    pending_videos = selected_videos

    # Initialize Heavy Engines
    temporal_memory = TemporalMemoryEngine(persist_directory="data/chroma_db")
    meme_memory = MemeManager(meme_dir="assets/memes")
    audio_memory = AudioMemoryIndex(client_instance=temporal_memory.client)
    renderer = AdvancedVideoRenderingEngine(output_dir="data/renders")

    for raw_video_path in pending_videos:
        print(f"\n--- [Queue] Processing source: {raw_video_path} ---")
        db.update_job_status(raw_video_path, "PROCESSING")
        
        manager = PipelineRecoveryManager(raw_video_path)
        
        try:
            # ---------------------------------------------------------
            # [Phase 2] The Macro Scout (Executive Director)
            # ---------------------------------------------------------
            blueprints = manager.load_state("macro_scout_output")
            
            if not blueprints:
                candidates_file = "data/inputs/candidate_windows.json"
                if not os.path.exists(candidates_file):
                    print(f"[Factory] Missing {candidates_file}. Run signal_fusion.py first!")
                    continue
                    
                print(f"[Phase 2] Handing candidate windows to the Macro Scout...")
                manager.log_event("macro_scout_start")
                blueprints = run_macro_scout(candidates_file, OPENAI_API_KEY)
                
                if not blueprints or not blueprints.get("selected_clips"):
                    print("[Factory] The Macro Scout rejected all clips. Skipping video.")
                    manager.log_event("macro_scout_reject_all")
                    continue
                manager.save_state("macro_scout_output", blueprints)

            best_clip_windows = blueprints.get("selected_clips", [])
            print(f"[Factory] Locked in {len(best_clip_windows)} viral narrative arcs!")

            # ---------------------------------------------------------
            # [Phase 3 & 4 & 5] The Autonomous Production Loop
            # ---------------------------------------------------------
            for idx, clip in enumerate(best_clip_windows):
                start_time = float(clip.get("start_time", 0.0))
                end_time = float(clip.get("end_time", 0.0))
                
                print(f"\n--- Orchestrating Event Block #{idx + 1} ({start_time}s to {end_time}s) ---")
                
                try: 
                    # ==========================================
                    # THE QUALITY GATE & STRICT RETRY LOOP
                    # ==========================================
                    is_valid = False
                    edf_data = None
                    
                    # Check cache first
                    cached_qwen = manager.load_state(f"qwen_semantics_clip_{idx}")
                    cached_edf = manager.load_state(f"master_edf_clip_{idx}")
                    
                    if cached_qwen and cached_edf:
                        print("♻️ [Recovery] Found cached EDF and Qwen semantics. Evaluating...")
                        is_valid, reason = enforce_safety_and_evaluate(cached_edf.get("editing_plan", []))
                        if is_valid:
                            edf_data = cached_edf
                        else:
                            print(f"⚠️ Cached EDF failed quality gate: {reason}. Forcing regeneration...")
                    
                    # Generate/Regenerate if needed
                    if not is_valid:
                        for attempt in range(2):
                            strict_flag = (attempt == 1) # True on the second attempt (the retry)
                            
                            if strict_flag:
                                print("♻️ Retrying Qwen prompt with STRICT hook constraints...")
                            else:
                                print(f"[Phase 3] Booting Micro Director (Attempt 1)...")
                            
                            try:
                                # Try passing strict mode to your Qwen analyzer
                                qwen_pacing_data = analyze_clip_pacing(
                                    raw_video_path, start_time, end_time, 
                                    QWEN_API_KEY, QWEN_PACING_URL, 
                                    strict_mode=strict_flag
                                )
                            except TypeError:
                                # Fallback if analyze_clip_pacing doesn't accept strict_mode kwarg yet
                                qwen_pacing_data = analyze_clip_pacing(
                                    raw_video_path, start_time, end_time, 
                                    QWEN_API_KEY, QWEN_PACING_URL
                                )
                            
                            if not qwen_pacing_data:
                                raise ValueError("Micro Director returned empty data.")

                            print(f"[Phase 4] Executing Master EDF Compiler V2...")
                            edf_data = generate_master_edf(
                                qwen_analysis=qwen_pacing_data, 
                                clip_start=start_time, 
                                clip_end=end_time, 
                                run_dir=manager.state_dir
                            )
                            
                            if not edf_data:
                                print("❌ [Phase 4] Failed to compile Master EDF. Skipping clip.")
                                break
                            
                            # 👇 THE MEANING-AWARE QUALITY GATE 👇
                            is_valid, reason = enforce_safety_and_evaluate(edf_data.get("editing_plan", []))
                            
                            if is_valid:
                                manager.save_state(f"qwen_semantics_clip_{idx}", qwen_pacing_data)
                                manager.save_state(f"master_edf_clip_{idx}", edf_data)
                                print("✅ Quality Gate Passed. Dispatching to FFmpeg...")
                                break
                            else:
                                print(f"🚫 Quality Gate Triggered: {reason}")
                                if attempt == 0:
                                    print("🧹 Flushing local VRAM before retrying with stricter constraints...")
                                    del edf_data
                                    del qwen_pacing_data
                                    flush_vram()
                                else:
                                    print(f"❌ Clip failed Quality Gate twice. Skipping render.")
                                    edf_data = None
                                    flush_vram()
                    
                    # If it failed twice, skip this clip and go to the next one
                    if not edf_data or not is_valid:
                        continue

                    

                    # --- PHASE 5: Fetch Memes/Audio [PAUSED FOR SPRINT A] ---
                    # Meme and BGM fetching are disabled while we validate Pacing Architecture

                    # --- PHASE 6: The Renderer (Dual Output) ---
                    shadow_filename = f"shadow_debug_clip_{idx + 1}.mp4"
                    production_filename = f"viral_short_clip_{idx + 1}.mp4"
                    
                    shadow_path = os.path.join(renderer.output_dir, shadow_filename)
                    prod_path = os.path.join(renderer.output_dir, production_filename)
                    
                    edf_file_path = os.path.join(manager.state_dir, f"master_edf_clip_{idx}.json")

                    print("\n[Phase 6] Dispatching EDF blueprint to FFmpeg Renderer...")
                    
                    # 1. GENERATE SHADOW DEBUG (Visual Observability)
                    if not os.path.exists(shadow_path):
                        renderer.render_shadow_debug(
                            input_path=raw_video_path,
                            edf_json_path=edf_file_path,
                            output_filename=shadow_filename
                        )
                    else:
                        print(f"♻️ [Recovery] Found existing shadow render: {shadow_filename}")

                    # 2. GENERATE SPRINT A PACING RENDER (Execution Validation)
                    if not os.path.exists(prod_path):
                        print("[Phase 6] Normalizing EDF Schema via Adapter Layer...")
                        normalized_edf = adapt_edf_v2_to_v1(edf_data)
                        
                        segments_count = len(normalized_edf.get("segments", []))
                        print(f"📊 [Debug] Adapter output segments: {segments_count}")
                        
                        if segments_count == 0:
                            print("❌ [Fatal] Adapter failed to generate valid segments. Halting render.")
                            continue

                        print("[Phase 6] Dispatching EDF blueprint to FFmpeg Renderer...")
                        renderer.render_dynamic_short(
                            input_path=raw_video_path,
                            output_filename=production_filename,
                            edf_blueprint=normalized_edf
                        )
                    else:
                        print(f"♻️ [Recovery] Found existing production render: {production_filename}")

                    manager.log_event("ffmpeg_render_success", {"clip_index": idx})
                    print(f"--- Event Block #{idx + 1} Completed Successfully ---")

                except Exception as clip_e:
                    print(f"⚠️ [Clip Error] Failed on Event Block #{idx + 1}: {clip_e}")
                    manager.log_event("clip_failure", {"clip_index": idx, "reason": str(clip_e)})
                    continue 

            db.update_job_status(raw_video_path, "COMPLETED")
            print(f"--- Job Finished. Output and logs secured in {manager.state_dir} ---")
            
        except Exception as e:
            print(f"[Queue Manager] Fatal failure on {raw_video_path}: {e}")
            manager.log_event("fatal_error", {"reason": str(e), "file": raw_video_path})
            db.update_job_status(raw_video_path, "FAILED")
            continue

if __name__ == "__main__":
    run_factory_pipeline()