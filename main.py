import os
import json
import shutil
from typing import List
from dotenv import load_dotenv

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

load_dotenv()
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "ollama")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QWEN_PACING_URL = os.getenv("QWEN_PACING_URL")   


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
    print("="*40)
    
    choice = input("Enter choice (1 or 2): ").strip()
    
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
        
        # 1. Boot the Deterministic Recovery Manager
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
                clip_duration = end_time - start_time
                
                print(f"\n--- Orchestrating Event Block #{idx + 1} ({start_time}s to {end_time}s) ---")
                
                try: 
                    # --- PHASE 3: The Micro Director ---
                    qwen_pacing_data = manager.load_state(f"qwen_semantics_clip_{idx}")
                    if not qwen_pacing_data:
                        print("[Phase 3] Booting Micro Director...")
                        qwen_pacing_data = analyze_clip_pacing(raw_video_path, start_time, end_time, QWEN_API_KEY, QWEN_PACING_URL)
                        
                        if not qwen_pacing_data:
                            raise ValueError("Micro Director returned empty data.")
                        manager.save_state(f"qwen_semantics_clip_{idx}", qwen_pacing_data)

                    # --- PHASE 4: Master EDF Compiler V2 ---
                    edf_data = manager.load_state(f"master_edf_clip_{idx}")
                    if not edf_data:
                        print(f"\n[Phase 4] Executing Master EDF Compiler V2...")
                        
                        edf_data = generate_master_edf(
                            qwen_analysis=qwen_pacing_data, 
                            clip_start=start_time, 
                            clip_end=end_time, 
                            run_dir=manager.state_dir
                        )
                        
                        if not edf_data:
                            print("❌ [Phase 4] Failed to compile Master EDF. Skipping clip.")
                            continue
                            
                        manager.save_state(f"master_edf_clip_{idx}", edf_data)

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
                        # 👇 1. Translate to legacy schema
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
                            edf_blueprint=normalized_edf # 👈 2. Pass the adapted data!
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