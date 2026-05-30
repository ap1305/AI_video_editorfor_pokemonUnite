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
from src.utils.gpt_edf_director import generate_edit_decision_file
from src.utils.pipeline_recovery import PipelineRecoveryManager
# Add this near your other from src... imports
from src.editing.edf_compiler import generate_master_edf

load_dotenv()
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "ollama")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QWEN_PACING_URL = os.getenv("QWEN_PACING_URL")   

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
                reasoning = clip.get("narrative_reasoning", "")
                
                print(f"\n--- Orchestrating Event Block #{idx + 1} ({start_time}s to {end_time}s) ---")
                
                # 👉 FIX 1: CLIP-LEVEL EXCEPTION HANDLING
                try: 
                    # --- PHASE 3: The Micro Director ---
                    qwen_pacing_data = manager.load_state(f"qwen_semantics_clip_{idx}")
                    if not qwen_pacing_data:
                        print("[Phase 3] Booting Micro Director...")
                        qwen_pacing_data = analyze_clip_pacing(raw_video_path, start_time, end_time, QWEN_API_KEY, QWEN_PACING_URL)
                        
                        if not qwen_pacing_data:
                            raise ValueError("Micro Director returned empty data.")
                        manager.save_state(f"qwen_semantics_clip_{idx}", qwen_pacing_data)

                   # ==========================================
                    # --- PHASE 4: Master EDF Compiler V2 ---
                    # ==========================================
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
                    # ==========================================

                    # --- PHASE 5: Fetch Memes/Audio ---
                    meme_candidates = edf_data.get("meme_candidates", [])
                    # 👉 FIX 3: EXPLICIT FALLBACK LOGGING
                    if not meme_candidates:
                        manager.log_event("meme_fallback_used", {"clip_index": idx})
                    final_vibe = meme_candidates[0].get("type", "hype") if meme_candidates else "hype"
                    
                    matched_meme_path = meme_memory.fetch_matching_meme(vibe_tag=final_vibe)
                    bgm_path = audio_memory.fetch_matching_audio(vibe_tag=final_vibe)

                    # --- PHASE 6: The Renderer ---
                    output_filename = f"viral_short_clip_{idx + 1}.mp4"
                    output_path = os.path.join(renderer.output_dir, output_filename)
                    
                    # 👉 FIX 4: IDEMPOTENCY CHECK ON VIDEO FILES
                    if os.path.exists(output_path):
                        print(f"♻️ [Recovery] Found existing render: {output_filename}. Skipping FFmpeg!")
                        continue

                    print("[Phase 6] Dispatching EDF blueprint to FFmpeg Renderer...")
                    manager.log_event("ffmpeg_render_start", {"clip_index": idx})
                    
                    renderer.render_dynamic_short(
                        input_path=raw_video_path,
                        output_filename=output_filename,
                        start_time=start_time,
                        end_time=end_time,
                        edf_blueprint=edf_data, 
                        meme_overlay_path=matched_meme_path,
                        bgm_path=bgm_path 
                    )
                    
                    manager.log_event("ffmpeg_render_success", {"clip_index": idx, "file": output_filename})
                    print(f"--- Event Block #{idx + 1} Completed Successfully ---")

                    # 👉 FIX 5: CLEANUP TEMP WORKSPACE
                    shutil.rmtree(renderer.temp_dir, ignore_errors=True)
                    os.makedirs(renderer.temp_dir, exist_ok=True)

                except Exception as clip_e:
                    # IF A CLIP CRASHES, LOG IT AND MOVE TO THE NEXT CLIP
                    print(f"⚠️ [Clip Error] Failed on Event Block #{idx + 1}: {clip_e}")
                    manager.log_event("clip_failure", {"clip_index": idx, "reason": str(clip_e)})
                    continue 

            db.update_job_status(raw_video_path, "COMPLETED")
            print(f"--- Job Finished. Output and logs secured in {manager.state_dir} ---")
            
        except Exception as e:
            # THIS ONLY CATCHES FATAL PIPELINE ERRORS (e.g., Macro Scout crashing)
            print(f"[Queue Manager] Fatal failure on {raw_video_path}: {e}")
            manager.log_event("fatal_error", {"reason": str(e), "file": raw_video_path})
            db.update_job_status(raw_video_path, "FAILED")
            continue

if __name__ == "__main__":
    run_factory_pipeline()