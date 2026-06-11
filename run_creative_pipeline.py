import os
import json
import traceback
from dotenv import load_dotenv

from src.creative.story_analyst import StoryAnalyst
from src.creative.meme_director import MemeDirector
from src.creative.meme_fetcher import MemeFetcher
from src.creative.contact_sheet_ranker import ContactSheetRanker
from src.creative.creative_validator import CreativeValidator
from src.creative.creative_renderer import CreativeRenderer
from src.creative.music_director import MusicDirector  # 👇 ADDED IMPORT

load_dotenv()

# 👇 NEW BATCH MODE CONSTANTS 👇
SHORTLISTED_CLIPS_DIR = "data/creative/shortlisted_clips"

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm"
}

def adjust_late_reaction_window_for_shorts(story_contract: dict) -> dict:
    """
    If Qwen places the meme reaction too close to the end of the clip,
    shift it slightly earlier so viewers can actually see it.

    This correction only applies when the reaction window is near the clip ending.
    """

    def sf(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    if not isinstance(story_contract, dict):
        return story_contract

    clip_dur = sf(story_contract.get("clip_duration"), 0.0)
    rw = story_contract.get("reaction_window", {})

    if clip_dur <= 0 or not isinstance(rw, dict):
        return story_contract

    start_t = sf(rw.get("start"), 0.0)
    end_t = sf(rw.get("end"), clip_dur)

    # Normalize first
    start_t = max(0.0, min(start_t, clip_dur))
    end_t = max(start_t, min(end_t, clip_dur))

    # Main rule:
    # If meme starts in the last 1 second, shift it 0.5 sec earlier.
    starts_too_late = (clip_dur - start_t) <= 1.0

    if starts_too_late:
        shift_back = 0.5
        start_t = max(0.0, start_t - shift_back)
        end_t = max(start_t + 0.5, end_t - shift_back)
        end_t = min(end_t, clip_dur)

    # Retention rule:
    # If the meme is still too short near the ending, give it at least 1.5 sec.
    min_visible_meme_time = 1.5

    if starts_too_late and (end_t - start_t) < min_visible_meme_time:
        end_t = min(clip_dur, start_t + min_visible_meme_time)

        # If we cannot extend forward, shift earlier.
        if (end_t - start_t) < min_visible_meme_time:
            start_t = max(0.0, end_t - min_visible_meme_time)

    story_contract["reaction_window"] = {
        "start": round(start_t, 2),
        "end": round(end_t, 2)
    }

    # Keep payoff before or equal to reaction start
    payoff = sf(story_contract.get("payoff_timestamp"), start_t)
    story_contract["payoff_timestamp"] = round(min(payoff, start_t), 2)

    return story_contract

def create_safe_story_fallback(clip_id: str, video_path: str) -> dict:
    return {
        "schema_version": "1.0",
        "clip_id": clip_id,
        "timeline_basis": "PACED_CLIP",
        "clip_duration": 5.0,
        "scene_description": "Fallback scene due to API failure.",
        "viewer_expectation": "Fallback expectation.",
        "actual_outcome": "Fallback outcome.",
        "comedy_mechanism": "none",
        "payoff_timestamp": 0.0,
        "reaction_window": {"start": 0.0, "end": 2.0},
        "protected_windows": [],
        "confidence": 0.0
    }

# 👇 NEW BATCH MODE HELPERS 👇
def sanitize_clip_id(filename: str) -> str:
    """
    Converts a video filename into a safe clip_id.
    Example: 'My Clip 01.mp4' -> 'my_clip_01'
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    safe = "".join(c.lower() if c.isalnum() else "_" for c in name)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "clip"

def list_shortlisted_clips(folder_path: str) -> list:
    """
    Returns all supported video files from the shortlisted clips folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return []

    clips = []

    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()

        if os.path.isfile(full_path) and ext in SUPPORTED_VIDEO_EXTENSIONS:
            clips.append(full_path)

    return sorted(clips)

def run_all_shortlisted_clips(folder_path: str = SHORTLISTED_CLIPS_DIR):
    """
    Runs the creative pipeline on every shortlisted clip in the folder.
    """
    clips = list_shortlisted_clips(folder_path)

    if not clips:
        print(f"❌ No clips found in: {folder_path}")
        print("Copy your shortlisted clips into this folder and rerun.")
        return

    print(f"📂 Found {len(clips)} shortlisted clips in: {folder_path}")

    batch_results = []

    for index, clip_path in enumerate(clips, start=1):
        clip_id = sanitize_clip_id(clip_path)

        print("\n" + "=" * 80)
        print(f"🎬 Processing clip {index}/{len(clips)}")
        print(f"   Clip ID: {clip_id}")
        print(f"   Path: {clip_path}")
        print("=" * 80)

        try:
            result = run_single_clip(
                clip_id=clip_id,
                paced_video_path=clip_path
            )
            batch_results.append({
                "clip_id": clip_id,
                "input_video": clip_path,
                "success": bool(result.get("success")) if isinstance(result, dict) else False,
                "final_output": result.get("final_output") if isinstance(result, dict) else None
            })
        except Exception as e:
            print(f"❌ Batch item failed for {clip_id}: {e}")
            traceback.print_exc()
            batch_results.append({
                "clip_id": clip_id,
                "input_video": clip_path,
                "success": False,
                "error": str(e)
            })

    os.makedirs("data/creative/e2e_runs", exist_ok=True)

    batch_summary_path = "data/creative/e2e_runs/batch_summary.json"
    with open(batch_summary_path, "w") as f:
        json.dump(batch_results, f, indent=2)

    print("\n" + "=" * 80)
    print("📦 Batch completed.")
    print(f"📁 Batch summary saved to: {batch_summary_path}")
    print("=" * 80)

# 👇 UPDATED SIGNATURE 👇
def run_single_clip(
    clip_id: str = "e2e_test_001",
    paced_video_path: str = "test_base_video.mp4"
):
    print("🚀 Booting Fault-Tolerant E2E Creative Pipeline...")
    
    meme_asset = "test_meme.mp4"
    
    if not os.path.exists(paced_video_path):
        return print(f"❌ FATAL: Paced video not found at '{paced_video_path}'.")

    qwen_api_key = os.getenv("QWEN_API_KEY")
    # Grab the same URL you used in the pacing analyzer
    qwen_base_url = os.getenv("QWEN_PACING_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1") 
    
    if not qwen_api_key:
        return print("❌ FATAL: QWEN_API_KEY is missing from .env")

    os.makedirs("data/creative/e2e_runs", exist_ok=True)
    os.makedirs("data/creative/render_plans", exist_ok=True)

    catalog = [{
        "asset_id": "real_test_meme",
        "title": "Sad Violin Fail",
        "tags": ["sad", "violin", "fail", "reaction", "missed", "target"],
        "file_path": meme_asset,
        "preview_path": meme_asset,
        "format": "mp4"
    }]
    os.makedirs("assets/memes/approved", exist_ok=True)
    with open("assets/memes/approved/local_meme_catalog.json", "w") as f:
        json.dump(catalog, f)

    # 👇 Initialize Director with API Key and Base URL matching your custom fetcher 👇
    # 1. Analyst: Pass the base_url so it hits Colab, not Aliyun!
    analyst = StoryAnalyst(api_key=qwen_api_key, base_url=qwen_base_url)
    analyst.model_name = "qwen-director" # Override the hardcoded model name
    
    # 2. Director: Update the model name to match your Colab server
    director = MemeDirector(api_key=qwen_api_key, base_url=qwen_base_url, model_name="qwen-director") 
    
    # 3. Fetcher
    # 3. Fetcher: Grab the real key from your .env file
    giphy_key = os.getenv("GIPHY_API_KEY")
    fetcher = MemeFetcher(giphy_api_key=giphy_key)
    
    # 4. Ranker: Update the model name here too
    ranker = ContactSheetRanker(api_key=qwen_api_key, base_url=qwen_base_url, model_name="qwen-director")
    validator = CreativeValidator()
    renderer = CreativeRenderer()
    music_director = MusicDirector()  # 👇 ADDED INITIALIZATION

    # --- 1. Story Analyst ---
    print("\n[Stage 1: Analyst]")
    try:
        story_contract = analyst.analyze_clip(clip_id, paced_video_path)
        story_contract = adjust_late_reaction_window_for_shorts(story_contract)
        print("   Final reaction_window:", story_contract.get("reaction_window"))
        if not story_contract: raise ValueError("Analyst returned empty contract.")
    except Exception as e:
        print(f"⚠️ Analyst failed ({e}). Using safe fallback story.")
        story_contract = create_safe_story_fallback(clip_id, paced_video_path)
        story_contract = adjust_late_reaction_window_for_shorts(story_contract)
        print("   Final reaction_window:", story_contract.get("reaction_window"))

    # 👇 ADDED STAGE 1.5 BLOCK 👇
    # --- 1.5 Music Director ---
    print("\n[Stage 1.5: Music Director]")
    try:
        background_music_plan = music_director.generate_plan(story_contract)
        print("   Background Music:", json.dumps(background_music_plan, indent=2))
    except Exception as e:
        print(f"⚠️ Music Director failed ({e}). Disabling background music.")
        background_music_plan = {
            "enabled": False,
            "music_mood": "neutral",
            "music_path": None,
            "skip_reason": f"Music Director crashed: {str(e)}"
        }
    # 👆 END STAGE 1.5 BLOCK 👆

    # --- 2. Meme Director ---
    print("\n[Stage 2: Director]")
    try:
        plan_contract = director.generate_plan(clip_id, story_contract)
    except Exception as e:
        print(f"❌ FATAL: Director failed to generate fallback plan: {e}")
        return

    treatment = plan_contract.get("creative_decision", {}).get("treatment", "NO_MEME")
    meme_needed = plan_contract.get("creative_decision", {}).get("meme_needed", False)
    fallback_chain = plan_contract.get("creative_decision", {}).get("fallback_chain", ["TEXT_AND_SOUND", "SOUND_ONLY", "NO_MEME"])
    queries = plan_contract.get("creative_decision", {}).get("search_queries", [])
    print(f"   Decided Treatment: {treatment}")

    # --- 3. Meme Fetcher ---
    print("\n[Stage 3: Fetcher]")
    candidates = []
    if meme_needed and queries:
        try:
            candidates = fetcher.retrieve_candidates(queries, max_total=4)
            print(f"   Found {len(candidates)} candidates.")
        except Exception as e:
            print(f"⚠️ Fetcher failed ({e}). Proceeding with 0 candidates.")
    else:
        print("   Skipping Fetcher (meme_needed is False or queries empty).")

    # --- 4. Contact Sheet Ranker ---
    print("\n[Stage 4: Ranker]")
    ranker_contract = {
        "selected_candidate_id": None,
        "selection_confidence": 0.0,
        "none_suitable": True,
        "reason": "No meme candidates returned or Ranker skipped.",
        "fallback_chain": fallback_chain
    }
    
    if candidates and treatment != "NO_MEME":
        try:
            # 👇 FIX: Build the visual contact sheet first, then pass it to the Ranker 👇
            b64_img, mapping, metadata_list = ranker.build_contact_sheet(clip_id, candidates)
            
            ranker_result = ranker.rank_candidates(
                clip_id=clip_id, 
                b64_img=b64_img,
                mapping=mapping,
                metadata_list=metadata_list,
                story=story_contract, 
                plan=plan_contract
            )
            if ranker_result: 
                ranker_contract = ranker_result
                print(f"   Ranker Output: {ranker_contract.get('selected_candidate_id', 'NONE')}")
        except Exception as e:
            print(f"⚠️ Ranker failed ({e}). Triggering fallback.")
            ranker_contract["reason"] = f"Ranker API crashed: {str(e)}"
    else:
        print("   Skipping Ranker.")

    # --- Candidate Resolution & Normalization ---
    winning_id = ranker_contract.get("selected_candidate_id")
    winning_meta = {}
    
    if winning_id:
        winning_meta = next((c for c in candidates if c.get("candidate_id") == winning_id), {})
    
    if winning_meta:
        fmt = winning_meta.get("format", "mp4")
        winning_meta["full_asset_format"] = winning_meta.get("full_asset_format", fmt)
        winning_meta["preview_format"] = winning_meta.get("preview_format", fmt)
        winning_meta.setdefault("candidate_id", winning_id)
        winning_meta.setdefault("full_asset_url", meme_asset)
        winning_meta.setdefault("duration", 2.0)
        winning_meta.setdefault("width", 320)
        winning_meta.setdefault("height", 180)

    # --- 5. Creative Validator ---
    print("\n[Stage 5: Validator]")
    try:
        render_plan = validator.validate_clip(
            clip_id=clip_id,
            paced_video_path=paced_video_path,
            story=story_contract,
            plan=plan_contract,
            ranker=ranker_contract,
            candidate_meta=winning_meta if winning_meta else None
        )
        
        # 👇 ADDED: Inject BGM plan into the final render plan 👇
        render_plan["background_music"] = background_music_plan
        
        render_plan_path = f"data/creative/render_plans/{clip_id}_render_plan.json"
        with open(render_plan_path, 'w') as f:
            json.dump(render_plan, f, indent=2)
    except Exception as e:
        print(f"❌ FATAL: Validator crashed: {e}")
        traceback.print_exc()
        return

    # --- 6. Creative Renderer ---
    print("\n[Stage 6: Renderer]")
    audit = {}
    try:
        audit = renderer.render_clip(render_plan_path, paced_video_path)
        if audit.get("success"):
            print(f"\n🎉 RENDER SUCCESS! Final video saved to: {audit.get('target_output')}")
        else:
            print("\n❌ RENDER FAILED!")
            for log in audit.get("logs", []): print(f"   - {log}")
    except Exception as e:
        print(f"❌ FATAL: Renderer crashed: {e}")
        audit = {"success": False, "logs": [str(e)]}

    # --- 7. Save Summary JSON ---
    summary = {
        "clip_id": clip_id,
        "input_video": paced_video_path,
        "initial_treatment": treatment,
        "candidates_found": len(candidates),
        "ranker_result": ranker_contract,
        "final_render_plan": render_plan,
        "render_audit": audit,
        "success": audit.get("success", False),
        "final_output": audit.get("target_output")
    }
    summary_path = f"data/creative/e2e_runs/{clip_id}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📁 E2E Summary saved to: {summary_path}")
    
    # 👇 ADDED RETURN 👇
    return summary

# 👇 UPDATED RUNNER BLOCK 👇
if __name__ == "__main__":
    print("\nSelect run mode:")
    print("1. Run single test clip")
    print("2. Run all clips from shortlisted folder")

    choice = input("\nEnter choice [1/2]: ").strip()

    if choice == "2":
        run_all_shortlisted_clips(SHORTLISTED_CLIPS_DIR)
    else:
        run_single_clip()