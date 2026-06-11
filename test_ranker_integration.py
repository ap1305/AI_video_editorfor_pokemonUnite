import os
import json
from dotenv import load_dotenv

from src.creative.meme_director import MemeDirector
from src.creative.meme_fetcher import MemeFetcher
from src.creative.creative_validator import CreativeValidator
giphy_api_key = os.getenv("GIPHY_API_KEY")
load_dotenv()

# --- Mock LLM Client for Director ---
class MockDirectorClient:
    def generate_json(self, prompt: str, **kwargs) -> dict:
        return {
            "treatment": "REACTION_OVERLAY",
            "creative_reason": "Validator Integration Test.",
            "comedy_mechanism": "slapstick_physics",
            "search_queries": ["funny reaction"],
            "avoid_concepts": [],
            "sound_function": "none",
            "fallback_text": "",
            "timing_intent": "IMMEDIATELY_AFTER_PAYOFF",
            "preferred_regions": ["TOP_LEFT"],
            "intensity": "MEDIUM",
            "confidence": 0.90
        }

def get_valid_story():
    return {
        "schema_version": "1.0",
        "clip_id": "validator_test_001",
        "timeline_basis": "PACED_CLIP",
        "clip_duration": 5.0,
        "scene_description": "Player misses.",
        "viewer_expectation": "Player scores.",
        "actual_outcome": "Player fails.",
        "comedy_mechanism": "expectation_reversal",
        "payoff_timestamp": 2.0,
        "reaction_window": {"start": 2.0, "end": 4.0},
        "protected_windows": [],
        "confidence": 0.85
    }

def run_validator_integration():
    print("🎬 Initializing Director -> Fetcher -> Validator Test...\n")
    
    # Real test assets
    base_video = "test_base_video.mp4"
    meme_asset = "test_meme.mp4"

    if not os.path.exists(base_video) or not os.path.exists(meme_asset):
        print("❌ FATAL: Missing real test video files.")
        return

    # Initialize Modules
    director = MemeDirector(client=MockDirectorClient(), model_name="mock")
    fetcher = MemeFetcher(giphy_api_key=giphy_api_key)
    validator = CreativeValidator()

    # 1. Director
    story = get_valid_story()
    meme_plan = director.generate_plan(story["clip_id"], story)
    
    # 2. Fetcher (Injecting the real test_meme.mp4)
    catalog = [{
        "asset_id": "real_test_meme",
        "title": "Real Test Meme",
        "tags": ["funny", "reaction"],
        "file_path": meme_asset,
        "preview_path": meme_asset
    }]
    os.makedirs("assets/memes/approved", exist_ok=True)
    with open("assets/memes/approved/local_meme_catalog.json", "w") as f:
        json.dump(catalog, f)
        
    candidates = fetcher.retrieve_candidates(meme_plan['creative_decision']['search_queries'], max_total=1)
    
    if not candidates:
        print("❌ Fetcher failed to retrieve the local test meme.")
        return

    # 3. Mock Ranker Output
    print("\n⏭️ Skipping Ranker Execution (Injecting Mock Ranker Contract)...")
    ranker_contract = {
        "selected_candidate_id": candidates[0]["candidate_id"],
        "selection_confidence": 0.95,
        "reason": "Perfect reaction."
    }
    winning_meta = candidates[0]

    # 4. Validator
    print("\n🔍 Running Creative Validator...")
    render_plan = validator.validate_clip(
        clip_id=story["clip_id"],
        paced_video_path=base_video,
        story=story,
        plan=meme_plan,
        ranker=ranker_contract,
        candidate_meta=winning_meta
    )
    
    print("\n✅ Validation Complete!")
    print(f"Validation Passed: {render_plan.get('validation_passed')}")
    print(f"Final Treatment:   {render_plan.get('final_resolved_treatment')}")
    print(f"Render Safe:       {render_plan.get('render_safe')}")
    
    print("\nValidation Notes:")
    for note in render_plan.get('validation_notes', []):
        print(f"  - {note}")
        
    plan_path = f"data/creative/render_plans/{story['clip_id']}_render_plan.json"
    print(f"\n📁 Final Render Plan saved to: {plan_path}")

if __name__ == "__main__":
    run_validator_integration()