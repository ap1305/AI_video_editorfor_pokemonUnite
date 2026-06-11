import os
import json
from src.creative.meme_director import MemeDirector
from src.creative.meme_fetcher import MemeFetcher

# --- 1. Mock LLM Client (from our previous test) ---
class MockLLMClient:
    def generate_json(self, prompt: str, **kwargs) -> dict:
        return {
            "treatment": "REACTION_OVERLAY",
            "creative_reason": "A massive fail requires a strong visual reaction.",
            "comedy_mechanism": "expectation_reversal",
            "search_queries": ["sad violin guy", "confused travolta gif", "wasted meme"],
            "avoid_concepts": ["celebration", "happy dance"],
            "sound_function": "fail_sting",
            "fallback_text": "Task failed successfully.",
            "timing_intent": "IMMEDIATELY_AFTER_PAYOFF",
            "preferred_regions": ["TOP_RIGHT", "TOP_LEFT"],
            "intensity": "HIGH",
            "confidence": 0.95
        }

# --- 2. Mock Story Contract ---
def get_valid_story():
    return {
        "schema_version": "1.0",
        "clip_id": "integration_test_001",
        "timeline_basis": "PACED_CLIP",
        "clip_duration": 15.0,
        "scene_description": "Player tries to execute a clutch save but completely misses.",
        "viewer_expectation": "The player will save the objective.",
        "actual_outcome": "The player hits an obstacle and gets knocked out.",
        "comedy_mechanism": "slapstick_physics",
        "payoff_timestamp": 10.5,
        "reaction_window": {"start": 10.5, "end": 14.5},
        "protected_windows": [],
        "confidence": 0.85
    }

def run_integration_test():
    print("🎬 Initializing Director -> Fetcher Integration Test...\n")
    
    # Initialize Modules
    director = MemeDirector(client=MockLLMClient(), model_name="mock-model")
    
    # Note: Using "MISSING" forces the fetcher into safe local-only mode 
    # to avoid hitting GIPHY during our unit tests.
    fetcher = MemeFetcher(giphy_api_key="MISSING")

    # Step 1: Director Generates the Plan
    print("1️⃣ Running Meme Director...")
    story = get_valid_story()
    meme_plan = director.generate_plan(story["clip_id"], story)
    
    treatment = meme_plan['creative_decision']['treatment']
    meme_needed = meme_plan['creative_decision']['meme_needed']
    search_queries = meme_plan['creative_decision']['search_queries']
    
    print(f"   Treatment: {treatment}")
    print(f"   Meme Needed: {meme_needed}")
    print(f"   Generated Queries: {search_queries}\n")

    # Step 2: Safety Check (Ensure we only fetch if the contract says so)
    if not meme_needed:
        print("⏭️ Meme not needed for this treatment. Fetcher bypassed. (Integration successful)")
        return

    # Step 3: Fetcher Retrieves Candidates
    print("2️⃣ Running Meme Fetcher...")
    candidates = fetcher.retrieve_candidates(search_queries=search_queries, max_total=4)
    
    print(f"   Candidates Retrieved: {len(candidates)}")
    for i, candidate in enumerate(candidates, 1):
        print(f"   [{i}] {candidate['candidate_id']} (Origin: {candidate['originating_search_query']})")

    print("\n✅ Integration Test Complete!")

if __name__ == "__main__":
    # Create dummy local catalog to test local retrieval logic
    os.makedirs("assets/memes/approved", exist_ok=True)
    dummy_catalog = [
        {
            "asset_id": "dummy_sad_violin",
            "title": "Sad Violin Guy",
            "tags": ["sad", "violin", "fail"],
            "file_path": "dummy.mp4",
            "preview_path": "dummy.mp4"
        }
    ]
    with open("assets/memes/approved/local_meme_catalog.json", "w") as f:
        json.dump(dummy_catalog, f)
        
    # Touch a dummy file to pass the os.path.isfile check
    with open("dummy.mp4", "w") as f:
        f.write("dummy content")

    try:
        run_integration_test()
    finally:
        # Cleanup
        if os.path.exists("dummy.mp4"): os.remove("dummy.mp4")