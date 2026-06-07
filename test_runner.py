import json
from src.creative.creative_validator import CreativeValidator

def run_test():
    print("🚀 Initializing Creative Validator...")
    validator = CreativeValidator()

    # 1. Mock inputs exactly as they would come from your pipeline
    clip_id = "test_clip_001"
    paced_video_path = "test_base_video.mp4" # REPLACE WITH A REAL LOCAL PATH

    story_contract = {
        "clip_id": clip_id,
        "timeline_basis": "PACED_CLIP",
        "reaction_window": {"start": 2.0, "end": 6.0},
        "protected_windows": [
            {"start": 6.5, "end": 10.0} # Protected payoff at the end
        ]
    }

    plan_contract = {
        "clip_id": clip_id,
        "creative_decision": {
            "treatment": "REACTION_OVERLAY",
            "fallback_chain": ["TEXT_AND_SOUND", "SOUND_ONLY", "NO_MEME"]
        },
        "placement": {
            "preferred_regions": ["TOP_RIGHT", "TOP_LEFT"],
            "opacity": 0.85
        }
    }

    ranker_contract = {
        "selected_candidate_id": "local_meme_123",
        "selection_confidence": 0.95,
        "reason": "Perfect reaction for the missed skillshot."
    }

    candidate_meta = {
        "candidate_id": "local_meme_123",
        "provider": "local_catalog",
        "full_asset_url": "test_meme.mp4", # REPLACE WITH A REAL LOCAL PATH
        "full_asset_format": "mp4",
        "source_page": "local_storage"
    }

    print("🔍 Running Validation...")
    # 2. Execute the validator
    result = validator.validate_clip(
        clip_id=clip_id,
        paced_video_path=paced_video_path,
        story=story_contract,
        plan=plan_contract,
        ranker=ranker_contract,
        candidate_meta=candidate_meta
    )

    # 3. Print the results
    print("\n✅ Validation Complete!")
    print(f"Validation Passed: {result.get('validation_passed')}")
    print(f"Render Safe:       {result.get('render_safe')}")
    print(f"Final Treatment:   {result.get('final_resolved_treatment')}")
    print("\nValidation Notes:")
    for note in result.get('validation_notes', []):
        print(f"  - {note}")
    
    print("\nFallback Steps Taken:")
    for step in result.get('fallback_steps_taken', []):
        print(f"  - {step}")

    print(f"\nRender Plan saved to: data/creative/render_plans/{clip_id}_render_plan.json")

if __name__ == "__main__":
    run_test()