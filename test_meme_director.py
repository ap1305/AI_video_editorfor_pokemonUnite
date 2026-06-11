import json
import os
from src.creative.meme_director import MemeDirector

# ==========================================
# 1. MOCK LLM CLIENT
# ==========================================
class MockLLMClient:
    def __init__(self, scenario="PERFECT_OVERLAY"):
        self.scenario = scenario

    def generate_json(self, prompt: str, **kwargs) -> dict:
        if self.scenario == "PERFECT_OVERLAY":
            return {
                "treatment": "REACTION_OVERLAY",
                "creative_reason": "A massive fail requires a strong visual reaction.",
                "comedy_mechanism": "expectation_reversal",
                "search_queries": ["sad violin guy", "confused travolta gif", "wasted gta meme"],
                "avoid_concepts": ["celebration", "happy dance"],
                "sound_function": "fail_sting",
                "fallback_text": "Task failed successfully.",
                "timing_intent": "IMMEDIATELY_AFTER_PAYOFF",
                "preferred_regions": ["TOP_RIGHT", "TOP_LEFT"],
                "intensity": "HIGH",
                "confidence": 0.95
            }
        elif self.scenario == "LOW_CONFIDENCE_OVERLAY":
            return {
                "treatment": "REACTION_OVERLAY",
                "creative_reason": "Maybe a meme goes here? Not sure.",
                "comedy_mechanism": "none",
                "search_queries": ["funny clip"],
                "avoid_concepts": [],
                "sound_function": "none",
                "fallback_text": "Oops.",
                "timing_intent": "DURING_REACTION_WINDOW",
                "preferred_regions": ["CENTER_LEFT"],
                "intensity": "LOW",
                "confidence": 0.60  # Below the 0.65 threshold for overlays
            }
        elif self.scenario == "GARBAGE_JSON":
            # Simulating an LLM returning a list instead of a dictionary object
            return ["This", "is", "not", "a", "dictionary"]

# ==========================================
# 2. MOCK STORY CONTRACTS
# ==========================================
def get_valid_story():
    return {
        "schema_version": "1.0",
        "clip_id": "test_clip_001",
        "timeline_basis": "PACED_CLIP",
        "clip_duration": 15.0,
        "scene_description": "Player tries to jump a gap in a car.",
        "viewer_expectation": "The car will land on the other side.",
        "actual_outcome": "The car hits an invisible wall and explodes.",
        "comedy_mechanism": "slapstick_physics",
        "payoff_timestamp": 10.5,
        "reaction_window": {"start": 10.5, "end": 14.5},
        "protected_windows": [],
        "confidence": 0.85
    }

def get_bad_story():
    story = get_valid_story()
    story["timeline_basis"] = "WRONG_BASIS"
    story["payoff_timestamp"] = "not_a_number"
    return story

# ==========================================
# 3. TEST EXECUTION
# ==========================================
def run_tests():
    print("🎬 Initializing Meme Director Test Suite...\n")
    
    # Test 1: The Happy Path
    print("--- TEST 1: The Happy Path (Perfect Overlay) ---")
    director = MemeDirector(client=MockLLMClient("PERFECT_OVERLAY"), model_name="mock-qwen", config={})
    plan1 = director.generate_plan("test_clip_001", get_valid_story())
    print(f"Treatment Selected: {plan1['creative_decision']['treatment']}")
    print(f"Fallback Chain:     {plan1['creative_decision']['fallback_chain']}")
    print(f"Search Queries:     {plan1['creative_decision']['search_queries']}")
    print(f"Warnings:           {plan1['director_metadata']['warnings']}\n")

    # Test 2: LLM Confidence Downgrade
    print("--- TEST 2: LLM Confidence Downgrade ---")
    director = MemeDirector(client=MockLLMClient("LOW_CONFIDENCE_OVERLAY"), model_name="mock-qwen", config={})
    plan2 = director.generate_plan("test_clip_001", get_valid_story())
    print(f"Treatment Selected: {plan2['creative_decision']['treatment']}")
    print(f"Search Queries:     {plan2['creative_decision']['search_queries']}")
    print(f"Warnings:           {plan2['director_metadata']['warnings']}\n")

    # Test 3: LLM Hallucination / Garbage Output
    print("--- TEST 3: LLM Hallucination (Garbage JSON) ---")
    director = MemeDirector(client=MockLLMClient("GARBAGE_JSON"), model_name="mock-qwen", config={})
    plan3 = director.generate_plan("test_clip_001", get_valid_story())
    print(f"Treatment Selected: {plan3['creative_decision']['treatment']}")
    print(f"Fallback Generated: {plan3['director_metadata']['fallback_generated']}")
    print(f"Warnings:           {plan3['director_metadata']['warnings']}\n")

    # Test 4: Upstream Input Failure
    print("--- TEST 4: Upstream Input Failure (Bad Story Contract) ---")
    director = MemeDirector(client=MockLLMClient("PERFECT_OVERLAY"), model_name="mock-qwen", config={})
    plan4 = director.generate_plan("test_clip_001", get_bad_story())
    print(f"Treatment Selected: {plan4['creative_decision']['treatment']}")
    print(f"Fallback Generated: {plan4['director_metadata']['fallback_generated']}")
    print(f"Warnings:           {plan4['director_metadata']['warnings']}\n")

if __name__ == "__main__":
    run_tests()