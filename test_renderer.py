import os
import json
from src.creative.creative_renderer import CreativeRenderer

def generate_test_plans():
    plans_dir = "data/creative/render_plans"
    os.makedirs(plans_dir, exist_ok=True)
    
    # 1. NO_MEME
    no_meme_path = os.path.join(plans_dir, "renderer_test_no_meme_plan.json")
    no_meme_plan = {
      "clip_id": "renderer_test_no_meme",
      "render_safe": True,
      "timeline_basis": "PACED_CLIP",
      "final_resolved_treatment": "NO_MEME",
      "selected_asset": None,
      "render_parameters": {},
      "usage_mark_required_after_render": False
    }
    with open(no_meme_path, 'w') as f:
        json.dump(no_meme_plan, f, indent=2)

    # 2. REACTION_OVERLAY
    overlay_path = os.path.join(plans_dir, "renderer_test_overlay_plan.json")
    overlay_plan = {
      "clip_id": "renderer_test_overlay",
      "render_safe": True,
      "timeline_basis": "PACED_CLIP",
      "final_resolved_treatment": "REACTION_OVERLAY",
      "selected_asset": {
        "candidate_id": "local:test_meme"
      },
      "render_parameters": {
        "asset_path": "test_meme.mp4",
        "start_time": 0.5,
        "end_time": 2.5,
        "opacity": 1.0,
        "loop_policy": "LOOP_TO_INTERVAL",
        "placement": {
          "box": [0.05, 0.10, 0.30, 0.35]
        }
      },
      "usage_mark_required_after_render": True
    }
    with open(overlay_path, 'w') as f:
        json.dump(overlay_plan, f, indent=2)

    return no_meme_path, overlay_path

def run_tests():
    print("🎬 Initializing Creative Renderer Sandbox...")
    
    base_video = "test_base_video.mp4"
    meme_asset = "test_meme.mp4"
    
    if not os.path.exists(base_video):
        print(f"❌ FATAL: Base video not found at '{base_video}'.")
        return
        
    if not os.path.exists(meme_asset):
        print(f"❌ FATAL: Meme asset not found at '{meme_asset}'.")
        print(f"   Run this first: ffmpeg -y -i {base_video} -t 2 -vf \"scale=320:-2\" {meme_asset}")
        return

    no_meme_path, overlay_path = generate_test_plans()
    renderer = CreativeRenderer()

    # --- TEST 1: NO_MEME ---
    print("\n[TEST 1] Executing NO_MEME...")
    audit_no_meme = renderer.render_clip(no_meme_path, base_video)
    if audit_no_meme.get("success"):
        print(f"✅ NO_MEME SUCCESS! -> {audit_no_meme.get('target_output')}")
    else:
        print("❌ NO_MEME FAILED!")
        for log in audit_no_meme.get("logs", []): print(f"  - {log}")

    # --- TEST 2: REACTION_OVERLAY ---
    print("\n[TEST 2] Executing REACTION_OVERLAY...")
    audit_overlay = renderer.render_clip(overlay_path, base_video)
    if audit_overlay.get("success"):
        print(f"✅ REACTION_OVERLAY SUCCESS! -> {audit_overlay.get('target_output')}")
        print(f"   History Updated: {audit_overlay.get('history_updated')}")
    else:
        print("❌ REACTION_OVERLAY FAILED!")
        for log in audit_overlay.get("logs", []): print(f"  - {log}")

if __name__ == "__main__":
    run_tests()