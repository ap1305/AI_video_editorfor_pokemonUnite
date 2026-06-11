import os
import shutil
import uuid
import subprocess
from src.preprocess.vod_moment_scanner import VODMomentScanner
from src.preprocess.gameplay_verifier import GameplayVerifier
from src.preprocess.verified_clip_cutter import VerifiedClipCutter

# 👇 Bringing in the tool we built specifically for your actual goal
from src.preprocess.match_extractor import MatchExtractor

def get_json_file(directory: str, prefix: str):
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".json")]
    if not files: return None
    if len(files) == 1: return os.path.join(directory, files[0])
    
    print("\n📂 Select File:")
    for idx, f in enumerate(files, 1): print(f"  {idx}. {f}")
    choice = int(input(f"Choice [1-{len(files)}]: ")) - 1
    return os.path.join(directory, files[choice])

def run_menu():
    raw_dir = "data/raw_vods"
    inputs_dir = "data/inputs"
    shortlisted_dir = "data/creative/shortlisted_clips"
    verified_dir = "data/vod_verified_candidates"
    gameplay_dir = "data/gameplay_segments" # For the full matches
    
    for d in [raw_dir, inputs_dir, shortlisted_dir, verified_dir, gameplay_dir]:
        os.makedirs(d, exist_ok=True)
    
    while True:
        print("\n" + "="*50)
        print("🎬 CRUSHER11 VOD STUDIO")
        print("="*50)
        print("--- GOAL A: SHORT VIRAL MOMENTS (30s) ---")
        print("1. Scan VOD for Action Peaks")
        print("2. Verify Peaks (Vision AI)")
        print("3. Cut Verified Moments")
        print("4. Copy to Shortlisted")
        
        print("\n--- GOAL B: FULL MATCH EXTRACTION (10+ mins) ---")
        print("5. Extract Complete Matches (State Machine)")
        
        print("\n6. Download New VOD")
        print("7. Exit")
        print("="*50)
        
        choice = input("\nEnter choice [1-7]: ").strip()
        
        if choice == "1":
            files = [f for f in os.listdir(raw_dir) if f.lower().endswith((".mp4", ".mkv"))]
            if not files:
                print("❌ No videos found in data/raw_vods/")
                continue
            for idx, f in enumerate(files, 1): print(f"  {idx}. {f}")
            v_idx = int(input("Select video: ")) - 1
            VODMomentScanner().scan_vod(os.path.join(raw_dir, files[v_idx]))
            
        elif choice == "2":
            file = get_json_file(inputs_dir, "candidate_windows_raw_")
            if file: GameplayVerifier().verify_candidates(file)
            
        elif choice == "3":
            file = get_json_file(inputs_dir, "verified_candidate_windows_")
            if file: VerifiedClipCutter().cut_approved_clips(file)
            
        elif choice == "4":
            clips = [f for f in os.listdir(verified_dir) if f.endswith(".mp4")]
            if not clips: continue
            if input(f"Promote {len(clips)} clips to Creative? (y/n): ").lower() == 'y':
                for c in clips: shutil.copy(os.path.join(verified_dir, c), os.path.join(shortlisted_dir, c))
                print(f"✅ Promoted.")

        # 👇 THIS IS WHAT YOU ACTUALLY WANT TO RUN
        elif choice == "5":
            files = [f for f in os.listdir(raw_dir) if f.lower().endswith((".mp4", ".mkv"))]
            if not files:
                print("❌ No videos found in data/raw_vods/")
                continue
            print("\n📂 Available Videos for Full Match Extraction:")
            for idx, f in enumerate(files, 1): print(f"  {idx}. {f}")
            v_idx = int(input("Select video: ")) - 1
            
            target_vod = os.path.join(raw_dir, files[v_idx])
            MatchExtractor().extract_matches(target_vod)

        elif choice == "6":
            url = input("\n🔗 Enter YouTube/Twitch URL: ").strip()
            safe_name = f"vod_download_{uuid.uuid4().hex[:6]}.mp4"
            target = os.path.join(raw_dir, safe_name)
            subprocess.run(["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best", "--merge-output-format", "mp4", "-o", target, url])

        elif choice == "7":
            print("🛑 Exiting.")
            break

if __name__ == "__main__":
    run_menu()