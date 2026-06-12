"""
CRUSHER11 VOD STUDIO - Full Match Extraction Pipeline (v3)

Takes long Pokemon Unite VODs (local file, YouTube/Twitch URL, or a folder
of videos) and extracts only the COMPLETE gameplay matches (GO -> Time's Up)
into data/gameplay_segments/. Lobby / queue / menus / loading footage is
ignored. This pipeline never writes to data/creative/shortlisted_clips/.

Menu:
    1. Extract full matches from local VOD
    2. Download YouTube/Twitch VOD and extract full matches
    3. Copy all videos from input folder into data/raw_vods
    4. List raw VODs
    5. Exit
"""

import os
import re
import shutil
import subprocess

from src.preprocess.match_extractor import MatchExtractor

RAW_DIR = os.path.join("data", "raw_vods")
GAMEPLAY_DIR = os.path.join("data", "gameplay_segments")
DEBUG_DIR = os.path.join("data", "debug")

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    for d in (RAW_DIR, GAMEPLAY_DIR, DEBUG_DIR, os.path.join("data", "inputs")):
        os.makedirs(d, exist_ok=True)


def list_raw_vods() -> list:
    if not os.path.isdir(RAW_DIR):
        return []
    files = [f for f in sorted(os.listdir(RAW_DIR))
             if f.lower().endswith(VIDEO_EXTS)
             and os.path.isfile(os.path.join(RAW_DIR, f))]
    return files


def pick_vod() -> str:
    files = list_raw_vods()
    if not files:
        print(f"❌ No videos found in {RAW_DIR}/")
        print("   Use option 2 (download) or option 3 (copy from folder) first.")
        return ""
    print(f"\n📂 Videos in {RAW_DIR}/:")
    for idx, f in enumerate(files, 1):
        size_gb = os.path.getsize(os.path.join(RAW_DIR, f)) / (1024 ** 3)
        print(f"  {idx}. {f}  ({size_gb:.2f} GB)")
    raw = input(f"\nSelect video [1-{len(files)}]: ").strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(files)):
        print("❌ Invalid selection.")
        return ""
    return os.path.join(RAW_DIR, files[int(raw) - 1])


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]", "_", name).strip()
    return re.sub(r"\s+", "_", name)[:120] or "vod"


# ---------------------------------------------------------------------------
# menu actions
# ---------------------------------------------------------------------------

def action_extract_local():
    vod = pick_vod()
    if vod:
        MatchExtractor().extract_matches(vod)


def action_download_and_extract():
    url = input("\n🔗 Enter YouTube/Twitch URL: ").strip()
    if not url:
        print("❌ No URL given.")
        return

    before = set(list_raw_vods())
    print("\n⬇️ [Downloader] Fetching VOD with yt-dlp (this can take a while)...")
    out_template = os.path.join(RAW_DIR, "%(title).80s_%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "--merge-output-format", "mp4",
        "--restrict-filenames",
        "-o", out_template,
        url,
    ]
    try:
        result = subprocess.run(cmd)
    except FileNotFoundError:
        print("❌ yt-dlp is not installed. Install it with: pip install yt-dlp")
        return
    if result.returncode != 0:
        print("❌ Download failed. Check the URL and try again.")
        return

    new_files = sorted(set(list_raw_vods()) - before)
    if not new_files:
        print("⚠️ Download finished but no new video appeared in data/raw_vods/.")
        return
    target = os.path.join(RAW_DIR, new_files[-1])
    print(f"✅ Downloaded: {target}")
    MatchExtractor().extract_matches(target)


def action_copy_from_folder():
    folder = input("\n📁 Enter input folder path: ").strip().strip('"').strip("'")
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        print(f"❌ Not a valid folder: {folder}")
        return

    copied, skipped = 0, 0
    for f in sorted(os.listdir(folder)):
        src = os.path.join(folder, f)
        if not os.path.isfile(src):
            continue
        if not f.lower().endswith(VIDEO_EXTS):
            print(f"   ⏭️ Skipped (unsupported type): {f}")
            skipped += 1
            continue
        dst = os.path.join(RAW_DIR, safe_filename(f))
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            print(f"   ⏭️ Skipped (already in raw_vods): {f}")
            skipped += 1
            continue
        print(f"   📥 Copying: {f} ...")
        shutil.copy2(src, dst)
        copied += 1
    print(f"\n✅ Copied {copied} file(s), skipped {skipped}. Destination: {RAW_DIR}/")


def action_list_raw_vods():
    files = list_raw_vods()
    if not files:
        print(f"\n📂 {RAW_DIR}/ is empty (no {', '.join(VIDEO_EXTS)} files).")
        return
    print(f"\n📂 {len(files)} video(s) in {RAW_DIR}/:")
    for idx, f in enumerate(files, 1):
        size_gb = os.path.getsize(os.path.join(RAW_DIR, f)) / (1024 ** 3)
        print(f"  {idx}. {f}  ({size_gb:.2f} GB)")


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def run_menu():
    ensure_dirs()
    while True:
        print("\n" + "=" * 50)
        print("🎬 CRUSHER11 VOD STUDIO - FULL MATCH EXTRACTION")
        print("=" * 50)
        print("1. Extract full matches from local VOD")
        print("2. Download YouTube/Twitch VOD and extract full matches")
        print("3. Copy all videos from input folder into data/raw_vods")
        print("4. List raw VODs")
        print("5. Exit")
        print("=" * 50)

        choice = input("\nEnter choice [1-5]: ").strip()
        try:
            if choice == "1":
                action_extract_local()
            elif choice == "2":
                action_download_and_extract()
            elif choice == "3":
                action_copy_from_folder()
            elif choice == "4":
                action_list_raw_vods()
            elif choice == "5":
                print("🛑 Exiting.")
                break
            else:
                print("❌ Invalid choice. Enter a number between 1 and 5.")
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted. Back to menu.")


if __name__ == "__main__":
    run_menu()
