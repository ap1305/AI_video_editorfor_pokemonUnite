import cv2
import numpy as np
import json
import os
from tqdm.notebook import tqdm

def extract_motion_chaos(video_path):
    print(f"👁️ [Phase 1c] Booting Visual Chaos Sensor for {video_path}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ [CRITICAL ERROR] OpenCV could not open: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs("data/inputs", exist_ok=True)
    chaos_events = []
    prev_gray = None

    print("🚀 Scanning video (2 FPS | Dual-Metric Telemetry | Tighter Crop)...")
    pbar = tqdm(total=total_frames, desc="Processing Visuals", unit="frame")

    current_frame = 0
    sample_interval = max(1, int(fps * 0.5))

    while cap.isOpened():
        ret = cap.grab()
        if not ret: break
        current_second = round(current_frame / fps, 1)

        if current_frame % sample_interval == 0:
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                current_frame += 1
                pbar.update(1)
                continue

            height, width, _ = frame.shape

            # Extreme tight crop for motion (ignores UI and minimap)
            y1 = int(height * 0.12)
            y2 = int(height * 0.55)
            x1 = int(width * 0.10)
            x2 = int(width * 0.90)

            gameplay_crop = frame[y1:y2, x1:x2]
            small_frame = cv2.resize(gameplay_crop, (0, 0), fx=0.5, fy=0.5)
            curr_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                frame_diff = cv2.absdiff(prev_gray, curr_gray)

                # The Dual-Metrics
                chaos_score = np.mean(frame_diff)
                motion_pixels = np.count_nonzero(frame_diff > 20)

                # Catch anything above absolute static zero
                if chaos_score > 2.0:
                    chaos_events.append({
                        "timestamp": current_second,
                        "event_type": "VISUAL_CHAOS_SPIKE",
                        "priority_weight": 6,
                        "raw_text": f"Chaos Score: {round(chaos_score, 1)} | Pixels: {motion_pixels}"
                    })

            prev_gray = curr_gray

        current_frame += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if chaos_events:
        scores = [float(e["raw_text"].split(" | ")[0].replace("Chaos Score: ", "")) for e in chaos_events]
        pixel_counts = [int(e["raw_text"].split(" | ")[1].replace("Pixels: ", "")) for e in chaos_events]

        print(f"\n📈 [Telemetry - Intensity] | Min={min(scores):.2f} Avg={np.mean(scores):.2f} Max={max(scores):.2f}")
        print(f"📈 [Telemetry - Pixels]    | Min={min(pixel_counts)} Avg={int(np.mean(pixel_counts))} Max={max(pixel_counts)}")

        threshold = np.percentile(scores, 90)
        significant_chaos = [e for e in chaos_events if float(e["raw_text"].split(" | ")[0].replace("Chaos Score: ", "")) >= threshold]
    else:
        significant_chaos = []

    out_path = "data/inputs/motion_events_log.json"
    with open(out_path, "w") as f:
        json.dump(significant_chaos, f, indent=4)

    print(f"✅ Motion extraction complete. Found {len(significant_chaos)} massive visual explosions.")

# Run it!
extract_motion_chaos('/content/drive/MyDrive/VS_Factory/inputs/PokemonUnite.mp4')
