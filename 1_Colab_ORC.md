!pip install easyocr opencv-python-headless

# from google.colab import drive
# drive.mount('/content/drive')
# from google.colab import drive
# drive.mount('/content/drive')
import cv2
import easyocr
import time
import numpy as np
import json
import os
import re
from collections import deque, Counter
from tqdm import tqdm

def canonicalize_text(text):
# ... the rest of the code is perfect!
    text = text.upper()
    if "DOUBLE" in text: return "DOUBLE KO"
    if "TRIPLE" in text: return "TRIPLE KO"
    if "QUADRA" in text: return "QUADRA KO"
    if "PENTA" in text: return "PENTA KO"
    if "STREAK" in text: return "KO STREAK"
    if re.search(r'\b(RAYQUAZA|ZAPDOS|REGIELEKI|REGIROCK|REGICE|REGISTEEL)\b', text): return "OBJECTIVE"
    if "GOAL" in text: return "GOAL"
    if "KO" in text: return "KO"
    return text

def preprocess_for_ocr(crop):
    """
    Simplified Preprocessing: Deep learning models (EasyOCR) prefer
    natural textures over heavily thresholded binary images.
    Upscaling provides more pixel density for small mobile text.
    """
    return cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

def classify_event(text):
    text = text.upper()
    if re.search(r'\b(YOU WERE DEFEATED|DEFEATED)\b', text) and not re.search(r'\b(SECURED|STOLEN)\b', text):
        return "PLAYER_DEATH", -10
    if re.search(r'\b(DOUBLE|TRIPLE|QUADRA|PENTA)\s+KO\b|\b\d+\s+KO\s+STREAK\b', text):
        return "MULTI_KO", 10
    if re.search(r'\b(RAYQUAZA|ZAPDOS|REGIELEKI|REGIROCK|REGICE|REGISTEEL)\s+(SECURED|STOLEN)\b', text):
        return "MAJOR_OBJECTIVE", 10
    if re.search(r'\b(SECURED|STOLEN)\b', text):
        return "MINOR_OBJECTIVE", 5
    if re.search(r'\b(GOAL|DESTROYED)\b', text):
        return "SCORING", 4
    if re.search(r'\bKO\b', text):
        return "SINGLE_KO", 2
    return "ACTION_DETECTED", 1

def extract_gameplay_events(video_path):
    print(f"🔍 [Phase 1a] Booting Event Extraction Engine for {video_path}...")

    allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :'
    reader = easyocr.Reader(['en'], gpu=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ [CRITICAL ERROR] OpenCV could not find or open the video at: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    structured_events = []
    rolling_memory = deque(maxlen=5)

    last_logged_signature = None
    last_logged_time = 0.0
    SIGNATURE_RESET_SECONDS = 8.0

    debug_dir = "data/ocr_debug_crops"
    os.makedirs(debug_dir, exist_ok=True)

    print("🚀 Scanning video (2 FPS | 2-Frame Consensus | Debug Mode)...")

    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")
    current_frame = 0

    # 👇 Patch 1: Safe frame sampling math
    sample_interval = max(1, int(fps * 0.5))

    while cap.isOpened():
        ret = cap.grab()
        if not ret: break

        current_second = round(current_frame / fps, 1)

        if current_frame % sample_interval == 0:
            ret, frame = cap.retrieve()

            # 👇 Patch 2: Corrupted frame safety guard
            if not ret or frame is None:
                current_frame += 1
                pbar.update(1)
                continue

            height, width, _ = frame.shape

            # 👇 VERTICAL SHORTS CROP (15% to 60%)
            y1 = int(height * 0.15)
            y2 = int(height * 0.60)
            x1 = 0
            x2 = width

            composite_crop = frame[y1:y2, x1:x2]
            clean_thresh = preprocess_for_ocr(composite_crop)

            # 👇 Patch 3: Safe crop saving using exact frame math, not float seconds
            if current_frame % int(fps * 10) == 0:
                cv2.imwrite(os.path.join(debug_dir, f"crop_check_{int(current_second)}s.jpg"), composite_crop)

            results = reader.readtext(clean_thresh, detail=1, allowlist=allowlist)

            valid_words = []
            highest_confidence = 0.0
            raw_ocr_reads = []

            for (bbox, text, prob) in results:
                box_width = bbox[1][0] - bbox[0][0]
                box_height = bbox[2][1] - bbox[1][1]
                box_area = box_width * box_height

                clean_text = text.upper().strip()
                raw_ocr_reads.append(f"[{prob:.2f} | Area:{int(box_area)}] {clean_text}")

                if clean_text.isdigit() or "POK" in clean_text or "UNIB" in clean_text or "UNIT" in clean_text:
                    continue

                clean_text = re.sub(r'\b\d{2}\s*:?\s*\d{2}\b', '', clean_text)
                clean_text = re.sub(r'\b\d{3,4}\b', '', clean_text)
                clean_text = re.sub(r'\b(POK|MON|UNIB|UNIT|ROAD|MASTER|POKMON|UNITE)\b', '', clean_text).strip()

                if not clean_text: continue

                VALID_SHORT_WORDS = {"KO", "GO", "GG", "UP"}
                if len(clean_text) <= 2 and clean_text not in VALID_SHORT_WORDS:
                    continue

                # 👇 Relaxed Filters for vertical text
                if prob > 0.35 and box_area > 50:
                    valid_words.append(clean_text)
                    if prob > highest_confidence: highest_confidence = round(prob, 2)

            detected_text = " ".join(valid_words)
            detected_text = re.sub(r'\s+', ' ', detected_text).strip()

            event_category, weight = classify_event(detected_text)

            # 👇 Patch 4: Targeted X-Ray Logging (Only log actionable events to prevent spam)
            if results and len(valid_words) > 0 and event_category != "ACTION_DETECTED":
                tqdm.write(f"📊 [X-RAY @ {current_second}s] {event_category} -> '{detected_text}'")
                tqdm.write(f"     ↳ RAW READ: {', '.join(raw_ocr_reads)}")

            if event_category != "ACTION_DETECTED":
                rolling_memory.append({
                    "category": event_category,
                    "text": detected_text,
                    "confidence": highest_confidence,
                    "weight": weight
                })
            else:
                rolling_memory.append({"category": "NONE", "confidence": 0.0})

            if len(rolling_memory) == 5:
                categories = [item['category'] for item in rolling_memory if item['category'] != "NONE"]

                if len(categories) > 0:
                    counter = Counter(categories)
                    most_common_category, count = counter.most_common(1)[0]

                    # 👇 RELAXED CONSENSUS: Only 2 matching frames required!
                    if count >= 2:
                        matching_frames = [item for item in rolling_memory if item['category'] == most_common_category]
                        avg_confidence = round(sum(f['confidence'] for f in matching_frames) / len(matching_frames), 2)
                        best_read = max(matching_frames, key=lambda x: x['confidence'])
                        current_signature = (most_common_category, best_read['text'])
                        time_since_last = current_second - last_logged_time

                        if current_signature != last_logged_signature or time_since_last > SIGNATURE_RESET_SECONDS:
                            event = {
                                "timestamp": current_second,
                                "event_type": most_common_category,
                                "priority_weight": best_read['weight'],
                                "raw_text": best_read['text'],
                                "canonical_text": canonicalize_text(best_read['text']),
                                "confidence": avg_confidence
                            }
                            structured_events.append(event)
                            last_logged_signature = current_signature
                            last_logged_time = current_second
                            tqdm.write(f"🎯 [CONSENSUS MATCH: {most_common_category}] {current_second}s | Text: '{best_read['text']}'")

        current_frame += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    os.makedirs("data/inputs", exist_ok=True)
    out_path = "data/inputs/ocr_structured_log.json"
    with open(out_path, "w") as f:
        json.dump(structured_events, f, indent=4)

    print(f"✅ Extraction complete. {len(structured_events)} events saved to {out_path}.")

# Run it!
extract_gameplay_events('/content/drive/MyDrive/VS_Factory/inputs/PokemonUnite.mp4')
