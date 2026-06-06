#cell 1 :-
from google.colab import drive
drive.mount('/content/drive')

#Cell 2:-
!pip install easyocr opencv-python-headless


#Cell 3 :- OCR

import cv2
import easyocr
import time
import numpy as np
import json
import os
import re
from collections import deque, Counter
from tqdm import tqdm

# Master Output Paths - Set strictly to local data/inputs/
BASE_DIR = "data/inputs"
OUT_JSON_PATH = os.path.join(BASE_DIR, "ocr_structured_log.json")
BROAD_DEBUG_DIR = os.path.join(BASE_DIR, "ocr_debug_crops")
SCORE_DEBUG_DIR = os.path.join(BASE_DIR, "score_debug_crops")

# Guarantee directories exist to prevent FileNotFoundError
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(BROAD_DEBUG_DIR, exist_ok=True)
os.makedirs(SCORE_DEBUG_DIR, exist_ok=True)

def canonicalize_text(text):
    text = text.upper()
    if "100" in text: return "100_GOAL"
    if "50" in text: return "50_GOAL"
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
    return cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

def classify_event(text):
    text = text.upper()
    if "100" in text: return "MASSIVE_SCORE", 50
    if "50" in text: return "MAJOR_SCORE", 30

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

def normalize_score_text(text):
    raw = text.upper().strip()
    tokens = re.findall(r'[A-Z0-9]+', raw)
    for tok in tokens:
        compact = tok.strip()
        if compact in {"100", "1OO", "I00", "IOO", "L00", "10O", "JOO"}: return "100"
        if re.match(r'^(100|1OO|I00|IOO|L00|10O|JOO)[A-Z]$', compact): return "100"
        if compact in {"50", "5O", "S0"}: return "50"
        if re.match(r'^(50|5O|S0)[A-Z]$', compact): return "50"
        if compact == "SO" and re.search(r'\b(SCORE|GOAL)\b', raw): return "50"
    return None

def extract_gameplay_events(video_path):
    print(f"🔍 [Phase 1a] Booting V9 Diagnostic OCR Engine for {video_path}...")

    allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :'
    reader = easyocr.Reader(['en'], gpu=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ [CRITICAL ERROR] OpenCV could not find or open the video at {video_path}.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    structured_events = []
    rolling_memory = deque(maxlen=5)

    last_logged_signature = None
    last_logged_time = 0.0
    SIGNATURE_RESET_SECONDS = 8.0

    last_score_signature = None
    last_score_time = -999.0
    # Strict 12-second debouncing to prevent static UI flicker spam
    SCORE_RESET_SECONDS = 12.0

    print("🚀 Scanning video (2 FPS | Dual-Crop V9 | Debug Mode)...")

    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")
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

            # ==========================================
            # PASS 1: NARRATIVE BROAD CROP (KOs & Objectives)
            # ==========================================
            y1_b, y2_b = int(height * 0.15), int(height * 0.60)
            x1_b, x2_b = 0, width
            crop_broad = preprocess_for_ocr(frame[y1_b:y2_b, x1_b:x2_b])

            broad_results = reader.readtext(crop_broad, detail=1, allowlist=allowlist)

            valid_words_broad = []
            highest_conf_broad = 0.0

            for (bbox, text, prob) in broad_results:
                clean_text = text.upper().strip()

                if normalize_score_text(clean_text) is not None:
                    continue

                clean_text = re.sub(r'\b\d{2}\s*:?\s*\d{2}\b', '', clean_text)
                clean_text = re.sub(r'\b(POK|MON|UNIB|UNIT|ROAD|MASTER|POKMON|UNITE|FPS)\b', '', clean_text).strip()
                if not clean_text: continue

                VALID_SHORT = {"KO", "GO", "GG", "UP"}
                if len(clean_text) <= 2 and clean_text not in VALID_SHORT: continue

                if prob > 0.35:
                    valid_words_broad.append(clean_text)
                    if prob > highest_conf_broad: highest_conf_broad = round(prob, 2)

            detected_broad = re.sub(r'\s+', ' ', " ".join(valid_words_broad)).strip()
            event_cat_broad, weight_broad = classify_event(detected_broad)

            if event_cat_broad not in ["ACTION_DETECTED", "MASSIVE_SCORE", "MAJOR_SCORE"]:
                rolling_memory.append({
                    "category": event_cat_broad,
                    "text": detected_broad,
                    "confidence": highest_conf_broad,
                    "weight": weight_broad
                })
            else:
                rolling_memory.append({"category": "NONE", "confidence": 0.0})

            if len(rolling_memory) == 5:
                categories = [item['category'] for item in rolling_memory if item['category'] != "NONE"]
                if len(categories) > 0:
                    counter = Counter(categories)
                    most_common_category, count = counter.most_common(1)[0]

                    if count >= 2:
                        matching_frames = [item for item in rolling_memory if item['category'] == most_common_category]
                        avg_conf = round(sum(f['confidence'] for f in matching_frames) / len(matching_frames), 2)
                        best_read = max(matching_frames, key=lambda x: x['confidence'])
                        current_sig = (most_common_category, best_read['text'])
                        time_since = current_second - last_logged_time

                        if current_sig != last_logged_signature or time_since > SIGNATURE_RESET_SECONDS:
                            structured_events.append({
                                "timestamp": current_second,
                                "event_type": most_common_category,
                                "priority_weight": best_read['weight'],
                                "raw_text": best_read['text'],
                                "canonical_text": canonicalize_text(best_read['text']),
                                "confidence": avg_conf
                            })
                            last_logged_signature = current_sig
                            last_logged_time = current_second

            # ==========================================
            # PASS 2: SNIPER SCORE CROP (Only 50/100)
            # ==========================================
            y1_s, y2_s = int(height * 0.20), int(height * 0.50)
            x1_s, x2_s = int(width * 0.30), int(width * 0.70)
            crop_score = preprocess_for_ocr(frame[y1_s:y2_s, x1_s:x2_s])

            score_allowlist = '0123456789IOOLJS '
            score_results = reader.readtext(crop_score, detail=1, allowlist=score_allowlist)

            for (bbox, text, prob) in score_results:
                clean_score = text.upper().strip()
                score_norm = normalize_score_text(clean_score)

                if score_norm and prob > 0.40:
                    box_area = (bbox[1][0] - bbox[0][0]) * (bbox[2][1] - bbox[1][1])
                    if box_area < 250: continue

                    event_type = "MASSIVE_SCORE" if score_norm == "100" else "MAJOR_SCORE"
                    weight = 50 if score_norm == "100" else 30

                    score_signature = f"{score_norm}_GOAL"

                    if score_signature != last_score_signature or (current_second - last_score_time) > SCORE_RESET_SECONDS:

                        # Generate the visual debug crop to find static UI noise
                        cv2.imwrite(
                            os.path.join(SCORE_DEBUG_DIR, f"score_{int(current_second)}s_{score_norm}.jpg"),
                            frame[y1_s:y2_s, x1_s:x2_s]
                        )

                        structured_events.append({
                            "timestamp": current_second,
                            "event_type": event_type,
                            "priority_weight": weight,
                            "raw_text": str(score_norm),
                            "canonical_text": score_signature,
                            "confidence": round(prob, 2)
                        })

                        last_score_signature = score_signature
                        last_score_time = current_second
                        tqdm.write(f"🎯 [SNIPER SCORE DETECTED: {event_type}] {current_second}s | Area: {int(box_area)}")

                    break

        current_frame += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    with open(OUT_JSON_PATH, "w") as f:
        json.dump(structured_events, f, indent=4)

    print(f"✅ Extraction complete. {len(structured_events)} events saved to {OUT_JSON_PATH}.")

    score_events = [e for e in structured_events if e.get("canonical_text") in ["100_GOAL", "50_GOAL"]]
    print("\n🏆 SCORE EVENTS FOUND:", len(score_events))
    if score_events:
        print(json.dumps(score_events[:10], indent=2))

# Provide your local video path here
video_target = '/content/drive/MyDrive/VS_Factory/inputs/PokemonUnite.mp4'
extract_gameplay_events(video_target)

##Cell 4 :- Audio
import subprocess
import numpy as np
from scipy.io import wavfile
import json
import os

def extract_audio_hype(video_path):
    print(f"🎧 [Phase 1b] Booting Audio Sensor for {video_path}...")

    os.makedirs("data/inputs", exist_ok=True)
    os.makedirs("data/temp", exist_ok=True)
    temp_wav = "data/temp/temp_audio.wav"

    # 1. Extract audio instantly using FFmpeg
    print("⏳ Extracting audio track (this takes about 5 seconds)...")
    command = f"ffmpeg -y -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 1 {temp_wav} -loglevel quiet"
    subprocess.call(command, shell=True)

    if not os.path.exists(temp_wav):
        print("❌ [Error] Failed to extract audio.")
        return

    # 2. Read the audio file
    print("🔊 Calculating Hype Volumes...")
    sample_rate, data = wavfile.read(temp_wav)

    # Calculate exactly how many seconds long the video is
    total_seconds = len(data) // sample_rate

    hype_events = []

    # 3. Calculate the volume (RMS) for every single second
    for second in range(total_seconds):
        start_idx = second * sample_rate
        end_idx = start_idx + sample_rate
        second_data = data[start_idx:end_idx]

        # Calculate Root Mean Square (RMS) to get true loudness
        rms_volume = np.sqrt(np.mean(second_data.astype(np.float64)**2))
        hype_events.append({
            "timestamp": float(second),
            "volume": float(rms_volume)
        })

    # 4. Find the baseline volume and identify the top 10% loudest moments
    volumes = [e["volume"] for e in hype_events]
    threshold = np.percentile(volumes, 90) # Top 10% loudest spikes

    significant_spikes = []
    for event in hype_events:
        if event["volume"] >= threshold:
            significant_spikes.append({
                "timestamp": event["timestamp"],
                "event_type": "AUDIO_HYPE_SPIKE",
                "priority_weight": 8, # Highly weighted for the Director!
                "raw_text": f"Volume Level: {int(event['volume'])}"
            })

    # Clean up temp file
    os.remove(temp_wav)

    out_path = "data/inputs/audio_hype_log.json"
    #out_path = "/content/drive/MyDrive/VS_Factory/inputs/audio_hype_log.json"
    with open(out_path, "w") as f:
        json.dump(significant_spikes, f, indent=4)

    print(f"✅ Audio extraction complete. Found {len(significant_spikes)} massive hype moments.")
    print(f"📁 Memory saved to {out_path}.")

# Run it!
extract_audio_hype('/content/drive/MyDrive/VS_Factory/inputs/PokemonUnite.mp4')

##Cell 5 :- Motion
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
    #out_path = "/content/drive/MyDrive/VS_Factory/inputs/motion_events_log.json"
    with open(out_path, "w") as f:
        json.dump(significant_chaos, f, indent=4)

    print(f"✅ Motion extraction complete. Found {len(significant_chaos)} massive visual explosions.")

# Run it!
extract_motion_chaos('/content/drive/MyDrive/VS_Factory/inputs/PokemonUnite.mp4')

## CELL 6 :- Fusion
import json
import os
from collections import Counter

BASE_DIR = 'data/inputs'

# ==========================================
# 0. NAMED CONSTANTS
# ==========================================
CONTEXT_PAD_SECONDS = 3.0
PAYOFF_TAIL_SECONDS = 5.0
CLUSTER_GAP_SECONDS = 5.0
IMPACT_COMBO_WINDOW_SECONDS = 10.0

# ==========================================
# 1. CORE HELPER FUNCTIONS
# ==========================================
def safe_float(value, default=0.0):
    try:
        if value is None: return default
        return float(value)
    except (ValueError, TypeError):
        return default

def get_event_timestamp(event):
    return safe_float(event.get("timestamp", event.get("time", 0.0)))

def parse_motion_score(raw_text):
    if not raw_text or not isinstance(raw_text, str):
        return 0.0
    try:
        return safe_float(raw_text.split(" | ")[0].replace("Chaos Score: ", ""))
    except Exception:
        return 0.0

def calculate_overlap_ratio(c_start, c_end, s_start, s_end):
    overlap_start = max(c_start, s_start)
    overlap_end = min(c_end, s_end)
    overlap_duration = max(0, overlap_end - overlap_start)

    c_duration = c_end - c_start
    s_duration = s_end - s_start
    smaller_duration = min(c_duration, s_duration)

    if smaller_duration <= 0:
        return 0.0
    return overlap_duration / smaller_duration

def apply_nms(candidates, overlap_threshold=0.30):
    sorted_candidates = sorted(candidates, key=lambda x: safe_float(x.get("importance_score", 0)), reverse=True)
    selected = []

    for candidate in sorted_candidates:
        keep = True
        c_start = candidate["start_time"]
        c_end = candidate["end_time"]

        for s in selected:
            overlap_ratio = calculate_overlap_ratio(c_start, c_end, s["start_time"], s["end_time"])
            if overlap_ratio > overlap_threshold:
                keep = False
                break

        if keep:
            selected.append(candidate)

    return selected

def validate_candidate_window(candidate, cand_start, cand_end):
    candidate["start_time"] = safe_float(candidate.get("start_time"), cand_start)
    candidate["end_time"] = safe_float(candidate.get("end_time"), cand_end)
    candidate["importance_score"] = safe_float(candidate.get("importance_score"))

    if not isinstance(candidate.get("ocr_events"), list):
        candidate["ocr_events"] = []

    if candidate["end_time"] <= candidate["start_time"]:
        candidate["start_time"] = cand_start
        candidate["end_time"] = cand_end

    if candidate["start_time"] < 0:
        candidate["start_time"] = 0.0

    return candidate

# ==========================================
# 4. IMPACT EVENT GENERATION
# ==========================================
def build_player_impact_events(ocr_events):
    if not ocr_events:
        return []

    sorted_events = sorted(ocr_events, key=get_event_timestamp)
    clusters = []
    current_cluster = []

    for event in sorted_events:
        ts = get_event_timestamp(event)
        if not current_cluster:
            current_cluster.append(event)
            continue

        last_ts = get_event_timestamp(current_cluster[-1])
        if ts - last_ts <= IMPACT_COMBO_WINDOW_SECONDS:
            current_cluster.append(event)
        else:
            clusters.append(current_cluster)
            current_cluster = [event]

    if current_cluster:
        clusters.append(current_cluster)

    impact_events = []

    for cluster in clusters:
        timestamps = [get_event_timestamp(e) for e in cluster]
        cluster_start = min(timestamps)
        cluster_end = max(timestamps)

        canonical_texts = [e.get("canonical_text", "") for e in cluster]
        event_types = [e.get("event_type", "") for e in cluster]

        has_ko = any("KO" in text or "KO" in etype for text, etype in zip(canonical_texts, event_types))
        has_100 = any(text == "100_GOAL" for text in canonical_texts)
        has_50 = any(text == "50_GOAL" for text in canonical_texts)
        has_death = any(etype == "PLAYER_DEATH" for etype in event_types)

        goal_value = 100 if has_100 else 50 if has_50 else 0

        score_events = [e for e in cluster if e.get("canonical_text") in ["50_GOAL", "100_GOAL"]]
        ko_events = [e for e in cluster if "KO" in e.get("canonical_text", "") or "KO" in e.get("event_type", "")]

        first_score_ts = min([get_event_timestamp(e) for e in score_events]) if score_events else None
        last_score_ts = max([get_event_timestamp(e) for e in score_events]) if score_events else None
        first_ko_ts = min([get_event_timestamp(e) for e in ko_events]) if ko_events else None
        last_ko_ts = max([get_event_timestamp(e) for e in ko_events]) if ko_events else None

        if goal_value and has_ko:
            first_meaning_ts = min(first_score_ts, first_ko_ts)
            last_meaning_ts = max(last_score_ts, last_ko_ts)

            combo_type = "KO_THEN_SCORE" if first_ko_ts < first_score_ts else "SCORE_THEN_KO"
            impact_score = 120 if goal_value == 100 else 95

            if has_death:
                impact_score -= 20

            impact_events.append({
                "timestamp": round(cluster_start, 1),
                "combo_type": combo_type,
                "first_score_ts": first_score_ts,
                "last_score_ts": last_score_ts,
                "first_ko_ts": first_ko_ts,
                "last_ko_ts": last_ko_ts,
                "first_meaning_ts": round(first_meaning_ts, 1),
                "last_meaning_ts": round(last_meaning_ts, 1),
                "payoff_ts": round(last_meaning_ts, 1),
                "goal_value": goal_value,
                "ko_signal": True,
                "goal_signal": True,
                "score_and_ko_combo": True,
                "death_signal": has_death,
                "player_impact_score": impact_score,
                "story_hint": f"Player gets a KO and scores {goal_value}" if combo_type == "KO_THEN_SCORE" else f"Player scores {goal_value} and then gets a KO"
            })

        elif goal_value:
            impact_events.append({
                "timestamp": round(cluster_start, 1),
                "combo_type": "MASSIVE_SCORE" if goal_value == 100 else "MAJOR_SCORE",
                "first_score_ts": first_score_ts,
                "last_score_ts": last_score_ts,
                "first_ko_ts": None,
                "last_ko_ts": None,
                "first_meaning_ts": round(first_score_ts, 1),
                "last_meaning_ts": round(last_score_ts, 1),
                "payoff_ts": round(last_score_ts, 1),
                "goal_value": goal_value,
                "ko_signal": False,
                "goal_signal": True,
                "score_and_ko_combo": False,
                "death_signal": has_death,
                "player_impact_score": 90 if goal_value == 100 else 70,
                "story_hint": f"Player scores {goal_value}"
            })

        elif has_ko:
            impact_score = 35
            story_hint = "Player gets a KO"
            if has_death:
                impact_score = 5
                story_hint = "Player gets a KO but dies with no clear payoff"

            impact_events.append({
                "timestamp": round(cluster_start, 1),
                "combo_type": "KO_ONLY",
                "first_score_ts": None,
                "last_score_ts": None,
                "first_ko_ts": first_ko_ts,
                "last_ko_ts": last_ko_ts,
                "first_meaning_ts": round(first_ko_ts, 1),
                "last_meaning_ts": round(last_ko_ts, 1),
                "payoff_ts": round(last_ko_ts, 1),
                "goal_value": 0,
                "ko_signal": True,
                "goal_signal": False,
                "score_and_ko_combo": False,
                "death_signal": has_death,
                "player_impact_score": impact_score,
                "story_hint": story_hint
            })

    return sorted(impact_events, key=lambda x: x.get("player_impact_score", 0), reverse=True)

# ==========================================
# 6. STORY-DRIVEN ENRICHMENT
# ==========================================
def enrich_candidates_with_impact(candidate_windows, impact_events):
    enriched_candidates = []

    for original_candidate in candidate_windows:
        candidate = dict(original_candidate)
        cand_start = safe_float(candidate.get("start_time", candidate.get("start", 0)))
        cand_end = safe_float(candidate.get("end_time", candidate.get("end", 0)))

        audio_d = safe_float(candidate.get("audio_density", 0))
        motion_d = safe_float(candidate.get("motion_density", 0))
        peak_v = safe_float(candidate.get("peak_volume", 0))

        visual_support_score = round((audio_d * 2.0) + (motion_d * 1.5) + (peak_v / 1000.0), 2)
        candidate["visual_support_score"] = visual_support_score
        candidate["risk_flags"] = []

        matching_impacts = [
            impact for impact in impact_events
            if safe_float(impact.get("first_meaning_ts", impact.get("timestamp", 0))) <= cand_end
            and safe_float(impact.get("last_meaning_ts", impact.get("timestamp", 0))) >= cand_start
        ]

        if matching_impacts:
            best_impact = max(matching_impacts, key=lambda x: safe_float(x.get("player_impact_score", 0)))
            combo_type = best_impact.get("combo_type", "GENERIC_ACTION")

            strong_combo_types = ["KO_THEN_SCORE", "SCORE_THEN_KO", "MASSIVE_SCORE", "MAJOR_SCORE"]
            if combo_type in strong_combo_types:
                first_meaning_ts = safe_float(best_impact.get("first_meaning_ts", cand_start))
                last_meaning_ts = safe_float(best_impact.get("last_meaning_ts", cand_end))
                payoff_ts = safe_float(best_impact.get("payoff_ts", cand_end))

                candidate["start_time"] = max(0.0, first_meaning_ts - CONTEXT_PAD_SECONDS)
                candidate["end_time"] = payoff_ts + PAYOFF_TAIL_SECONDS

            candidate["combo_type"] = combo_type
            candidate["goal_value"] = best_impact.get("goal_value", 0)
            candidate["player_impact_score"] = best_impact.get("player_impact_score", 10)
            candidate["story_hint"] = best_impact.get("story_hint", "Standard gameplay action")
            candidate["goal_signal"] = best_impact.get("goal_signal", False)
            candidate["ko_signal"] = best_impact.get("ko_signal", False)
            candidate["score_and_ko_combo"] = best_impact.get("score_and_ko_combo", False)
            candidate["death_signal"] = best_impact.get("death_signal", False)

            for ts_key in ["first_score_ts", "last_score_ts", "first_ko_ts", "last_ko_ts", "first_meaning_ts", "last_meaning_ts", "payoff_ts"]:
                candidate[ts_key] = best_impact.get(ts_key)

            candidate["editor_reason"] = f"Identified as {combo_type}. Visual support score: {visual_support_score}."

            existing_ocr = candidate.get("ocr_events", [])
            enriched_ocr = list(existing_ocr)
            if candidate["goal_value"] == 100: enriched_ocr.append("100_GOAL")
            elif candidate["goal_value"] == 50: enriched_ocr.append("50_GOAL")
            if candidate["ko_signal"]: enriched_ocr.append("KO")
            if candidate["death_signal"]: enriched_ocr.append("PLAYER_DEATH")
            candidate["ocr_events"] = sorted(list(set(enriched_ocr)))

        else:
            candidate["combo_type"] = "GENERIC_ACTION"
            candidate["goal_value"] = 0
            candidate["player_impact_score"] = 10
            candidate["story_hint"] = "Standard gameplay action"
            candidate["editor_reason"] = f"Generic action with visual support score: {visual_support_score}."

            candidate["goal_signal"] = False
            existing_ocr = candidate.get("ocr_events", [])
            candidate["ko_signal"] = any("KO" in str(x) for x in existing_ocr)
            candidate["score_and_ko_combo"] = False
            candidate["death_signal"] = any("PLAYER_DEATH" in str(x) or "DEFEATED" in str(x) for x in existing_ocr)

        if visual_support_score < 4.0:
            candidate["risk_flags"].append("LOW_VISUAL_SUPPORT")
            if candidate["score_and_ko_combo"]:
                candidate["risk_flags"].append("POSSIBLE_STATIC_SCORE_OR_HUD_NOISE")
                candidate["importance_score"] = safe_float(candidate.get("importance_score", 0)) * 0.85

        if candidate["death_signal"] and candidate["goal_value"] == 0:
            candidate["risk_flags"].append("DEATH_WITH_NO_PAYOFF")

        actual_duration = candidate["end_time"] - candidate["start_time"]
        if actual_duration < 8.0:
            candidate["risk_flags"].append("SHORT_CONTEXT")
        if actual_duration > 35.0:
            candidate["risk_flags"].append("LONG_DEAD_AIR_RISK")

        base_conf = min(1.0, visual_support_score / 20.0)
        if candidate.get("score_and_ko_combo"): base_conf = min(1.0, base_conf + 0.3)
        elif candidate.get("goal_value", 0) > 0: base_conf = min(1.0, base_conf + 0.2)

        if "LOW_VISUAL_SUPPORT" in candidate["risk_flags"]: base_conf *= 0.5
        candidate["story_confidence"] = round(base_conf, 2)

        candidate["editor_rules"] = {
            "duration_policy": "no_fixed_duration",
            "start_policy": "start_near_first_meaningful_action",
            "context_policy": "keep_only_needed_context",
            "end_policy": "end_after_visible_payoff",
            "compression_policy": "compress_dead_air_only",
            "quality_policy": "story_completion_over_duration"
        }

        candidate = validate_candidate_window(candidate, cand_start, cand_end)
        enriched_candidates.append(candidate)

    return enriched_candidates

# ==========================================
# 12. BASE GENERATION & FUSION
# ==========================================
def fuse_signals():
    print("⚙️ [Phase 1.5] Booting Unified Signal Fusion Layer...")

    ocr_path = os.path.join(BASE_DIR, "ocr_structured_log.json")
    audio_path = os.path.join(BASE_DIR, "audio_hype_log.json")
    motion_path = os.path.join(BASE_DIR, "motion_events_log.json")

    total_ocr = 0
    total_base_candidates = 0

    if not os.path.exists(ocr_path):
        print(f"❌ [Error] Missing OCR log! Expected at: {ocr_path}")
        return
    if not os.path.exists(audio_path):
        print(f"❌ [Error] Missing Audio log! Expected at: {audio_path}")
        return

    with open(ocr_path, "r") as f:
        ocr_events = json.load(f)
        total_ocr = len(ocr_events)
    with open(audio_path, "r") as f: audio_events = json.load(f)

    motion_events = []
    if os.path.exists(motion_path):
        with open(motion_path, "r") as f: motion_events = json.load(f)

    all_events = []
    for e in ocr_events: all_events.append({"time": get_event_timestamp(e), "type": "OCR", "data": e})
    for e in audio_events: all_events.append({"time": get_event_timestamp(e), "type": "AUDIO", "data": e})
    for e in motion_events: all_events.append({"time": get_event_timestamp(e), "type": "MOTION", "data": e})

    all_events.sort(key=lambda x: x["time"])

    clusters = []
    if all_events:
        current_cluster = [all_events[0]]
        for event in all_events[1:]:
            if event["time"] - current_cluster[-1]["time"] <= CLUSTER_GAP_SECONDS:
                current_cluster.append(event)
            else:
                clusters.append(current_cluster)
                current_cluster = [event]
        clusters.append(current_cluster)

    candidate_windows = []

    OCR_BONUS = {
        "PENTA KO": 200, "QUADRA KO": 120, "TRIPLE KO": 80, "DOUBLE KO": 50,
        "OBJECTIVE": 150, "100_GOAL": 200, "50_GOAL": 100, "MASSIVE_SCORE": 200,
        "MAJOR_SCORE": 100, "GOAL": 25, "KO": 5, "PLAYER_DEATH": -20
    }

    for idx, cluster in enumerate(clusters):
        start_time = cluster[0]["time"]
        end_time = cluster[-1]["time"]

        c_ocr = [e["data"] for e in cluster if e["type"] == "OCR"]
        c_audio = [e["data"] for e in cluster if e["type"] == "AUDIO"]
        c_motion = [e["data"] for e in cluster if e["type"] == "MOTION"]

        audio_density = len(c_audio)
        motion_density = len(c_motion)

        peak_volume = safe_float(max([safe_float(e["raw_text"].replace("Volume Level: ", "")) for e in c_audio]) if c_audio else 0)
        ocr_labels = [e.get("canonical_text", "UNKNOWN") for e in c_ocr]

        ocr_semantic_score = sum([OCR_BONUS.get(e.get("canonical_text", ""), 0) for e in c_ocr])
        motion_score_sum = sum([parse_motion_score(m["raw_text"]) for m in c_motion])

        capped_chaos_bonus = min(50.0, (motion_score_sum * 0.15))
        capped_audio_bonus = min(50.0, (audio_density * 2.0))

        raw_score = capped_audio_bonus + capped_chaos_bonus + (peak_volume / 1000) + ocr_semantic_score

        has_objective = any(l == "OBJECTIVE" for l in ocr_labels)
        pre_buffer = 15.0 if has_objective else 7.0

        if len(c_ocr) > 0 or (audio_density + motion_density) >= 2:
            candidate_windows.append({
                "window_id": f"CANDIDATE_{idx+1}",
                "start_time": max(0.0, start_time - pre_buffer),
                "end_time": end_time + 4.0,
                "importance_score": round(raw_score, 1),
                "ocr_events": list(set(ocr_labels)),
                "audio_density": audio_density,
                "motion_density": motion_density,
                "peak_volume": peak_volume
            })
            total_base_candidates += 1

    impact_events = build_player_impact_events(ocr_events)
    enriched_windows = enrich_candidates_with_impact(candidate_windows, impact_events)

    for w in enriched_windows:
        w["importance_score"] += safe_float(w.get("player_impact_score", 0))

    enriched_windows.sort(key=lambda x: safe_float(x.get("importance_score", 0)), reverse=True)
    deduplicated_windows = apply_nms(enriched_windows)

    final_candidates = []
    combo_counts = Counter()

    for cand in deduplicated_windows:
        c_type = cand.get("combo_type", "GENERIC_ACTION")
        if c_type in ["SCORE_THEN_KO", "KO_THEN_SCORE"] and combo_counts[c_type] >= 3:
            if len(final_candidates) > 10:
                continue

        final_candidates.append(cand)
        combo_counts[c_type] += 1
        if len(final_candidates) >= 20:
            break

    impact_path = os.path.join(BASE_DIR, "player_impact_events.json")
    debug_path = os.path.join(BASE_DIR, "candidate_windows_enriched_debug.json")
    out_path = os.path.join(BASE_DIR, "candidate_windows.json")

    with open(impact_path, "w") as f: json.dump(impact_events, f, indent=4)
    with open(debug_path, "w") as f: json.dump(enriched_windows, f, indent=4)
    with open(out_path, "w") as f: json.dump(final_candidates, f, indent=4)

    print(f"\n✅ [Audit] Fusion complete! Suppressed overlapping clips.")
    print(f"  • Total OCR Events Loaded: {total_ocr}")
    print(f"  • Total Impact Events: {len(impact_events)}")
    print(f"  • Total Base Candidates: {total_base_candidates}")
    print(f"  • Total Candidates after NMS & Diversity: {len(final_candidates)}")

    max_score = final_candidates[0]['importance_score'] if final_candidates else 0
    if max_score > 2000:
        print(f"  ⚠️ WARNING: Max importance score is suspiciously high ({max_score}). Check OCR noise.")
    if final_candidates and final_candidates[0]['combo_type'] == "GENERIC_ACTION":
        print(f"  ⚠️ WARNING: Top candidate is GENERIC_ACTION. Match may lack highlights.")

    print("\n📊 DIVERSITY DISTRIBUTION (Top 20):")
    for combo, count in combo_counts.items():
        print(f"  • {combo}: {count}")

    print("\n🎬 TOP 5 STORY CANDIDATES:")
    for i, cand in enumerate(final_candidates[:5]):
        flags = f" 🚩 {cand['risk_flags']}" if cand['risk_flags'] else ""
        print(f"  #{i+1} | {cand['window_id']} | {cand['start_time']}s -> {cand['end_time']}s | Score: {cand['importance_score']} | Conf: {cand['story_confidence']} | {cand['combo_type']}{flags}")

if __name__ == "__main__":
    fuse_signals()