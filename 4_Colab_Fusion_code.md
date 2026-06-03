import json
import os

def parse_motion_score(raw_text):
    try:
        return float(raw_text.split(" | ")[0].replace("Chaos Score: ", ""))
    except Exception:
        return 0.0

def apply_nms(candidates, overlap_threshold=0.30):
    """
    Non-Maximum Suppression for temporal 1D data.
    Removes lower-scoring clips that overlap heavily with higher-scoring ones.
    """
    selected = []
    for candidate in candidates:
        keep = True
        c_start = candidate["start_time"]
        c_end = candidate["end_time"]
        c_duration = c_end - c_start

        for s in selected:
            s_start = s["start_time"]
            s_end = s["end_time"]

            # Calculate overlap in seconds
            overlap_start = max(c_start, s_start)
            overlap_end = min(c_end, s_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            # If it overlaps by more than 30% of its own length, suppress it
            if (overlap_duration / c_duration) > overlap_threshold:
                keep = False
                break

        if keep:
            selected.append(candidate)

    return selected

def fuse_signals():
    print("⚙️ [Phase 1.5] Booting Unified Signal Fusion Layer...")

    ocr_path = "data/inputs/ocr_structured_log.json"
    audio_path = "data/inputs/audio_hype_log.json"
    motion_path = "data/inputs/motion_events_log.json"

    if not os.path.exists(ocr_path) or not os.path.exists(audio_path):
        print("❌ [Error] Missing core sensor logs.")
        return

    with open(ocr_path, "r") as f: ocr_events = json.load(f)
    with open(audio_path, "r") as f: audio_events = json.load(f)

    motion_events = []
    if os.path.exists(motion_path):
        with open(motion_path, "r") as f: motion_events = json.load(f)

    all_events = []
    for e in ocr_events: all_events.append({"time": e["timestamp"], "type": "OCR", "data": e})
    for e in audio_events: all_events.append({"time": e["timestamp"], "type": "AUDIO", "data": e})
    for e in motion_events: all_events.append({"time": e["timestamp"], "type": "MOTION", "data": e})

    all_events.sort(key=lambda x: x["time"])

    CLUSTER_GAP = 5.0
    clusters = []
    if all_events:
        current_cluster = [all_events[0]]
        for event in all_events[1:]:
            if event["time"] - current_cluster[-1]["time"] <= CLUSTER_GAP:
                current_cluster.append(event)
            else:
                clusters.append(current_cluster)
                current_cluster = [event]
        clusters.append(current_cluster)

    candidate_windows = []

    OCR_BONUS = {
        "PENTA KO": 200, "QUADRA KO": 120, "TRIPLE KO": 80, "DOUBLE KO": 50,
        "OBJECTIVE": 150, "GOAL": 25, "KO": 5, "PLAYER_DEATH": -20
    }

    for idx, cluster in enumerate(clusters):
        start_time = cluster[0]["time"]
        end_time = cluster[-1]["time"]

        c_ocr = [e["data"] for e in cluster if e["type"] == "OCR"]
        c_audio = [e["data"] for e in cluster if e["type"] == "AUDIO"]
        c_motion = [e["data"] for e in cluster if e["type"] == "MOTION"]

        audio_density = len(c_audio)
        motion_density = len(c_motion)

        peak_volume = max([float(e["raw_text"].replace("Volume Level: ", "")) for e in c_audio]) if c_audio else 0
        ocr_labels = [e.get("canonical_text", "UNKNOWN") for e in c_ocr]

        ocr_semantic_score = sum([OCR_BONUS.get(e.get("canonical_text", ""), 0) for e in c_ocr])
        motion_score_sum = sum([parse_motion_score(m["raw_text"]) for m in c_motion])

        raw_score = (audio_density * 2.0) + (motion_score_sum * 0.15) + (peak_volume / 1000) + ocr_semantic_score

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
                "motion_density": motion_density
            })

    # Sort strictly by score before passing to NMS
    candidate_windows.sort(key=lambda x: x["importance_score"], reverse=True)

    # Apply Non-Maximum Suppression to deduplicate the timeline
    deduplicated_windows = apply_nms(candidate_windows)

    out_path = "data/inputs/candidate_windows.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        # Save only the top 10 unique candidates
        json.dump(deduplicated_windows[:10], f, indent=4)

    print(f"✅ Fusion complete! Suppressed overlapping clips. Top {len(deduplicated_windows[:10])} unique candidates saved to {out_path}.")

if __name__ == "__main__":
    fuse_signals()
