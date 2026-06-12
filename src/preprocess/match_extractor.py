"""
Full-Match Extractor for Pokemon Unite VODs (v3).

GOAL
----
Take a long (1-3 hour) VOD and extract ONLY complete gameplay matches:
GO -> Time's Up / result screen. Lobby, queue, menus, loading and waiting
footage must never end up in the output.

WHY THE OLD APPROACHES FAILED
-----------------------------
1. Raw template matching against `gameplay_timer.png` fails because that
   template is a crop of the literal digits "07:10". The moment the clock
   shows any other time the correlation drops, while still producing
   misleading mid-range scores (timer=0.79 on non-matching frames).
2. Motion/audio peaks find 30-second highlights, not match boundaries.

HOW THIS EXTRACTOR WORKS
------------------------
Phase 1  CALIBRATION   Sample ~36 frames across the whole VOD and locate
                       where the in-game countdown timer lives on screen
                       (multi-scale template hints + white-digit-cluster
                       detection, clustered across samples). This survives
                       stream overlays / non-fullscreen gameplay layouts.
Phase 2  COARSE SCAN   One frame every few seconds. Each sample gets a
                       cheap "timer structure" score (white digit blobs on
                       a dark plate inside the calibrated ROI) - no exact
                       digit template, no full-frame OCR.
Phase 3  STATE MACHINE Hysteresis: a match starts only after the HUD is
                       stably present, and ends only after the HUD has been
                       gone for a sustained gap (so brief HUD occlusions
                       don't split a match in half).
Phase 4  VALIDATION    Targeted OCR (tiny timer ROI only, sampled frames
                       only) reads MM:SS and checks the value counts DOWN
                       at ~1 second per second across the segment. Lobbies,
                       menus, replays and queue screens can never produce a
                       coherent countdown, so this kills false positives.
                       Duration sanity bounds reject anything clip-sized.
Phase 5  REFINEMENT    Fine 1s scan snaps the start to the first HUD frame
                       (~GO) and the end to the last HUD frame, then looks
                       for the "Time's Up" / result-screen templates just
                       after HUD loss to anchor the true match end.
Phase 6  EXPORT        ffmpeg stream-copy of each validated match into
                       data/gameplay_segments/ plus debug JSON timelines
                       into data/debug/ explaining every accept/reject.

Outputs:
    data/gameplay_segments/{vod_name}_match_{NNN}_{start}s_to_{end}s.mp4
    data/debug/match_extractor_timeline_{vod_name}.json
    data/debug/match_extractor_rejected_{vod_name}.json
"""

import os
import re
import cv2
import json
import math
import subprocess
import numpy as np
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffprobe_duration(path: str) -> float:
    """Container-level duration (more reliable than frame_count/fps on VODs)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=60,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def _fmt_t(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


class MatchExtractor:
    """State-machine based full-match extractor for Pokemon Unite VODs."""

    VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov")

    def __init__(
        self,
        # --- sampling -----------------------------------------------------
        coarse_step_sec: float = 4.0,     # coarse scan sampling interval
        fine_step_sec: float = 1.0,       # boundary refinement interval
        analysis_width: int = 1280,       # frames resized to this width
        # --- state machine ------------------------------------------------
        enter_consecutive: int = 3,       # samples of HUD needed to enter IN_MATCH
        exit_gap_sec: float = 28.0,       # HUD must vanish this long to end a match
        # --- sanity bounds ------------------------------------------------
        min_match_sec: float = 300.0,     # reject anything shorter (no Shorts!)
        max_match_sec: float = 1200.0,    # reject anything longer (whole-VOD bug guard)
        # --- padding around detected boundaries ----------------------------
        start_pre_pad_sec: float = 6.0,   # catch the "GO!" drop-in
        end_post_pad_sec: float = 12.0,   # catch "Time's Up" / result splash
        # --- detection thresholds ------------------------------------------
        structure_threshold: float = 0.70,
        ocr_consistency_min: float = 0.50,  # fraction of countdown steps that must be coherent
        ocr_min_readings: int = 5,
        presence_ratio_min: float = 0.75,   # fallback gate when OCR is unavailable
        # --- layout ---------------------------------------------------------
        crop_box: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        # If gameplay only occupies part of the stream layout you can restrict
        # the search area, e.g. (0.0, 0.0, 1.0, 0.52) for "top 52% of frame".
        # Calibration usually makes this unnecessary.
        use_ocr: bool = True,
    ):
        self.templates_dir = os.path.join("assets", "templates", "pokemon_unite")
        self.output_dir = os.path.join("data", "gameplay_segments")
        self.debug_dir = os.path.join("data", "debug")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)

        self.coarse_step = float(coarse_step_sec)
        self.fine_step = float(fine_step_sec)
        self.analysis_width = int(analysis_width)
        self.enter_consecutive = int(enter_consecutive)
        self.exit_gap_sec = float(exit_gap_sec)
        self.min_match_sec = float(min_match_sec)
        self.max_match_sec = float(max_match_sec)
        self.start_pre_pad = float(start_pre_pad_sec)
        self.end_post_pad = float(end_post_pad_sec)
        self.structure_threshold = float(structure_threshold)
        self.ocr_consistency_min = float(ocr_consistency_min)
        self.ocr_min_readings = int(ocr_min_readings)
        self.presence_ratio_min = float(presence_ratio_min)
        self.crop_box = tuple(crop_box)
        self.use_ocr = bool(use_ocr)

        self.templates = self._load_templates()
        self._ocr_engine = None          # lazy: "easyocr" | "tesseract" | "none"
        self._easyocr_reader = None
        self.timer_roi: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h in analysis coords

    # ------------------------------------------------------------------ I/O

    def _load_templates(self) -> Dict[str, np.ndarray]:
        names = {
            "timer": "gameplay_timer.png",       # only used as a calibration HINT
            "time_up": "time_up.png",            # end-of-match anchor
            "result": "result_screen.png",       # end-of-match anchor
        }
        out = {}
        for key, fname in names.items():
            path = os.path.join(self.templates_dir, fname)
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    out[key] = img
            else:
                print(f"   ⚠️ [Match Extractor] Optional template missing: {path}")
        return out

    def _grab_frame(self, cap: cv2.VideoCapture, t_sec: float) -> Optional[np.ndarray]:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        scale = self.analysis_width / float(w)
        frame = cv2.resize(frame, (self.analysis_width, max(2, int(h * scale))))
        # optional layout crop (ignore webcam / chat panels)
        fh, fw = frame.shape[:2]
        x1, y1 = int(fw * self.crop_box[0]), int(fh * self.crop_box[1])
        x2, y2 = int(fw * self.crop_box[2]), int(fh * self.crop_box[3])
        return frame[y1:y2, x1:x2]

    # ------------------------------------------------- digit-cluster detection

    def _find_digit_cluster(self, gray_band: np.ndarray) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        """
        Find a horizontal cluster of bright digit-like blobs on a dark plate
        (the Unite countdown clock) inside `gray_band`.
        Returns (confidence 0..1, bbox (x,y,w,h)) in band coordinates.
        Purely structural: works for ANY time shown, unlike digit templates.
        """
        if gray_band.size == 0:
            return 0.0, None
        band_h = gray_band.shape[0]

        _, bright = cv2.threshold(gray_band, 175, 255, cv2.THRESH_BINARY)
        n, _, stats, cents = cv2.connectedComponentsWithStats(bright, connectivity=8)

        comps = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if h < 8 or h > band_h * 0.95:
                continue
            if w > h * 1.6 or area < 12:          # digits are taller than wide
                continue
            comps.append((x, y, w, h, cents[i][0], cents[i][1]))
        if len(comps) < 3:
            return 0.0, None

        # group components that sit on the same text line, close together
        comps.sort(key=lambda c: c[4])
        best_score, best_bbox = 0.0, None
        used = [False] * len(comps)
        for i in range(len(comps)):
            if used[i]:
                continue
            group = [comps[i]]
            used[i] = True
            for j in range(i + 1, len(comps)):
                if used[j]:
                    continue
                last = group[-1]
                cand = comps[j]
                h_ref = max(last[3], cand[3])
                same_line = abs(cand[5] - last[5]) < h_ref * 0.6
                close = (cand[0] - (last[0] + last[2])) < h_ref * 1.2
                similar = 0.4 < (cand[3] / max(1, last[3])) < 2.5
                if same_line and close and similar:
                    group.append(cand)
                    used[j] = True
            if not (3 <= len(group) <= 7):        # M:SS .. MM:SS (+colon blobs)
                continue
            gx1 = min(g[0] for g in group)
            gy1 = min(g[1] for g in group)
            gx2 = max(g[0] + g[2] for g in group)
            gy2 = max(g[1] + g[3] for g in group)
            gw, gh = gx2 - gx1, gy2 - gy1
            if gh == 0:
                continue
            aspect = gw / float(gh)
            if not (1.3 <= aspect <= 5.0):        # "10:00" is wide-ish
                continue
            # plate behind digits should be dark
            pad = max(2, gh // 4)
            py1, py2 = max(0, gy1 - pad), min(gray_band.shape[0], gy2 + pad)
            px1, px2 = max(0, gx1 - pad), min(gray_band.shape[1], gx2 + pad)
            plate = gray_band[py1:py2, px1:px2]
            mask = bright[py1:py2, px1:px2] == 0
            bg = float(np.median(plate[mask])) if mask.any() else 255.0

            score = 0.5                                          # plausible cluster found
            score += 0.25 if bg < 120 else (0.1 if bg < 150 else 0.0)
            score += 0.25 if 4 <= len(group) <= 6 else 0.1       # 4 digits (+colon)
            if score > best_score:
                best_score, best_bbox = score, (gx1, gy1, gw, gh)
        return best_score, best_bbox

    # ----------------------------------------------------------- calibration

    def _calibrate_timer_roi(self, cap: cv2.VideoCapture, duration: float, vod_name: str) -> bool:
        """
        Locate the countdown timer once per VOD by voting across ~36 frames.
        Uses BOTH the (weak) timer template as a position hint and the
        structural digit-cluster detector, then picks the modal location.
        """
        print("   🔭 [Calibration] Locating the in-game countdown timer...")
        n_samples = 36
        ts = np.linspace(duration * 0.03, duration * 0.97, n_samples)
        struct_hits: List[Tuple[float, float, float]] = []   # (cx, cy, weight)
        struct_bboxes: List[Tuple[int, int, int, int]] = []
        tpl_hits: List[Tuple[float, float, float]] = []
        tpl_bboxes: List[Tuple[int, int, int, int]] = []
        sample_frame = None

        timer_tpl = self.templates.get("timer")
        for t in ts:
            frame = self._grab_frame(cap, float(t))
            if frame is None:
                continue
            sample_frame = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            band = gray[: int(gray.shape[0] * 0.50), :]   # timer lives in upper half

            # 1) structural detection (time-agnostic) - primary signal
            score, bbox = self._find_digit_cluster(band)
            if bbox is not None and score >= 0.7:
                x, y, w, h = bbox
                struct_hits.append((x + w / 2.0, y + h / 2.0, score))
                struct_bboxes.append(bbox)

            # 2) template hint (multi-scale, low threshold) - fallback only,
            #    since the template contains literal digits and is unreliable
            if timer_tpl is not None:
                for s in (0.6, 0.8, 1.0, 1.25, 1.55, 1.9):
                    tpl = cv2.resize(timer_tpl, None, fx=s, fy=s)
                    th, tw = tpl.shape[:2]
                    if th >= band.shape[0] or tw >= band.shape[1]:
                        continue
                    res = cv2.matchTemplate(band, tpl, cv2.TM_CCOEFF_NORMED)
                    _, mx, _, loc = cv2.minMaxLoc(res)
                    if mx >= 0.55:
                        tpl_hits.append((loc[0] + tw / 2.0, loc[1] + th / 2.0, mx * 0.5))
                        tpl_bboxes.append((loc[0], loc[1], tw, th))

        # prefer pure structural evidence; fall back to template hints
        if len(struct_hits) >= 6:
            hits, bboxes = struct_hits, struct_bboxes
        else:
            hits = struct_hits + tpl_hits
            bboxes = struct_bboxes + tpl_bboxes

        if not hits:
            print("   ⚠️ [Calibration] No stable timer location found.")
            print("      Falling back to scanning the whole top band each frame (slower, less precise).")
            self.timer_roi = None
            return False

        # vote: cluster hit centres, keep the heaviest cluster
        clusters: List[Dict] = []
        for cx, cy, wgt in hits:
            placed = False
            for c in clusters:
                if abs(c["cx"] - cx) < 28 and abs(c["cy"] - cy) < 20:
                    tot = c["w"] + wgt
                    c["cx"] = (c["cx"] * c["w"] + cx * wgt) / tot
                    c["cy"] = (c["cy"] * c["w"] + cy * wgt) / tot
                    c["w"] = tot
                    c["n"] += 1
                    placed = True
                    break
            if not placed:
                clusters.append({"cx": cx, "cy": cy, "w": wgt, "n": 1})
        best = max(clusters, key=lambda c: c["w"])
        if best["n"] < 4:
            print(f"   ⚠️ [Calibration] Timer location too unstable ({best['n']} votes). Using band-scan fallback.")
            self.timer_roi = None
            return False

        # median bbox size among hits near the winning cluster
        near = [b for b in bboxes
                if abs((b[0] + b[2] / 2) - best["cx"]) < 30 and abs((b[1] + b[3] / 2) - best["cy"]) < 22]
        med_w = int(np.median([b[2] for b in near])) if near else 70
        med_h = int(np.median([b[3] for b in near])) if near else 24

        # expand with a generous margin so digit changes / glow / slight
        # mis-centring never clip the digits out of the ROI
        mx, my = int(med_w * 0.55), int(med_h * 1.1)
        x = max(0, int(best["cx"] - med_w / 2) - mx)
        y = max(0, int(best["cy"] - med_h / 2) - my)
        w = med_w + 2 * mx
        h = med_h + 2 * my
        self.timer_roi = (x, y, w, h)
        print(f"   ✅ [Calibration] Timer locked at ROI x={x} y={y} w={w} h={h} "
              f"({best['n']} supporting samples).")

        if sample_frame is not None:
            dbg = sample_frame.copy()
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(self.debug_dir, f"{vod_name}_timer_roi_debug.jpg"), dbg)
        return True

    # ------------------------------------------------------------- presence

    def _timer_roi_crop(self, gray: np.ndarray) -> np.ndarray:
        if self.timer_roi is not None:
            x, y, w, h = self.timer_roi
            return gray[y:y + h, x:x + w]
        return gray[: int(gray.shape[0] * 0.50), :]   # fallback: whole top band

    def _presence_score(self, gray: np.ndarray) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        roi = self._timer_roi_crop(gray)
        return self._find_digit_cluster(roi)

    # ------------------------------------------------------------------ OCR

    def _init_ocr(self):
        if self._ocr_engine is not None:
            return
        if not self.use_ocr:
            self._ocr_engine = "none"
            return
        try:
            import easyocr  # noqa
            self._easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            self._ocr_engine = "easyocr"
            print("   🔤 [OCR] Using easyocr for countdown validation.")
            return
        except Exception:
            pass
        try:
            import pytesseract  # noqa
            pytesseract.get_tesseract_version()
            self._ocr_engine = "tesseract"
            print("   🔤 [OCR] Using pytesseract for countdown validation.")
            return
        except Exception:
            self._ocr_engine = "none"
            print("   ⚠️ [OCR] No OCR backend available (easyocr/pytesseract).")
            print("      Falling back to structural validation only - install easyocr for best accuracy.")

    _TIME_RE = re.compile(r"(\d{1,2})\s*[:;.]\s*(\d{2})")

    def _read_timer_seconds(self, gray: np.ndarray) -> Optional[int]:
        """Targeted OCR on the tiny timer ROI only. Returns remaining seconds."""
        self._init_ocr()
        if self._ocr_engine == "none":
            return None
        roi = self._timer_roi_crop(gray)
        if roi.size == 0:
            return None
        # upscale + binarize for crisp digits
        scale = max(1.0, 48.0 / roi.shape[0])
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, binimg = cv2.threshold(roi, 165, 255, cv2.THRESH_BINARY)

        text = ""
        try:
            if self._ocr_engine == "easyocr":
                res = self._easyocr_reader.readtext(binimg, allowlist="0123456789:", detail=0)
                text = " ".join(res)
            else:
                import pytesseract
                inv = cv2.bitwise_not(binimg)  # tesseract prefers dark text on white
                text = pytesseract.image_to_string(
                    inv, config="--psm 7 -c tessedit_char_whitelist=0123456789:")
        except Exception:
            return None

        m = self._TIME_RE.search(text)
        if not m:
            return None
        mins, secs = int(m.group(1)), int(m.group(2))
        if secs > 59 or mins > 20:
            return None
        return mins * 60 + secs

    # ------------------------------------------------------------ validation

    def _validate_countdown(self, readings: List[Tuple[float, int]]) -> Dict:
        """
        readings: list of (vod_time, remaining_seconds) within one segment.
        A real Unite match counts DOWN at ~1 sec/sec. Check pairwise steps.
        """
        if len(readings) < self.ocr_min_readings:
            return {"method": "ocr", "ok": False, "reason": f"only {len(readings)} timer readings",
                    "consistency": 0.0, "readings": len(readings)}
        good, total = 0, 0
        for (t1, r1), (t2, r2) in zip(readings, readings[1:]):
            dt = t2 - t1
            if dt <= 0 or dt > 120:
                continue
            total += 1
            actual_drop = r1 - r2
            if abs(actual_drop - dt) <= max(2.0, 0.25 * dt):
                good += 1
        consistency = good / total if total else 0.0

        # global drop-rate check: across the whole segment the clock must lose
        # ~1 second per second (median-of-edges to be robust to OCR misreads)
        k = min(3, len(readings) // 2)
        t_first = float(np.median([t for t, _ in readings[:k]]))
        r_first = float(np.median([r for _, r in readings[:k]]))
        t_last = float(np.median([t for t, _ in readings[-k:]]))
        r_last = float(np.median([r for _, r in readings[-k:]]))
        span = t_last - t_first
        slope = (r_first - r_last) / span if span > 0 else 0.0
        slope_ok = 0.6 <= slope <= 1.4

        ok = consistency >= self.ocr_consistency_min and slope_ok
        if ok:
            reason = "countdown coherent"
        elif not slope_ok:
            reason = f"timer not counting down at 1s/s (drop rate {slope:.2f})"
        else:
            reason = f"countdown incoherent ({consistency:.0%} of steps)"
        return {
            "method": "ocr", "ok": ok, "reason": reason,
            "consistency": round(consistency, 3),
            "drop_rate": round(slope, 3),
            "readings": len(readings),
            "max_remaining": max(r for _, r in readings),
            "min_remaining": min(r for _, r in readings),
        }

    # ------------------------------------------------------------ refinement

    def _refine_boundary(self, cap: cv2.VideoCapture, t_lo: float, t_hi: float,
                         find_first_present: bool) -> float:
        """Fine 1s scan in [t_lo, t_hi] to snap to first/last HUD-present frame."""
        last_hit = None
        t = t_lo
        while t <= t_hi:
            frame = self._grab_frame(cap, t)
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score, _ = self._presence_score(gray)
                if score >= self.structure_threshold:
                    if find_first_present:
                        return t
                    last_hit = t
            t += self.fine_step
        if find_first_present:
            return t_lo
        return last_hit if last_hit is not None else t_lo

    def _find_end_anchor(self, cap: cv2.VideoCapture, t_lo: float, t_hi: float) -> Optional[float]:
        """Look for the Time's Up / result-screen templates just after HUD loss."""
        best_t, best_score = None, 0.0
        for key in ("time_up", "result"):
            tpl0 = self.templates.get(key)
            if tpl0 is None:
                continue
            t = t_lo
            while t <= t_hi:
                frame = self._grab_frame(cap, t)
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    for s in (0.6, 0.85, 1.1, 1.4):
                        tpl = cv2.resize(tpl0, None, fx=s, fy=s)
                        if tpl.shape[0] >= gray.shape[0] or tpl.shape[1] >= gray.shape[1]:
                            continue
                        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
                        _, mx, _, _ = cv2.minMaxLoc(res)
                        if mx > best_score:
                            best_score, best_t = mx, t
                t += 2.0
        return best_t if best_score >= 0.62 else None

    # ---------------------------------------------------------------- export

    def _cut_match(self, vod_path: str, start: float, end: float, out_path: str) -> bool:
        dur = end - start
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{start:.2f}",
            "-i", vod_path,
            "-t", f"{dur:.2f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            out_path,
        ]
        r = subprocess.run(cmd)
        return r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0

    # ------------------------------------------------------------------ main

    def extract_matches(self, vod_path: str) -> List[str]:
        if not os.path.exists(vod_path):
            print(f"❌ VOD not found: {vod_path}")
            return []

        vod_name = os.path.splitext(os.path.basename(vod_path))[0]
        print(f"\n🎮 [Match Extractor] Analyzing VOD: {vod_path}")

        duration = _ffprobe_duration(vod_path)
        cap = cv2.VideoCapture(vod_path)
        if duration <= 0:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            duration = frames / fps if fps > 0 else 0
        if duration <= 0:
            print("❌ Could not determine VOD duration.")
            cap.release()
            return []
        print(f"   ⏱️ Duration: {_fmt_t(duration)}")

        # ---- Phase 1: calibration ----------------------------------------
        self._calibrate_timer_roi(cap, duration, vod_name)

        # ---- Phase 2: coarse scan -----------------------------------------
        print(f"   🔍 [Coarse Scan] 1 frame every {self.coarse_step:.0f}s "
              f"(timer-structure check on a tiny ROI)...")
        timeline: List[Dict] = []
        n_steps = int(duration // self.coarse_step)
        report_every = max(1, n_steps // 12)
        for i in range(n_steps + 1):
            t = i * self.coarse_step
            frame = self._grab_frame(cap, t)
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score, bbox = self._presence_score(gray)
            present = score >= self.structure_threshold
            entry = {"t": round(t, 1), "score": round(float(score), 2), "present": bool(present)}
            if present and self.use_ocr:
                rem = self._read_timer_seconds(gray)
                if rem is not None:
                    entry["timer_remaining"] = rem
            timeline.append(entry)
            if i % report_every == 0:
                print(f"      ... {100.0 * t / duration:5.1f}% scanned "
                      f"({_fmt_t(t)})", flush=True)

        # ---- Phase 3: hysteresis state machine -----------------------------
        exit_gap_samples = max(2, int(math.ceil(self.exit_gap_sec / self.coarse_step)))
        segments: List[Dict] = []
        in_match = False
        consec_in = 0
        consec_out = 0
        seg_start_idx = 0
        last_present_idx = 0

        for idx, e in enumerate(timeline):
            if e["present"]:
                consec_in += 1
                consec_out = 0
                last_present_idx = idx
                if not in_match and consec_in >= self.enter_consecutive:
                    in_match = True
                    seg_start_idx = idx - (self.enter_consecutive - 1)
                    print(f"   ⚔️ HUD stable -> candidate match starting near "
                          f"{_fmt_t(timeline[seg_start_idx]['t'])}")
            else:
                consec_out += 1
                consec_in = 0
                if in_match and consec_out >= exit_gap_samples:
                    in_match = False
                    segments.append({"start_idx": seg_start_idx, "end_idx": last_present_idx})
                    print(f"   🛑 HUD gone for {self.exit_gap_sec:.0f}s -> candidate ends near "
                          f"{_fmt_t(timeline[last_present_idx]['t'])}")
        if in_match:
            segments.append({"start_idx": seg_start_idx, "end_idx": last_present_idx})

        # ---- Phase 4 + 5: validate & refine ---------------------------------
        accepted: List[Dict] = []
        rejected: List[Dict] = []
        for seg in segments:
            s_t = timeline[seg["start_idx"]]["t"]
            e_t = timeline[seg["end_idx"]]["t"]
            raw_dur = e_t - s_t
            entries = timeline[seg["start_idx"]: seg["end_idx"] + 1]
            presence_ratio = sum(1 for e in entries if e["present"]) / max(1, len(entries))
            readings = [(e["t"], e["timer_remaining"]) for e in entries if "timer_remaining" in e]

            info = {
                "raw_start": s_t, "raw_end": e_t, "raw_duration": round(raw_dur, 1),
                "presence_ratio": round(presence_ratio, 3),
                "ocr_readings": len(readings),
            }

            # duration sanity first (kills clip-sized and whole-VOD segments)
            if raw_dur < self.min_match_sec:
                info["reject_reason"] = (f"too short ({raw_dur / 60:.1f} min < "
                                         f"{self.min_match_sec / 60:.0f} min) - not a full match")
                rejected.append(info)
                print(f"   ⚠️ Rejected segment {_fmt_t(s_t)}-{_fmt_t(e_t)}: {info['reject_reason']}")
                continue
            if raw_dur > self.max_match_sec:
                info["reject_reason"] = (f"too long ({raw_dur / 60:.1f} min > "
                                         f"{self.max_match_sec / 60:.0f} min) - detector likely stuck")
                rejected.append(info)
                print(f"   ⚠️ Rejected segment {_fmt_t(s_t)}-{_fmt_t(e_t)}: {info['reject_reason']}")
                continue

            # countdown validation (the strong gameplay proof)
            if readings:
                val = self._validate_countdown(readings)
            else:
                ok = presence_ratio >= self.presence_ratio_min
                val = {"method": "structure_only", "ok": ok,
                       "reason": ("stable HUD presence" if ok else
                                  f"HUD presence too sparse ({presence_ratio:.0%})")}
            info["validation"] = val
            if not val["ok"]:
                info["reject_reason"] = f"validation failed: {val['reason']}"
                rejected.append(info)
                print(f"   ⚠️ Rejected segment {_fmt_t(s_t)}-{_fmt_t(e_t)}: {info['reject_reason']}")
                continue

            # refine boundaries with a fine 1s scan
            print(f"   🎯 Refining boundaries for match {_fmt_t(s_t)}-{_fmt_t(e_t)}...")
            hud_start = self._refine_boundary(
                cap, max(0.0, s_t - self.coarse_step * (self.enter_consecutive + 2)),
                s_t + self.coarse_step, find_first_present=True)
            hud_end = self._refine_boundary(
                cap, e_t - self.coarse_step, min(duration, e_t + self.exit_gap_sec),
                find_first_present=False)
            anchor = self._find_end_anchor(cap, hud_end - 4.0,
                                           min(duration, hud_end + self.exit_gap_sec))
            if anchor is not None:
                info["end_anchor"] = round(anchor, 1)
                hud_end = max(hud_end, anchor)

            final_start = max(0.0, hud_start - self.start_pre_pad)
            final_end = min(duration, hud_end + self.end_post_pad)
            info.update({"final_start": round(final_start, 1),
                         "final_end": round(final_end, 1),
                         "final_duration": round(final_end - final_start, 1)})
            accepted.append(info)

        # ---- Phase 6: export -------------------------------------------------
        outputs: List[str] = []
        print(f"\n✂️ [Match Extractor] {len(accepted)} validated full match(es). Slicing...")
        for i, m in enumerate(accepted, 1):
            out_name = (f"{vod_name}_match_{i:03d}_"
                        f"{int(m['final_start'])}s_to_{int(m['final_end'])}s.mp4")
            out_path = os.path.join(self.output_dir, out_name)
            m["output_file"] = out_path
            if os.path.exists(out_path):
                print(f"   ⏩ Match {i} already extracted, skipping.")
                outputs.append(out_path)
                continue
            print(f"   🎬 Match {i}: {_fmt_t(m['final_start'])} -> {_fmt_t(m['final_end'])} "
                  f"({m['final_duration'] / 60:.1f} min)")
            if self._cut_match(vod_path, m["final_start"], m["final_end"], out_path):
                outputs.append(out_path)
            else:
                print(f"   ❌ ffmpeg failed for match {i}")
                m["export_error"] = True

        cap.release()

        # ---- debug artifacts ---------------------------------------------------
        timeline_path = os.path.join(self.debug_dir, f"match_extractor_timeline_{vod_name}.json")
        with open(timeline_path, "w") as f:
            json.dump({
                "vod": vod_path,
                "duration_sec": round(duration, 1),
                "params": {
                    "coarse_step_sec": self.coarse_step,
                    "enter_consecutive": self.enter_consecutive,
                    "exit_gap_sec": self.exit_gap_sec,
                    "min_match_sec": self.min_match_sec,
                    "max_match_sec": self.max_match_sec,
                    "structure_threshold": self.structure_threshold,
                    "timer_roi": self.timer_roi,
                    "ocr_engine": self._ocr_engine or "uninitialized",
                },
                "accepted_matches": accepted,
                "samples": timeline,
            }, f, indent=2)
        rejected_path = os.path.join(self.debug_dir, f"match_extractor_rejected_{vod_name}.json")
        with open(rejected_path, "w") as f:
            json.dump(rejected, f, indent=2)

        print(f"\n🎉 Done. {len(outputs)} full match(es) saved to {self.output_dir}/")
        print(f"   🧾 Debug timeline : {timeline_path}")
        print(f"   🧾 Rejected log   : {rejected_path}")
        return outputs


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        MatchExtractor().extract_matches(sys.argv[1])
    else:
        print("Usage: python -m src.preprocess.match_extractor <path_to_vod>")
        print("Or run via: python run_vod_pipeline.py")
