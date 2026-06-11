import os
import cv2
import json
import subprocess
import tempfile
import numpy as np
from scipy.signal import find_peaks
from scipy.io import wavfile

class VODMomentScanner:
    def __init__(self):
        self.inputs_dir = "data/inputs"
        os.makedirs(self.inputs_dir, exist_ok=True)
        
        self.motion_weight = 0.75
        self.audio_weight = 0.25
        self.window_duration_sec = 30.0
        self.max_candidates = 30

    def _extract_audio_energy(self, vod_path: str, duration_sec: int) -> np.ndarray:
        print("   [Audio] Extracting raw audio stream for energy analysis...")
        fd, temp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", vod_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_wav]
        
        try:
            subprocess.run(cmd, check=True)
            sample_rate, data = wavfile.read(temp_wav)
            data = data.astype(np.float32)
            expected_samples = duration_sec * sample_rate
            
            if len(data) < expected_samples:
                data = np.pad(data, (0, expected_samples - len(data)))
            else:
                data = data[:expected_samples]
                
            chunks = data.reshape(duration_sec, sample_rate)
            energy = np.sqrt(np.mean(chunks**2, axis=1))
            if np.max(energy) > 0: energy = energy / np.max(energy)
            return energy
        except Exception as e:
            print(f"   ⚠️ [Audio] Extraction failed: {e}. Defaulting to zero.")
            return np.zeros(duration_sec)
        finally:
            if os.path.exists(temp_wav): os.remove(temp_wav)

    def _extract_motion_density(self, vod_path: str, fps: float, duration_sec: int) -> np.ndarray:
        print("   [Motion] Scanning motion density (1 frame/sec timestamp seek)...")
        cap = cv2.VideoCapture(vod_path)
        motion = np.zeros(duration_sec)
        prev_gray = None

        for sec_index in range(duration_sec):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec_index * 1000)
            ret, frame = cap.read()
            if not ret: continue

            small_frame = cv2.resize(frame, (640, 360))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion[sec_index] = np.sum(thresh)

            prev_gray = gray
            if sec_index % max(1, (duration_sec // 10)) == 0:
                print(f"      ... {(sec_index / duration_sec) * 100:.0f}% scanned")

        cap.release()
        if np.max(motion) > 0: motion = motion / np.max(motion)
        return motion

    def scan_vod(self, vod_path: str):
        print(f"\n🔍 [VOD Moment Scanner] Analyzing: {vod_path}")
        cap = cv2.VideoCapture(vod_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_sec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) if fps > 0 else 0
        cap.release()

        audio_scores = self._extract_audio_energy(vod_path, duration_sec)
        motion_scores = self._extract_motion_density(vod_path, fps, duration_sec)
        
        combined_scores = (self.motion_weight * motion_scores) + (self.audio_weight * audio_scores)
        smoothed_scores = np.convolve(combined_scores, np.ones(5) / 5, mode='same')
        
        peaks, _ = find_peaks(smoothed_scores, height=np.mean(smoothed_scores), distance=30)
        
        raw_windows = []
        half_w = self.window_duration_sec / 2.0
        
        for p in peaks:
            if motion_scores[p] < 0.05: continue
            raw_windows.append({
                "peak": float(p),
                "start": max(0.0, float(p - half_w)),
                "end": min(float(duration_sec), float(p + half_w)),
                "score": float(smoothed_scores[p] * 100),
                "m_dense": float(motion_scores[p] * 100),
                "a_dense": float(audio_scores[p] * 100)
            })

        raw_windows.sort(key=lambda x: x["score"], reverse=True)
        
        candidates = []
        for idx, w in enumerate(raw_windows[:self.max_candidates], 1):
            candidates.append({
                "window_id": f"RAW_CANDIDATE_{str(idx).zfill(3)}",
                "start_time": round(w["start"], 1),
                "end_time": round(w["end"], 1),
                "peak_time": round(w["peak"], 1),
                "importance_score": round(w["score"], 1),
                "motion_density": round(w["m_dense"], 1),
                "audio_density": round(w["a_dense"], 1),
                "status": "unverified",
                "source_vod": vod_path
            })
            
        candidates.sort(key=lambda x: x["start_time"])
        
        vod_name = os.path.splitext(os.path.basename(vod_path))[0]
        output_file = os.path.join(self.inputs_dir, f"candidate_windows_raw_{vod_name}.json")
        
        with open(output_file, 'w') as f: 
            json.dump(candidates, f, indent=2)
            
        print(f"✅ Generated {len(candidates)} raw candidates with peak_time anchors in {output_file}")
        print("ℹ️ Rough preview cutting skipped. Run Verifier next.")