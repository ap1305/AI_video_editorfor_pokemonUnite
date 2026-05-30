import os
import numpy as np
from scipy.io import wavfile
import ffmpeg
from typing import List, Tuple

class AudioHighlightDetector:
    def __init__(self, temp_dir: str = "data/temp_audio"):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def find_action_peaks(self, video_path: str, num_clips: int = 5, clip_duration: int = 15) -> List[Tuple[float, float]]:
        """
        Scans a video's audio track to find the loudest moments (team fights, shouting).
        Returns a list of (start_time, end_time) tuples for the best clips.
        """
        temp_wav = os.path.join(self.temp_dir, "analysis_track.wav")
        
        # 1. Extract audio quickly and downsample it for fast analysis
        print(f"[Audio Engine] Extracting audio from {os.path.basename(video_path)} for spike analysis...")
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
        try:
            (
                ffmpeg
                .input(video_path)
                .output(temp_wav, acodec='pcm_s16le', ac=1, ar='16k')
                .run(overwrite_output=True, quiet=True)
            )
        except ffmpeg.Error as e:
            print(f"[Audio Engine Error] Failed to extract audio: {e.stderr.decode('utf8') if e.stderr else str(e)}")
            return []

        # 2. Read the audio data mathematically
        sample_rate, data = wavfile.read(temp_wav)
        
        # 3. Calculate volume (RMS) per second
        samples_per_sec = sample_rate
        total_seconds = len(data) // samples_per_sec
        volume_per_second = []

        for sec in range(total_seconds):
            start_idx = sec * samples_per_sec
            end_idx = start_idx + samples_per_sec
            # Calculate the Root Mean Square (RMS) to find the true loudness
            chunk = data[start_idx:end_idx].astype(np.float64)
            rms = np.sqrt(np.mean(chunk**2))
            volume_per_second.append(rms)

        # 4. Find the highest volume peaks
        volume_array = np.array(volume_per_second)
        peaks = []
        
        # We don't want 5 clips from the exact same 20-second team fight.
        # This loop finds the loudest second, saves it, and then "mutes" the surrounding area 
        # so it is forced to find the NEXT big fight in the video.
        for _ in range(num_clips):
            if len(volume_array) == 0 or np.max(volume_array) == 0:
                break
                
            loudest_sec = np.argmax(volume_array)
            
            # Create the 15-second window (Start 5 seconds before the peak, end 10 seconds after)
            start_time = max(0.0, float(loudest_sec - 5))
            end_time = min(float(total_seconds), start_time + clip_duration)
            
            peaks.append((start_time, end_time))
            
            # "Mute" this section so we don't pick it again
            mute_start = max(0, loudest_sec - clip_duration)
            mute_end = min(total_seconds, loudest_sec + clip_duration)
            volume_array[mute_start:mute_end] = 0

        # Sort the clips chronologically
        peaks.sort(key=lambda x: x[0])
        
        # Cleanup temp file
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
        print(f"[Audio Engine] Successfully isolated the {len(peaks)} loudest action sequences.")
        return peaks