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
    with open(out_path, "w") as f:
        json.dump(significant_spikes, f, indent=4)

    print(f"✅ Audio extraction complete. Found {len(significant_spikes)} massive hype moments.")
    print(f"📁 Memory saved to {out_path}.")

# Run it!
extract_audio_hype('/content/drive/MyDrive/VS_Factory/inputs/PokemonUnite.mp4')
