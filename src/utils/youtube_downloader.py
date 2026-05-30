import os
import yt_dlp

def fetch_youtube_video(url: str, output_dir: str = "data/inputs") -> str:
    """
    Downloads a YouTube video/stream VOD at the best available mp4 quality.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # We use the YouTube video ID as the filename to avoid weird character crashes
    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_template,
        'quiet': False,
        'no_warnings': True,
    }
    
    print(f"\n[Ingestion Engine] Fetching YouTube Video: {url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            final_filename = ydl.prepare_filename(info_dict)
            
        print(f"[Ingestion Engine] ✅ Download complete: {final_filename}")
        return final_filename
    except Exception as e:
        raise RuntimeError(f"Failed to download YouTube video: {e}")