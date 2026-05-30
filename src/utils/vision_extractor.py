import cv2
import base64
from typing import List

def extract_frames_for_llm(video_path: str, start_time: float, end_time: float, num_frames: int = 12) -> List[str]:
    """
    Physically opens a video file, navigates to the exact timestamps, 
    extracts evenly spaced frames, and converts them to base64 strings for Qwen Vision.
    """
    print(f"[Vision Extractor] Slicing {num_frames} frames from {start_time}s to {end_time}s...")
    
    # 1. Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    # 2. Figure out the video's framerate (FPS) to calculate exact frame math
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0 # Fallback just in case the video metadata is corrupted
        
    # 3. Convert seconds to exact frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames_in_window = end_frame - start_frame
    
    # Ensure we don't crash on microscopic clips
    if total_frames_in_window <= 0:
        total_frames_in_window = num_frames
        
    # Calculate how many frames to skip to get exactly `num_frames` (e.g., 12 frames)
    step = max(1, total_frames_in_window // num_frames)
    
    base64_frames = []
    
    # 4. Loop through and grab the exact frames
    for i in range(num_frames):
        target_frame = start_frame + (i * step)
        
        # Tell OpenCV to jump directly to this exact frame in the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # 5. OPTIMIZATION: Resize the frame! 
        # Sending 12 full 1080p images to Qwen will crash the API and cost a fortune.
        # We resize it to a smaller width (720px) which is plenty for the AI to understand the context.
        height, width = frame.shape[:2]
        new_width = 720
        new_height = int((new_width / width) * height)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # 6. Compress to JPEG and convert to a Base64 string so the JSON payload accepts it
        _, buffer = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_string = base64.b64encode(buffer).decode('utf-8')
        base64_frames.append(b64_string)
        
    # Always clean up your memory!
    cap.release()
    
    if not base64_frames:
        raise RuntimeError(f"Failed to extract any frames from {video_path} between {start_time}s and {end_time}s.")
        
    print(f"         ✅ Successfully processed and encoded {len(base64_frames)} frames.")
    return base64_frames