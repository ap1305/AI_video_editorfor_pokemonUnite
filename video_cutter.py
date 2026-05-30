import chromadb
from transformers import AutoProcessor, AutoModel
import torch
import subprocess
import os

def run_highlight_extraction():
    # --- 1. SEARCH THE LOCAL DATABASE ---
    print("🔍 Loading SigLIP to process your text query...")
    # (This runs locally on your PC. It might take a few seconds the first time to download the model, but it's very small)
    model_id = "google/siglip-base-patch16-224"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()

    # Point to the database folder you just downloaded and placed!
    db_path = "data/chroma_db"
    
    if not os.path.exists(db_path):
         print(f"🚨 ERROR: Cannot find database at {db_path}. Did you put the downloaded folder in the right spot?")
         return

    chroma_client = chromadb.PersistentClient(path=db_path)
    video_index = chroma_client.get_collection(name="video_timeline_index")

    query_text = "A massive team fight with flashing abilities and low health bars near the center objective"
    print(f"🔍 Searching local database for: '{query_text}'...")

    # Vectorize the text locally
    inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_outputs = model.get_text_features(**inputs)
        
        # Squeeze out the extra dimensions to get a clean 1D vector
        if hasattr(text_outputs, "text_embeds"):
            text_tensor = text_outputs.text_embeds
        elif hasattr(text_outputs, "pooler_output"):
            text_tensor = text_outputs.pooler_output
        else:
            text_tensor = text_outputs
            
        text_vector = text_tensor.squeeze().numpy().tolist()

    # Query ChromaDB
    results = video_index.query(
        query_embeddings=[text_vector],
        n_results=1 # Just get the absolute best match
    )

    metadata = results['metadatas'][0][0]
    start_time = metadata['start']
    end_time = metadata['end']
    
    print(f"🎯 Best Match Found: {start_time}s to {end_time}s")

    # --- 2. CUT THE VIDEO WITH FFMPEG ---
    input_video = "data/inputs/PokemonUnite.mp4" # Make sure your raw video is here!
    
    if not os.path.exists(input_video):
         print(f"🚨 ERROR: Cannot find raw video at {input_video}.")
         return

    duration = end_time - start_time
    output_path = "data/output/epic_highlight_01.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    command = [
        "ffmpeg", "-y", 
        "-ss", str(start_time), 
        "-i", input_video, 
        "-t", str(duration), 
        "-c", "copy", 
        output_path
    ]
    
    print(f"✂️ Slicing video locally via FFmpeg...")
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ Success! Your 15-second highlight is ready at: {output_path}")

if __name__ == "__main__":
    run_highlight_extraction()