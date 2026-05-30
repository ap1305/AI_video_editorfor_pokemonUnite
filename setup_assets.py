import os
import chromadb
from chromadb.utils import embedding_functions

# Chroma comes with a highly optimized, lightweight, local embedding model by default
default_ef = embedding_functions.DefaultEmbeddingFunction()

def initialize_asset_database():
    """Reads your asset descriptions and permanently embeds them into ChromaDB."""
    print("=== Booting Asset Indexer ===")
    
    client = chromadb.PersistentClient(path="data/chroma_db")
    
    # 1. Setup the Meme Collection
    meme_collection = client.get_or_create_collection(
        name="vectorized_meme_library",
        embedding_function=default_ef
    )
    
    # 2. Setup the Audio Collection
    audio_collection = client.get_or_create_collection(
        name="vectorized_audio_library",
        embedding_function=default_ef
    )

    # ---------------------------------------------------------
    # YOUR MEME REGISTRY (Rich Descriptions!)
    # ---------------------------------------------------------
    memes_to_add = [
        {
            "id": "meme_spongebob_waiting",
            "path": "assets/memes/spongebob_few_moments_later.mp4",
            # This rich description is what gets converted to a vector!
            "description": "funny fail, waiting a long time, awkward silence, sad, boring"
        },
        {
            "id": "meme_wow_guy",
            "path": "assets/memes/wow_anime_guy.mp4",
            "description": "hype, amazing play, clutch factor, mind blown, incredible, winning"
        }
    ]

    # ---------------------------------------------------------
    # YOUR BACKGROUND MUSIC REGISTRY
    # ---------------------------------------------------------
    audio_to_add = [
        {
            "id": "audio_heavy_phonk",
            "path": "assets/audio/heavy_phonk_bass.mp3",
            "description": "hype, team fight, chaotic, aggressive, intense, clutch"
        },
        {
            "id": "audio_sneaky_tiptoe",
            "path": "assets/audio/sneaky_tiptoe_cartoon.mp3",
            "description": "stealthy, ambush, hiding in bushes, quiet, sneaking"
        }
    ]

    print("\n[Indexer] Embedding Memes...")
    for meme in memes_to_add:
        # We pass the 'description' to the 'documents' field. Chroma automatically vectorizes it!
        meme_collection.upsert(
            ids=[meme["id"]],
            documents=[meme["description"]], 
            metadatas=[{"file_path": meme["path"]}]
        )
        print(f" ✅ Vectorized: {meme['id']}")

    print("\n[Indexer] Embedding Audio...")
    for audio in audio_to_add:
        audio_collection.upsert(
            ids=[audio["id"]],
            documents=[audio["description"]],
            metadatas=[{"file_path": audio["path"]}]
        )
        print(f" ✅ Vectorized: {audio['id']}")
        
    print("\n=== Asset Indexing Complete! ===")

if __name__ == "__main__":
    initialize_asset_database()