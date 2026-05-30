import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

class TemporalMemoryEngine:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        """
        Initializes the local vector database instance for video frame tracking.
        """
        # Remove the settings argument completely
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        self.collection = self.client.get_or_create_collection(
            name="gameplay_frame_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

    def search_raw_moments(self, query_embedding: List[float], max_results: int = 50) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
            include=["metadatas", "distances"]
        )
        
        parsed_results = []
        if not results or not results['ids'] or len(results['ids'][0]) == 0:
            return parsed_results

        for idx in range(len(results['ids'][0])):
            parsed_results.append({
                "frame_id": results['ids'][0][idx],
                "distance": results['distances'][0][idx],
                "metadata": results['metadatas'][0][idx]
            })
            
        return parsed_results

    def cluster_temporal_windows(
        self, 
        matched_moments: List[Dict[str, Any]], 
        max_gap_seconds: float = 3.0,
        min_duration_seconds: float = 2.0
    ) -> List[Tuple[float, float, List[str]]]:
        if not matched_moments:
            return []

        moments = []
        for m in matched_moments:
            moments.append((float(m['metadata']['timestamp']), m['frame_id']))
        moments.sort(key=lambda x: x[0])

        event_blocks: List[Tuple[float, float, List[str]]] = []
        current_start = moments[0][0]
        current_end = moments[0][0]
        current_frames = [moments[0][1]]

        for i in range(1, len(moments)):
            current_time, frame_id = moments[i]
            
            if current_time - current_end <= max_gap_seconds:
                current_end = current_time
                current_frames.append(frame_id)
            else:
                if current_end - current_start >= min_duration_seconds:
                    event_blocks.append((current_start, current_end, current_frames))
                
                current_start = current_time
                current_end = current_time
                current_frames = [frame_id]
                
        if current_end - current_start >= min_duration_seconds:
            event_blocks.append((current_start, current_end, current_frames))

        print(f"[Temporal Index] Formed {len(event_blocks)} dynamic gameplay event blocks.")
        return event_blocks


# ==============================================================
# THE NEW OPENAI MEME INDEX (Isolated Database)
# ==============================================================
class MemeMemoryIndex:
    def __init__(self):
        """Initializes a dedicated connection to the Meme database using OpenAI."""
        load_dotenv()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Remove the settings argument completely
        self.meme_client = chromadb.PersistentClient(path="data/meme_db")
        
        self.collection = self.meme_client.get_or_create_collection(name="meme_index")

    def fetch_matching_meme(self, vibe_tag: str) -> str | None:
        """Finds a meme based on the semantic text vibe using OpenAI."""
        try:
            # 1. Convert the Qwen vibe word into math using OpenAI
            response = self.openai_client.embeddings.create(
                input=vibe_tag,
                model="text-embedding-3-small"
            )
            query_vector = response.data[0].embedding

            # 2. Search the Meme Database
            results = self.collection.query(
                query_embeddings=[query_vector], 
                n_results=1,
                include=["metadatas"]
            )
            
            if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
                return results['metadatas'][0][0]['file_path']
            return None
            
        except Exception as e:
            print(f"[Meme Index] Error fetching meme: {e}")
            return None


class AudioMemoryIndex:
    def __init__(self, client_instance):
        """Initializes a collection dedicated to background music and soundbeds."""
        self.collection = client_instance.get_or_create_collection(
            name="vectorized_audio_library",
            metadata={"hnsw:space": "cosine"}
        )

    def register_audio_asset(self, asset_id: str, embedding: list[float], local_path: str, tag: str):
        """Indexes local audio tracks into the vector registry based on their vibe."""
        self.collection.add(
            ids=[asset_id],
            embeddings=[embedding],
            metadatas=[{"file_path": local_path, "vibe_tag": tag}]
        )

    def fetch_matching_audio(self, vibe_tag: str) -> str | None:
        """Finds BGM based on the semantic text vibe."""
        results = self.collection.query(
            query_texts=[vibe_tag], 
            n_results=1,
            include=["metadatas"]
        )
        if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
            return results['metadatas'][0][0]['file_path']
        return None