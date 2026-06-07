import os
import json
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Any, Optional

def safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default

def safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

class MemeFetcher:
    # Stop words to remove generic noise from semantic matching
    STOP_WORDS = {"reaction", "funny", "meme", "gif", "video", "sticker", "transparent", "overlay"}

    def __init__(self, giphy_api_key: str, local_catalog_path: str = "assets/memes/approved/local_meme_catalog.json"):
        self.giphy_api_key = giphy_api_key
        self.local_catalog_path = local_catalog_path
        
        # Robust session handling: User-Agent, 429 support, and Retry-After respect
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "PokemonUnite-CreativeBot/1.0"})
        retries = Retry(
            total=3, 
            backoff_factor=1, 
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

    def _tokenize(self, text: str) -> set:
        """Tokenize text and remove generic stop words for better semantic scoring."""
        if not text:
            return set()
        tokens = set(re.findall(r'\b\w+\b', text.lower()))
        return tokens - self.STOP_WORDS

    def _search_local(self, queries: List[str], max_size_bytes: int = 20 * 1024 * 1024) -> List[Dict[str, Any]]:
        """Searches local catalog using token scoring, validates file existence & size."""
        candidates = []
        if not os.path.isfile(self.local_catalog_path):
            return candidates
            
        try:
            with open(self.local_catalog_path, 'r', encoding='utf-8') as f:
                catalog = json.load(f)
                
            for item in catalog:
                preview_path = item.get("preview_path", item.get("file_path", ""))
                full_path = item.get("file_path", "")
                
                # Strict local file validation
                if not os.path.isfile(preview_path) or not os.path.isfile(full_path):
                    continue
                if os.path.getsize(preview_path) > max_size_bytes or os.path.getsize(full_path) > max_size_bytes:
                    continue

                item_text = (item.get("title", "") + " " + " ".join(item.get("tags", [])))
                item_tokens = self._tokenize(item_text)
                
                best_score = 0
                best_query = ""
                
                # Score against all queries, keep the best match
                for query in queries:
                    query_tokens = self._tokenize(query)
                    if not query_tokens:
                        continue
                    
                    score = len(query_tokens.intersection(item_tokens))
                    if score > best_score:
                        best_score = score
                        best_query = query
                        
                if best_score > 0:
                    candidates.append({
                        "candidate_id": f"local:{item.get('asset_id', 'unknown')}",
                        "provider_asset_id": str(item.get('asset_id', 'unknown')),
                        "title": str(item.get("title", "Local Asset")),
                        "preview_url": str(preview_path), 
                        "full_asset_url": str(full_path),
                        "preview_format": str(item.get("preview_format", "mp4")).lower(),
                        "full_asset_format": str(item.get("format", "mp4")).lower(),
                        "source_page": "local_storage",
                        "originating_search_query": best_query,
                        "retrieval_score": best_score,
                        "retrieval_type": "local",
                        "width": safe_int(item.get("width")),
                        "height": safe_int(item.get("height")),
                        "duration": safe_float(item.get("duration"))
                    })
        except Exception as e:
            print(f"⚠️ [Meme Fetcher] Local cache read error: {e}")
            
        # Rank by highest semantic overlap
        candidates.sort(key=lambda x: x["retrieval_score"], reverse=True)
        return candidates

    def retrieve_candidates(self, search_queries: List[str], max_total: int = 4) -> List[Dict[str, Any]]:
        # Safe return on invalid max_total
        if max_total <= 0:
            return []
            
        # Filter invalid/empty queries
        valid_queries = [q.strip() for q in search_queries if q and str(q).strip()]
        if not valid_queries:
            print("⚠️ [Meme Fetcher] Filtered out all empty/invalid queries.")
            return []

        final_candidates = []
        seen_ids = set()

        # 1. Exhaustive Local Search
        local_matches = self._search_local(valid_queries)
        for m in local_matches:
            if m["candidate_id"] not in seen_ids:
                final_candidates.append(m)
                seen_ids.add(m["candidate_id"])

        if len(final_candidates) >= max_total:
            return final_candidates[:max_total]

        # Return safely if GIPHY key is missing (local-only mode)
        if not self.giphy_api_key or self.giphy_api_key.strip() == "" or self.giphy_api_key == "MISSING":
            print("⚠️ [Meme Fetcher] GIPHY key missing or invalid. Returning local results only.")
            return final_candidates

        # 2. Online Search with Per-Query Bucketing
        query_buckets = {q: [] for q in valid_queries}
        
        # Calculate a safe limit per query to ensure we have enough candidates for round-robin
        remaining_slots = max_total - len(final_candidates)
        limit_per_query = max(2, (remaining_slots // len(valid_queries)) + 2)

        for query in valid_queries:
            # Endpoint selection based on transparency intent
            is_sticker = "transparent" in query.lower() or "sticker" in query.lower()
            endpoint = "stickers" if is_sticker else "gifs"
            url = f"https://api.giphy.com/v1/{endpoint}/search"
            
            params = {
                "api_key": self.giphy_api_key,
                "q": query,
                "limit": limit_per_query,
                "rating": "pg-13"
            }
            
            try:
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json().get("data", [])
                
                for item in data:
                    raw_id = item.get("id")
                    if not raw_id: 
                        continue
                    
                    gif_id = f"giphy:{raw_id}"
                    
                    images = item.get("images", {})
                    original = images.get("original", {})
                    preview = images.get("fixed_height_small", {})
                    
                    preview_url = preview.get("url") or preview.get("mp4")
                    full_url = original.get("mp4") or original.get("url")
                    
                    # Strictly reject missing URLs
                    if not preview_url or not full_url:
                        continue
                        
                    preview_format = "mp4" if ".mp4" in str(preview_url).lower() else ("webp" if ".webp" in str(preview_url).lower() else "gif")
                    full_format = "mp4" if ".mp4" in str(full_url).lower() else ("webp" if ".webp" in str(full_url).lower() else "gif")

                    query_buckets[query].append({
                        "candidate_id": gif_id,
                        "provider_asset_id": str(raw_id),
                        "title": str(item.get("title", "Untitled Giphy Asset")),
                        "preview_url": str(preview_url),
                        "full_asset_url": str(full_url),
                        "preview_format": preview_format,
                        "full_asset_format": full_format,
                        "source_page": str(item.get("url", "")),
                        "originating_search_query": query,
                        "retrieval_score": 1, # Base score for external API results
                        "retrieval_type": "sticker" if is_sticker else "gif",
                        "width": safe_int(original.get("width")),
                        "height": safe_int(original.get("height")),
                        "duration": safe_float(original.get("duration") or item.get("duration"))
                    })
            except requests.exceptions.RequestException as e:
                print(f"⚠️ [Meme Fetcher] Network/API Error on query '{query}': {e}")
            except Exception as e:
                print(f"⚠️ [Meme Fetcher] Unexpected Error processing query '{query}': {e}")
                
        # 3. Round-Robin Distribution (prevents query #1 from monopolizing slots)
        idx = 0
        while len(final_candidates) < max_total:
            added_in_round = False
            for query in valid_queries:
                bucket = query_buckets[query]
                if idx < len(bucket):
                    cand = bucket[idx]
                    if cand["candidate_id"] not in seen_ids:
                        final_candidates.append(cand)
                        seen_ids.add(cand["candidate_id"])
                        added_in_round = True
                        
                    if len(final_candidates) >= max_total:
                        break
                        
            if not added_in_round:
                break # All buckets are exhausted
            idx += 1
                            
        return final_candidates[:max_total]