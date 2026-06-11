import os
import json
import uuid
import math
import re
import logging
from typing import Dict, Any, List, Optional
from src.utils.llm_client import execute_with_colab_fallback

# Setup operational logging
logger = logging.getLogger("MemeDirector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

def safe_float(val: Any, default: float = 0.0) -> float:
    try: return float(val)
    except (ValueError, TypeError): return default

def clean_queries(raw: Any, limit: int = 4) -> List[str]:
    if isinstance(raw, str): raw = [raw]
    if not isinstance(raw, list): return []
    res = []
    for q in raw:
        if isinstance(q, str):
            c = re.sub(r'http\S+', '', q)
            c = re.sub(r'\s+', ' ', c).strip()
            if c and c not in res: res.append(c[:80])
    return res[:limit]

def clean_concepts(raw: Any, limit: int = 5) -> List[str]:
    if isinstance(raw, str): raw = [raw]
    if not isinstance(raw, list): return []
    res, seen = [], set()
    for item in raw:
        if not isinstance(item, str): continue
        val = re.sub(r'http\S+', '', item)
        val = re.sub(r'\s+', ' ', val).strip()
        if not val: continue
        val = val[:60]
        key = val.casefold()
        if key not in seen:
            seen.add(key)
            res.append(val)
    return res[:limit]

def clean_fallback_text(raw: Any, max_chars: int = 100, max_words: int = 20) -> str:
    if not isinstance(raw, str): return ""
    val = re.sub(r'http\S+', '', raw).replace('\n', ' ').strip()
    if not val: return ""
    words = val.split()[:max_words]
    return " ".join(words)[:max_chars].strip()

class MemeDirector:
    ALLOWED_TREATMENTS = {"REACTION_OVERLAY", "TEXT_AND_SOUND", "SOUND_ONLY", "NO_MEME"}
    ALLOWED_REGIONS = {"TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT", "CENTER_LEFT", "CENTER_RIGHT", "TOP_CENTER", "BOTTOM_CENTER"}
    ALLOWED_INTENSITIES = {"LOW", "MEDIUM", "HIGH"}
    TIMING_INTENTS = {"IMMEDIATELY_AFTER_PAYOFF", "DURING_REACTION_WINDOW", "END_OF_REACTION_WINDOW", "NONE"}
    
    APPROVED_SOUNDS = {
        "none", "surprise_impact", "success_sting", "fail_sting", 
        "awkward_pause", "dramatic_hit", "sad_sting", "comedic_accent", "record_scratch"
    }

    FALLBACK_CHAINS = {
        "REACTION_OVERLAY": ["TEXT_AND_SOUND", "SOUND_ONLY", "NO_MEME"],
        "TEXT_AND_SOUND": ["SOUND_ONLY", "NO_MEME"],
        "SOUND_ONLY": ["NO_MEME"],
        "NO_MEME": []
    }

    # 👇 THIS BLOCK IS FIXED: Accepts api_key and base_url instead of client 👇
    def __init__(self, api_key: str, base_url: str, model_name: str, config: Dict[str, Any] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.config = config or {}
        
        self.output_dir = "data/creative/meme_plans"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.min_story_conf = safe_float(self.config.get("min_story_confidence", 0.55))
        self.min_director_conf = safe_float(self.config.get("min_director_confidence", 0.50))
        self.max_text_chars = int(self.config.get("max_text_chars", 100))
        self.max_text_words = int(self.config.get("max_text_words", 20))
        self.allow_text_only = bool(self.config.get("allow_text_only", False))
        
        # LLM client settings
        self.temperature = float(self.config.get("temperature", 0.4))
        self.max_tokens = int(self.config.get("max_tokens", 800))
        self.request_timeout = float(self.config.get("request_timeout", 60.0))

    def _sanitize_id(self, raw_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", str(raw_id))

    def _create_fallback_plan(self, clip_id: str, warnings: List[str]) -> dict:
        logger.warning(f"Clip {clip_id}: Generating fallback NO_MEME plan. Warnings: {warnings}")
        return {
            "schema_version": "1.0",
            "clip_id": clip_id,
            "timeline_basis": "PACED_CLIP",
            "creative_decision": {
                "treatment": "NO_MEME", "meme_needed": False,
                "creative_reason": "Director fallback was required.", "comedy_mechanism": "none",
                "search_queries": [], "avoid_concepts": [],
                "sound_intent": {"function": "none", "approved_library_only": True},
                "fallback_chain": [], "fallback_text": ""
            },
            "placement": {
                "timing_intent": "NONE", "suggested_trigger_timestamp": 0.0, "latest_end_timestamp": 0.0,
                "preferred_regions": [], "intensity": "LOW", "opacity": 1.0
            },
            "director_metadata": {
                "model": self.model_name, "decision_confidence": 0.0,
                "fallback_generated": True, "warnings": warnings
            }
        }

    def _validate_story_input(self, clip_id: str, story: dict) -> List[str]:
        warnings = []
        if not isinstance(story, dict): return ["FATAL: story_contract is not a dictionary."]
        if str(story.get("clip_id", "")) != clip_id: warnings.append("FATAL: clip_id mismatch.")
        if story.get("timeline_basis") != "PACED_CLIP": warnings.append("FATAL: timeline_basis must be PACED_CLIP.")
        
        dur = safe_float(story.get("clip_duration"))
        if not math.isfinite(dur) or dur <= 0: warnings.append("FATAL: Invalid clip_duration.")
        
        rw = story.get("reaction_window", {})
        s, e = safe_float(rw.get("start", -1)), safe_float(rw.get("end", -1))
        if s < 0 or e > dur or s >= e: warnings.append("FATAL: Invalid reaction_window bounds.")
        
        payoff = safe_float(story.get("payoff_timestamp", -1.0))
        if not math.isfinite(payoff) or not (0.0 <= payoff <= dur): warnings.append("FATAL: Invalid payoff_timestamp.")
        
        for text_field in ["viewer_expectation", "actual_outcome", "comedy_mechanism"]:
            if not story.get(text_field): warnings.append(f"FATAL: Missing {text_field}.")
            
        conf = safe_float(story.get("confidence", -1.0))
        if not (0.0 <= conf <= 1.0): warnings.append("FATAL: Invalid story confidence.")

        return warnings

    def _build_prompt(self, story: dict) -> str:
        return f"""
Analyze this scene and decide the creative treatment. Return strict JSON.

STORY CONTEXT:
Scene: {story.get('scene_description')}
Expectation: {story.get('viewer_expectation')}
Outcome: {story.get('actual_outcome')}
Comedy Mechanism: {story.get('comedy_mechanism')}

CONSTRAINTS:
1. Choose ONE treatment: REACTION_OVERLAY, TEXT_AND_SOUND, SOUND_ONLY, NO_MEME.
2. If REACTION_OVERLAY, provide 2-4 specific GIPHY/Tenor search queries.
3. If TEXT_AND_SOUND or REACTION_OVERLAY, provide short punchy fallback_text.
4. sound_function must be one of: {", ".join(self.APPROVED_SOUNDS)}

Respond ONLY with a JSON object matching this schema exactly:
{{
  "treatment": "...",
  "creative_reason": "...",
  "comedy_mechanism": "...",
  "search_queries": [],
  "avoid_concepts": [],
  "sound_function": "none",
  "fallback_text": "",
  "timing_intent": "IMMEDIATELY_AFTER_PAYOFF",
  "preferred_regions": ["TOP_RIGHT", "TOP_LEFT"],
  "intensity": "HIGH",
  "confidence": 0.85
}}
"""

    def _normalize_llm_response(self, clip_id: str, story: dict, llm_output: dict, warnings: List[str]) -> dict:
        story_conf = safe_float(story.get("confidence", 1.0))
        dir_conf = max(0.0, min(1.0, safe_float(llm_output.get("confidence", 0.5))))
        
        treatment = str(llm_output.get("treatment", "NO_MEME")).upper()
        if treatment not in self.ALLOWED_TREATMENTS:
            warnings.append(f"invalid_treatment_normalized: {treatment}")
            treatment = "NO_MEME"

        # 1. Resolve Sound Early
        raw_sound = llm_output.get("sound_function")
        s_func = str(raw_sound).strip().lower() if raw_sound is not None else "none"
        if s_func not in self.APPROVED_SOUNDS:
            warnings.append(f"sound_intent_not_supported: {s_func}")
            s_func = "none"

        # 2. Extract Validated Resources
        final_text = clean_fallback_text(llm_output.get("fallback_text", ""), self.max_text_chars, self.max_text_words)
        final_queries = clean_queries(llm_output.get("search_queries", []))
        final_concepts = clean_concepts(llm_output.get("avoid_concepts", []))

        # 3. Confidence Downgrades
        if story_conf < self.min_story_conf:
            warnings.append("story_confidence_below_threshold")
            treatment = "NO_MEME"
        elif dir_conf < self.min_director_conf:
            warnings.append("director_confidence_below_threshold")
            treatment = "NO_MEME"
        elif dir_conf < 0.65 and treatment == "REACTION_OVERLAY":
            warnings.append("director_confidence_downgraded_overlay")
            treatment = "TEXT_AND_SOUND"

        # 4. State Machine Cascade
        if treatment == "REACTION_OVERLAY" and not final_queries:
            warnings.append("empty_search_queries_for_overlay_downgrade")
            treatment = "TEXT_AND_SOUND"

        if treatment == "TEXT_AND_SOUND":
            if not final_text:
                warnings.append("missing_fallback_text_downgrade")
                treatment = "SOUND_ONLY"
            elif s_func == "none" and not self.allow_text_only:
                warnings.append("text_and_sound_missing_sound_downgrade")
                treatment = "SOUND_ONLY"

        if treatment == "SOUND_ONLY" and s_func == "none":
            warnings.append("sound_only_requires_valid_sound_downgrade")
            treatment = "NO_MEME"

        # 5. Finalize derivations
        meme_needed = (treatment == "REACTION_OVERLAY")
        if not meme_needed: final_queries = []
        if treatment == "NO_MEME":
            final_text, s_func = "", "none"

        # 6. Timing Resolution
        rw = story.get("reaction_window", {})
        clip_dur = safe_float(story.get("clip_duration", 0.0))
        t_start, t_end = safe_float(rw.get("start", 0.0)), safe_float(rw.get("end", clip_dur))
        payoff = safe_float(story.get("payoff_timestamp", -1.0))
        
        t_intent = str(llm_output.get("timing_intent", "IMMEDIATELY_AFTER_PAYOFF")).upper()
        if t_intent not in self.TIMING_INTENTS: t_intent = "IMMEDIATELY_AFTER_PAYOFF"
        
        if t_intent == "IMMEDIATELY_AFTER_PAYOFF" and math.isfinite(payoff) and payoff >= 0.0:
            t_start = max(t_start, payoff)

        if treatment == "NO_MEME":
            t_intent, t_start, t_end = "NONE", 0.0, 0.0

        # 7. Placement Resolution
        raw_regions = llm_output.get("preferred_regions", ["TOP_RIGHT", "TOP_LEFT"])
        if isinstance(raw_regions, str): raw_regions = [raw_regions]
        elif not isinstance(raw_regions, list): raw_regions = []

        regions = []
        if treatment != "NO_MEME":
            for r in raw_regions:
                r_up = str(r).upper()
                if r_up in self.ALLOWED_REGIONS and r_up not in regions: regions.append(r_up)
            regions = regions[:3]
            if not regions: regions = ["TOP_RIGHT", "TOP_LEFT"]

        intensity = str(llm_output.get("intensity", "MEDIUM")).upper()
        if intensity not in self.ALLOWED_INTENSITIES: intensity = "MEDIUM"

        return {
            "schema_version": "1.0",
            "clip_id": clip_id,
            "timeline_basis": "PACED_CLIP",
            "creative_decision": {
                "treatment": treatment,
                "meme_needed": meme_needed,
                "creative_reason": str(llm_output.get("creative_reason", "Deterministic pipeline mapping.")),
                "comedy_mechanism": str(llm_output.get("comedy_mechanism", story.get("comedy_mechanism", "none"))),
                "search_queries": final_queries,
                "avoid_concepts": final_concepts,
                "sound_intent": {"function": s_func, "approved_library_only": True},
                "fallback_chain": self.FALLBACK_CHAINS[treatment],
                "fallback_text": final_text
            },
            "placement": {
                "timing_intent": t_intent,
                "suggested_trigger_timestamp": t_start,
                "latest_end_timestamp": t_end,
                "preferred_regions": regions,
                "intensity": intensity if treatment != "NO_MEME" else "LOW",
                "opacity": 0.95 if treatment != "NO_MEME" else 1.0
            },
            "director_metadata": {
                "model": self.model_name,
                "decision_confidence": dir_conf,
                "fallback_generated": False,
                "warnings": warnings
            }
        }

    def generate_plan(self, clip_id: str, story_contract: dict) -> dict:
        warnings = []
        safe_clip_id = self._sanitize_id(clip_id)
        target_path = os.path.join(self.output_dir, f"{safe_clip_id}_meme_plan.json")
        part_path = target_path + f".{uuid.uuid4().hex[:6]}.part"

        logger.info(f"Clip {clip_id}: Validating story contract...")
        fatal_errors = self._validate_story_input(clip_id, story_contract)
        if fatal_errors:
            return self._atomic_save(self._create_fallback_plan(clip_id, fatal_errors), target_path, part_path)

        prompt = self._build_prompt(story_contract)
        llm_json = {}
        
        try:
            logger.info(f"Clip {clip_id}: Calling LLM '{self.model_name}' via Colab fallback...")
            messages = [{"role": "user", "content": prompt}]
            
            # 👇 THIS CALL IS FIXED: Uses self.api_key and self.base_url 👇
            raw_response = execute_with_colab_fallback(self.api_key, self.base_url, messages)
            
            cleaned = raw_response.strip()
            md_ticks = chr(96) * 3 
            md_json = md_ticks + "json"
            
            if cleaned.startswith(md_json): cleaned = cleaned[7:]
            elif cleaned.startswith(md_ticks): cleaned = cleaned[3:]
            if cleaned.endswith(md_ticks): cleaned = cleaned[:-3]
            
            cleaned = cleaned.strip()
            start_idx = cleaned.find('{')
            end_idx = cleaned.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                cleaned = cleaned[start_idx:end_idx+1]
                
            llm_json = json.loads(cleaned)
            
        except Exception as e:
            msg = f"llm_api_failure: {str(e)}"
            logger.error(f"Clip {clip_id}: {msg}")
            warnings.append(msg)
            return self._atomic_save(self._create_fallback_plan(clip_id, warnings), target_path, part_path)

        logger.info(f"Clip {clip_id}: Normalizing LLM response...")
        final_plan = self._normalize_llm_response(clip_id, story_contract, llm_json, warnings)
        
        logger.info(f"Clip {clip_id}: Saving final meme plan. Treatment: {final_plan['creative_decision']['treatment']}")
        return self._atomic_save(final_plan, target_path, part_path)

    def _atomic_save(self, plan: dict, target_path: str, part_path: str) -> dict:
        try:
            with open(part_path, 'w', encoding='utf-8') as f:
                json.dump(plan, f, indent=2, ensure_ascii=False)
            os.replace(part_path, target_path)
        except Exception as e:
            if os.path.exists(part_path): os.remove(part_path)
            plan["director_metadata"]["warnings"].append(f"atomic_write_failed: {str(e)}")
            logger.error(f"Atomic write failed for {target_path}: {e}")
        return plan