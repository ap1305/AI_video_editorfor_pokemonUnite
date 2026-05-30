import os
import re

def clean_ocr_log(input_path="data/OCR_Reports/bulleye_detection.txt", output_path="data/inputs/cleaned_timeline.txt"):
    """
    Reads the raw Colab OCR log, vaporizes the stream overlays and UI noise, 
    and saves a perfectly clean action timeline using strict boundaries.
    """
    print(f"🧹 [Event Extractor] Booting up... Reading {input_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: Could not find {input_path}")
        return False

    # 1. The Expanded Hit List (UI noise added, Victorious/Defeated removed from trash)
    static_noise = [
        r"^\s*🎯\s*\[BULLSEYE\] Action found at", 
        "LIVE NOW POKEMON UNITE ROAD TO MASTER",
        r"FPS[;:]?\s*\d*", 
        "BATTLE DATA MOVESET PERFORMANCE",
        "VIEW BATTLE REPORT", 
        "DETAILS BATTLE DATA",
        "DMG", "DEALT", "TAKEN", "RECOVERY",
        r"UN[LI]TE\s*MOV[IE]?\w*",  # Catches UNITE MOVE UI bugs
        "POTION",
        r"ENERGY\s*HELD"
    ]

    # 2. Strict Action Matching (Victorious/Defeated added here as GOLD items)
    other_keywords = ["STREAK", "STOLEN", "GOAL", "DEFENDING", "ATTACKING", "VICTORIOUS", "DEFEATED"]
    
    # This regex specifically looks for standalone "KO", "K0", "2KO", "3-KO", etc.
    # It will mathematically IGNORE words like "KOTRATUNED"
    ko_pattern = re.compile(r'\b\d?-?K[O0]\b', re.IGNORECASE)

    cleaned_lines = []
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            if not line.strip():
                continue
                
            # Nuke scoreboard stats instantly (VICTORIOUS is no longer deleted here)
            if any(end_word in line for end_word in ["BATTLE REPORT", "DMG", "DETAILS BATTLE DATA"]):
                continue

            clean_line = line
            
            # Vaporize the noise
            for noise in static_noise:
                clean_line = re.sub(noise, "", clean_line, flags=re.IGNORECASE)

            # Iron out the weird spacing
            clean_line = re.sub(r'\s+', ' ', clean_line).strip()

            # STRICT FILTER: Does it have a real KO, or one of the other keywords?
            has_ko = bool(ko_pattern.search(clean_line))
            has_other_action = any(kw in clean_line.upper() for kw in other_keywords)

            if has_ko or has_other_action:
                cleaned_lines.append(clean_line)

        # Save the polished output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in cleaned_lines:
                f.write(f"{line}\n")
                
        print(f"✅ [Event Extractor] Success!")
        print(f"📉 Reduced {len(lines)} messy lines down to {len(cleaned_lines)} pure action events.")
        print(f"📁 Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ [Event Extractor] Failed to process log: {e}")
        return False

if __name__ == "__main__":
    clean_ocr_log()