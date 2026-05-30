import re

def clean_and_structure_ocr_log(raw_log_path: str, clean_log_path: str):
    """
    Acts as the 'Event Extractor'. 
    Strips out static stream text and keeps only high-value action lines.
    """
    # 1. The garbage we want to instantly delete
    static_noise = [
        "LIVE NOW POKEMON UNITE ROAD TO MASTER",
        "FPS:", 
        "FPS;",
        "VICTORIOUS",
        "DEFEATED",
        "BATTLE DATA MOVESET PERFORMANCE",
        "VIEW BATTLE REPORT",
        "DETAILS BATTLE DATA"
    ]

    # 2. The gold we actually care about (Action Triggers)
    action_keywords = ["KO", "STREAK", "STOLEN", "GOAL", "DEFENDING", "ATTACKING"]

    cleaned_lines = []
    
    try:
        with open(raw_log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Filter out lines that are clearly just the end-of-game scoreboard
            if "VICTORIOUS" in line or "DMG" in line or "BATTLE REPORT" in line:
                continue

            # Strip out the static stream overlay noise
            clean_line = line
            for noise in static_noise:
                # Use regex for case-insensitive replacement
                clean_line = re.sub(noise, "", clean_line, flags=re.IGNORECASE)

            # Clean up awkward spacing left behind by deletions
            clean_line = re.sub(r'\s+', ' ', clean_line).strip()

            # ONLY keep the line if it actually contains an action keyword
            if any(keyword in clean_line.upper() for keyword in action_keywords):
                cleaned_lines.append(clean_line)

        # Write the beautiful, structured timeline to a new file
        with open(clean_log_path, 'w', encoding='utf-8') as f:
            for line in cleaned_lines:
                f.write(f"{line}\n")
                
        print(f"[Event Extractor] Scrubbed {len(lines)} raw lines down to {len(cleaned_lines)} action events.")
        return True

    except Exception as e:
        print(f"[Event Extractor] Failed to clean log: {e}")
        return False