import json
import os

# Your human-verified elite moments
human_moments = [
    {"time_str": "2:52", "timestamp": 172, "event": "2v1 kill (went behind)", "quality": "good"},
    {"time_str": "4:29", "timestamp": 269, "event": "3v1 fight", "quality": "good"},
    {"time_str": "5:40", "timestamp": 340, "event": "1v1 base kill", "quality": "good"},
    {"time_str": "6:45", "timestamp": 405, "event": "Zeraora base kill", "quality": "good"},
    {"time_str": "7:00", "timestamp": 420, "event": "Bulbasaur kill + 50 goal", "quality": "excellent"},
    {"time_str": "7:54", "timestamp": 474, "event": "Massive 3v3 fight, 2 kills", "quality": "excellent"},
    {"time_str": "8:32", "timestamp": 512, "event": "100 goal + Zeraora kill", "quality": "excellent"},
    {"time_str": "9:00", "timestamp": 540, "event": "1v1 kill", "quality": "good"},
    {"time_str": "9:35", "timestamp": 575, "event": "Base defense Lucario kill", "quality": "good"},
    {"time_str": "10:00", "timestamp": 600, "event": "Died to 4 people", "quality": "bad"} 
]

def run_coverage_report():
    print("🔍 Booting Candidate Coverage Diagnostic...\n")
    
    # Adjust this path if your candidate_windows.json saves somewhere else
    candidates_path = os.path.join("data", "inputs", "candidate_windows.json")
    
    if not os.path.exists(candidates_path):
        print(f"❌ Error: Could not find {candidates_path}")
        print("Please ensure your last run saved the candidate_windows.json file.")
        return

    with open(candidates_path, 'r') as f:
        candidates = json.load(f)

    print(f"📂 Loaded {len(candidates)} Candidate Windows from pipeline.\n")
    print("="*65)
    print(f"{'TIME':<8} | {'STATUS':<12} | {'EVENT'}")
    print("="*65)

    total_good = 0
    covered_good = 0

    for moment in human_moments:
        is_covered = False
        target = moment["timestamp"]
        
        for window in candidates:
            # Bulletproof dictionary lookup supporting multiple schema versions
            start = window.get("start_time", window.get("start"))
            end = window.get("end_time", window.get("end"))
            
            if start is not None and end is not None:
                if (start - 3) <= target <= (end + 3):
                    is_covered = True
                    break
        
        status = "✅ COVERED" if is_covered else "❌ MISSED"
        
        if moment["quality"] != "bad":
            total_good += 1
            if is_covered:
                covered_good += 1
        else:
            status = "✅ IGNORED" if not is_covered else "❌ INCLUDED"

        print(f"{moment['time_str']:<8} | {status:<12} | {moment['event']}")

    print("="*65)
    
    if total_good > 0:
        coverage_rate = (covered_good / total_good) * 100
        print(f"📊 Final Discovery Rate: {covered_good}/{total_good} ({coverage_rate:.1f}%)")

if __name__ == "__main__":
    run_coverage_report()