import json
import os

def load_ground_truth(label_path, target_half=1):
    if not os.path.exists(label_path):
        print(f"Error: Label file not found at {label_path}")
        return []
        
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    events = []
    for ann in data['annotations']:
        game_time = ann['gameTime'] # "1 - 16:30"
        label = ann['label']
        
        parts = game_time.split(' - ')
        half = int(parts[0])
        minutes, seconds = map(int, parts[1].split(':'))
        total_seconds = minutes * 60 + seconds
        
        if half == target_half:
            events.append({
                'time': total_seconds,
                'label': label,
                'gameTime': game_time
            })
    return events

def evaluate(prediction_file, label_file):
    if not os.path.exists(prediction_file):
        print("Prediction file not found. Run test_full_match.py first.")
        return

    # Load Predictions
    with open(prediction_file, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    
    # Load Ground Truth
    truths = load_ground_truth(label_file)
    
    target_classes = ["Goal", "Yellow card", "Red card", "Penalty", "Substitution"]
    filtered_truths = [t for t in truths if t['label'] in target_classes]
    
    print(f"\n📊 EVALUATION REPORT (Half 1)")
    print("=" * 80)
    print(f"Ground Truth Events: {len(filtered_truths)} (ignoring fouls/corners/etc)")
    print(f"Predicted Events:    {len(preds)}")
    print("-" * 80)
    
    hits = 0
    matched_truths_indices = set()
    
    print(f"{'TIME':<8} | {'PREDICTION':<15} | {'CONF':<5} | {'RESULT':<10} | {'GROUND TRUTH MATCH'}")
    print("-" * 80)
    
    for p in preds:
        p_time = p['time']
        p_label = p['label']
        score = p.get('score', 0)
        m, s = divmod(int(p_time), 60)
        p_time_str = f"{m:02d}:{s:02d}"
        
        match_found = False
        match_details = ""
        
        # Check against all truths
        # Model time usually lags slightly behind real time (reaction time) OR ahead if transcript segment logic varies
        # Let's verify within +/- 60 seconds to be generous with Whisper alignment issues
        best_match_idx = -1
        min_diff = 999
        
        for i, t in enumerate(truths):
            diff = abs(p_time - t['time'])
            if diff <= 60 and p_label == t['label']:
                if diff < min_diff:
                    min_diff = diff
                    best_match_idx = i
        
        if best_match_idx != -1:
            match_found = True
            t = truths[best_match_idx]
            match_details = f"{t['label']} at {t['gameTime']} (diff {int(p_time - t['time'])}s)"
            matched_truths_indices.add(best_match_idx)
            
        status = "✅ MATCH" if match_found else "❌ FP"
        if match_found: hits += 1
        
        if not match_found and p_label == "Goal": status = "❌ FALSE GOAL" # Highlight critical errors
        
        print(f"{p_time_str:<8} | {p_label:<15} | {score:.2f}  | {status:<10} | {match_details}")

    print("=" * 80)
    
    # Missed Events
    print("\n⚠️ MISSED EVENTS (FN):")
    for i, t in enumerate(truths):
        if t['label'] in target_classes and i not in matched_truths_indices:
             print(f" - {t['gameTime']} : {t['label']}")

    # Summary
    precision = hits / len(preds) if preds else 0
    recall = hits / len(filtered_truths) if filtered_truths else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n📈 METRICS:")
    print(f"   Precision: {precision:.2f} (How many predictions were correct?)")
    print(f"   Recall:    {recall:.2f}    (How many real events did we find?)")
    print(f"   F1 Score:  {f1:.2f}")

if __name__ == "__main__":
    PRED_FILE = "highlights_full.json"
    LABEL_FILE = r"dataset\soccernet\england_epl\2016-2017\2016-10-29 - 19-30 Crystal Palace 2 - 4 Liverpool\Labels-v2.json"
    
    evaluate(PRED_FILE, LABEL_FILE)
