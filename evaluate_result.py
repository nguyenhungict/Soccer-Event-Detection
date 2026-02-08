
import json
import os

# Mapping Ground Truth về V4 Classes
def map_gt_v4(label):
    if label == "Goal": return "Goal"
    elif label in ["Yellow card", "Red card"]: return "Card"
    elif label == "Penalty": return "Penalty"
    else: return None  # Substitution, etc. -> Ignore

def load_ground_truth(label_path, target_half=1):
    if not os.path.exists(label_path):
        print(f"❌ Label file not found: {label_path}")
        return []
        
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    events = []
    # SoccerNet Format
    anns = data.get('annotations', data)
        
    for ann in anns:
        label_raw = ann.get('label', '')
        v4_label = map_gt_v4(label_raw)
        
        if v4_label:
            game_time = ann['gameTime'] # "1 - 16:30"
            if ' - ' not in game_time: continue
            
            parts = game_time.split(' - ')
            half = int(parts[0])
            m, s = map(int, parts[1].split(':'))
            total_s = m * 60 + s
            
            if half == target_half:
                events.append({
                    'time': total_s,
                    'label': v4_label,
                    'orig_label': label_raw,
                    'gameTime': game_time
                })
    return events

def evaluate(pred_file, label_file=None):
    if not os.path.exists(pred_file):
        print(f"❌ Prediction file not found: {pred_file}")
        return

    # Load Preds JSON
    with open(pred_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Support both list (old) and dict with metadata (new)
    if isinstance(data, list):
        preds = data
        metadata = {}
    else:
        preds = data.get('highlights', [])
        metadata = data.get('metadata', {})

    # Auto-detect Label File
    if not label_file:
        label_file = metadata.get('label_file')
        if label_file:
            print(f"✅ Auto-detected Label File: {label_file}")
        else:
            print("⚠️ No metadata. Checking default path...")
            # Default Label File (Match with Test Script)
            LABEL_DEFAULT = r"E:\USTH_ICT\B3\NLP\dataset\sn-echoes\Dataset\whisper_v1\england_epl\2014-2015\2015-02-21 - 18-00 Chelsea 1 - 1 Burnley\Labels-v2.json"
            label_file = LABEL_DEFAULT

    # Verify Label File
    if not os.path.exists(label_file):
         print(f"❌ Label file not found: {label_file}")
         return

    # Load GT
    truths = load_ground_truth(label_file)
    
    print(f"\n📊 EVALUATION REPORT (V4 Model)")
    print("=" * 70)
    print(f"Match: {os.path.basename(os.path.dirname(label_file))}")
    print(f"Ground Truth (Target): {len(truths)}")
    print(f"Predictions (Model):   {len(preds)}")
    print("-" * 70)
    
    hits = 0
    matched_gt = set()
    
    print(f"{'TIME':<8} | {'PRED':<10} | {'CONF':<5} | {'RESULT':<10} | {'DETAILS'}")
    print("-" * 70)
    
    for p in preds:
        p_time = p['time']
        p_label = p['label']
        score = p.get('score', 0)
        
        m, s = divmod(int(p_time), 60)
        time_str = f"{m:02d}:{s:02d}"
        
        # Check match (Window +/- 30s)
        best_idx = -1
        min_diff = 31 # Threshold 30s
        
        for i, t in enumerate(truths):
            # Check label matches V4 label
            if p_label == t['label']:
                diff = abs(p_time - t['time'])
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i
        
        status = "❌ FP"
        details = ""
        
        if best_idx != -1:
            status = "✅ MATCH"
            t = truths[best_idx]
            original = t['orig_label']
            details = f"Matched {original} (Diff {int(p_time - t['time'])}s)"
            hits += 1
            matched_gt.add(best_idx)
        else:
            # Check if nearly missed (wrong label but correct time)
            for t in truths:
                 if abs(p_time - t['time']) < 30:
                     details = f"⚠️ Wrong Label? (GT: {t['orig_label']})"
                     break

        print(f"{time_str:<8} | {p_label:<10} | {score:.2f}  | {status:<10} | {details}")
        
    print("-" * 70)
    
    # Missed
    print("\n⚠️ MISSED EVENTS (FN):")
    for i, t in enumerate(truths):
        if i not in matched_gt:
            print(f" - {t['gameTime']} : {t['orig_label']} ({t['label']})")
            
    # Metrics
    precision = hits / len(preds) if preds else 0
    recall = len(matched_gt) / len(truths) if truths else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nV4 METRICS:")
    print(f"   Precision: {precision:.2f}")
    print(f"   Recall:    {recall:.2f}")
    print(f"   F1 Score:  {f1:.2f}")

if __name__ == "__main__":
    PRED = "highlights_v4.json"
    
    # Default Label File (Leicester vs Arsenal - High Scoring!)
    LABEL_DEFAULT = r"E:\USTH_ICT\B3\NLP\dataset\soccernet\england_epl\2015-2016\2015-09-26 - 17-00 Leicester 2 - 5 Arsenal\Labels-v2.json"
    
    # Run in Auto Mode, fallback to DEFAULT
    if os.path.exists(PRED):
        evaluate(PRED, LABEL_DEFAULT)
    else:
        print("Please run test_full_match.py first!")
