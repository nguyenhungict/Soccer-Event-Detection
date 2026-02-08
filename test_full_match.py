
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATHS = [
    "./models_v4",                   # Correct path (files are here)
    "./models_v4/final_model",       # Fallback 1
    "./models_local_v4/final_model"  # Fallback 2
]

# File test 
TRANSCRIPT_FILE = r"E:\USTH_ICT\B3\NLP\dataset\sn-echoes\Dataset\whisper_v2_en\england_epl\2015-2016\2015-09-26 - 17-00 Leicester 2 - 5 Arsenal\1_asr.json"

THRESHOLDS = {
    "Goal": 0.90,    
    "Card": 0.995,    
    "Penalty": 0.9  
}

class FullMatchTester:
    def __init__(self, model_path):
        self.device = "cpu" 
        print(f"⚙️ Loading model from: {model_path}")
        print(f"   Device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.id2label = self.model.config.id2label
            print(f"✅ Model loaded! Classes: {list(self.id2label.values())}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit()

    def load_transcript(self, file_path):
        print(f"Loading transcript: {file_path}")
        if not os.path.exists(file_path):
            print("❌ File not found!")
            return []
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = []
        if 'segments' in data:
            for seg_id, val in data['segments'].items():
                segments.append({
                    'start': float(val[0]),
                    'end': float(val[1]),
                    'text': str(val[2]).strip()
                })
        else:
            segments = data

        segments.sort(key=lambda x: x['start'])
        print(f"✅ Loaded {len(segments)} segments")
        return segments

    def create_windows(self, segments):
        windows = []
        for i in range(1, len(segments) - 1):
            text = f"{segments[i-1]['text']} {segments[i]['text']} {segments[i+1]['text']}"
            windows.append({
                'start': segments[i-1]['start'],
                'center': segments[i]['start'],
                'end': segments[i+1]['end'],
                'text': text.strip()
            })
        return windows

    def detect_events(self, windows):
        print(f"Analyzing {len(windows)} windows...")
        events = []
        batch_size = 32
        
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i+batch_size]
            texts = [w['text'] for w in batch]
            
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = probs.cpu().numpy()
            
            for j, prob in enumerate(probs):
                top_idx = np.argmax(prob)
                label = self.id2label[top_idx]
                score = float(prob[top_idx])
                
                # Chỉ lấy Event quan trọng (Bỏ No-Event)
                if label in ["Goal", "Card", "Penalty"]:
                    threshold = THRESHOLDS.get(label, 0.5)
                    
                    if score >= threshold:
                        win = batch[j]
                        m, s = divmod(int(win['center']), 60)
                        
                        # In ra màn hình
                        # print(f"✅ DETECTED: ⏱️ {m:02d}:{s:02d} | 🎯 {label:<10} ({score:.2f})")
                        
                        events.append({
                            "time": win['center'],
                            "label": label,
                            "score": score,
                            "text": win['text']
                        })

            if i % 500 == 0:
                print(f"   Processed {i}/{len(windows)}...", end='\r')
        
        print("\n✅ Analysis complete!")
        return self.merge_events(events)

    def merge_events(self, events):
        """Gộp các event trùng nhau trong khoảng thời gian (45s)"""
        if not events: return []
        merged = []
        curr = events[0]
        
        for next_evt in events[1:]:
            # Clean up duplicates within 45 seconds to avoid double counting
            # Nếu cùng loại và gần nhau < 45s -> Gộp, lấy cái có score cao hơn
            if (next_evt['time'] - curr['time'] < 45) and (next_evt['label'] == curr['label']):
                if next_evt['score'] > curr['score']:
                    curr = next_evt
            else:
                merged.append(curr)
                curr = next_evt
        merged.append(curr)
        return merged

if __name__ == "__main__":
    # 1. Tìm Model
    final_model_path = None
    for p in MODEL_PATHS:
        if os.path.exists(p):
            final_model_path = p
            break
    
    if not final_model_path:
        print("❌ Model not found! Please check './models_v4' or './models_local_v4'")
        exit()
        
    # 2. Setup & Run
    tester = FullMatchTester(final_model_path)
    segments = tester.load_transcript(TRANSCRIPT_FILE)
    
    if not segments:
        # Fallback tìm file bất kỳ nếu file định sẵn không thấy
        import glob
        print("⚠️ Default transcript not found, searching folder...")
        files = glob.glob("E:/USTH_ICT/B3/NLP/dataset/**/*.json", recursive=True)
        asr = [f for f in files if "asr.json" in f]
        if asr:
            TRANSCRIPT_FILE = asr[0]
            print(f"Using found file: {TRANSCRIPT_FILE}")
            segments = tester.load_transcript(TRANSCRIPT_FILE)
    
    if segments:
        windows = tester.create_windows(segments)
        highlights = tester.detect_events(windows)
        
        print("\n" + "="*60)
        print(f"🎉 FOUND {len(highlights)} HIGHLIGHTS (V4: Goal, Card, Penalty)")
        print("="*60)
        
    # Find Labels file automatically
    import glob
    from pathlib import Path
    
    label_file = None
    try:
        # Transcript: .../League/Season/Match/1_asr.json
        # Labels:     .../soccernet/League/Season/Match/Labels-v2.json
        
        # Go up 4 levels from 1_asr.json to get to Dataset root (assuming structure)
        # Or simpler: replace 'whisper_vX' with 'soccernet' and filename with 'Labels-v2.json'
        
        parts = Path(TRANSCRIPT_FILE).parts
        match_name = parts[-2]
        season = parts[-3]
        league = parts[-4]
        
        # Construct path to Soccernet labels
        # Need to find where 'dataset' folder is
        base_dataset = str(Path(TRANSCRIPT_FILE).parent.parent.parent.parent.parent)
        
        possible_label = os.path.join(base_dataset, "soccernet", league, season, match_name, "Labels-v2.json")
        
        if os.path.exists(possible_label):
            label_file = possible_label
            print(f"✅ Found Label File: {label_file}")
        else:
             # Fuzzy search fallback (Dynamic)
             print(f"⚠️ Precise label path not found, searching for {match_name}...")
             # Search entire soccernet folder for this match folder
             search_root = os.path.join(base_dataset, "soccernet")
             search = glob.glob(f"{search_root}/**/{match_name}/Labels-v2.json", recursive=True)
             
             if search:
                 label_file = search[0]
                 print(f"✅ Found Label File (Dynamic Search): {label_file}")
             else:
                 print(f"❌ Could not find labels for {match_name}")
    except Exception as e:
        print(f"⚠️ Error finding label file: {e}")

    # Output structure
    output_data = {
        "metadata": {
            "transcript_file": TRANSCRIPT_FILE,
            "label_file": label_file
        },
        "highlights": highlights
    }

    # Save results
    out_file = "highlights_v4.json"
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
        print(f"\n💾 Saved to {out_file}")
