
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class FullMatchTester:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️ Loading model from: {model_path}")
        print(f"   Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label
        print("✅ Model loaded!")

    def load_transcript(self, file_path):
        print(f"📂 Loading transcript: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = []
        # Parse SoccerNet-Echoes format
        if 'segments' in data:
            for seg_id, val in data['segments'].items():
                segments.append({
                    'start': float(val[0]),
                    'end': float(val[1]),
                    'text': str(val[2]).strip()
                })
        else:
            # Fallback for simple list format
            segments = data

        segments.sort(key=lambda x: x['start'])
        print(f"✅ Loaded {len(segments)} segments")
        return segments

    def create_windows(self, segments):
        windows = []
        for i in range(1, len(segments) - 1):
            prev = segments[i-1]
            curr = segments[i]
            next_ = segments[i+1]
            
            text = f"{prev['text']} {curr['text']} {next_['text']}"
            windows.append({
                'start': prev['start'],
                'center': curr['start'],
                'end': next_['end'],
                'text': text
            })
        return windows

    def detect_events(self, windows):
        print(f"🧠 analyzing {len(windows)} windows...")
        events = []
        batch_size = 16 # Adjust based on RAM
        
        # Thresholds (Optimized based on Eval Report)
        thresholds = {
            "Goal": 0.7,           # Lowered to 0.7 to catch missed goals (FN)
            "Red card": 0.8,       # Increased for safety
            "Penalty": 0.8,        # Increased for safety
            "Yellow card": 0.75,   # Increased to Kill FPs (was 0.68)
            "Substitution": 0.95   # Increased drastically (was 0.92) to kill FPs
        }

        for i in range(0, len(windows), batch_size):
            batch = windows[i:i+batch_size]
            texts = [w['text'] for w in batch]
            
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=160).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = probs.cpu().numpy()
            pred_ids = np.argmax(probs, axis=1)
            
            for j, pred_id in enumerate(pred_ids):
                label = self.id2label[pred_id]
                score = probs[j][pred_id]
                
                req_score = thresholds.get(label, 0.5)
                
                if label != "No-Event" and score >= req_score:
                    win = batch[j]
                    events.append({
                        "time": win['center'],
                        "label": label,
                        "score": float(score),
                        "text": win['text']
                    })
                    
            if i % 100 == 0:
                print(f"   Processed {i}/{len(windows)}...", end='\r')
        
        print("\n✅ Analysis complete!")
        return self.merge_events(events)

    def merge_events(self, events):
        if not events: return []
        merged = []
        curr = events[0]
        
        for next_evt in events[1:]:
            # Clean up duplicates within 15 seconds
            if (next_evt['time'] - curr['time'] < 15) and (next_evt['label'] == curr['label']):
                if next_evt['score'] > curr['score']:
                    curr = next_evt
            else:
                merged.append(curr)
                curr = next_evt
        merged.append(curr)
        return merged

if __name__ == "__main__":
    # CONFIG PATHS
    MODEL_PATH = "./models/soccer_event_temporal"
    # Crystal Palace vs Liverpool (1st Half)
    TRANSCRIPT_FILE = "./dataset/sn-echoes/Dataset/whisper_v3/england_epl/2016-2017/2016-10-29 - 19-30 Crystal Palace 2 - 4 Liverpool/1_asr.json"
    
    # Resolve absolute paths
    MODEL_PATH = os.path.abspath(MODEL_PATH)
    TRANSCRIPT_FILE = os.path.abspath(TRANSCRIPT_FILE)
    
    if not os.path.exists(TRANSCRIPT_FILE):
        print(f"❌ Transcript file not found: {TRANSCRIPT_FILE}")
        # Try finding ANY json file if specific one is missing
        import glob
        files = glob.glob("./dataset/**/*.json", recursive=True)
        asr_files = [f for f in files if "asr.json" in f]
        if asr_files:
            TRANSCRIPT_FILE = os.path.abspath(asr_files[0])
            print(f"⚠️ Falling back to: {TRANSCRIPT_FILE}")
        else:
            exit()
            
    tester = FullMatchTester(MODEL_PATH)
    segments = tester.load_transcript(TRANSCRIPT_FILE)
    windows = tester.create_windows(segments)
    
    if len(windows) > 0:
        highlights = tester.detect_events(windows)
        
        print("\n" + "="*60)
        print(f"🎉 FOUND {len(highlights)} HIGHLIGHTS")
        print("="*60)
        
        for h in highlights:
            m, s = divmod(int(h['time']), 60)
            print(f"⏱️ {m:02d}:{s:02d} | {h['label']:<12} ({h['score']:.2f}) | \"...{h['text'][-50:]}...\"")
            
        # Save to JSON
        with open("highlights_full.json", "w", encoding='utf-8') as f:
            json.dump(highlights, f, indent=2)
            print(f"\nSaved to highlights_full.json")
