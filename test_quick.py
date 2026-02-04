"""
Quick Test Script for Soccer Event Model
Sử dụng transcript giả lập để verify model logic.
"""
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class QuickTester:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️ Loading model from: {model_path}")
        print(f"⚙️ Device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.id2label = self.model.config.id2label
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def get_mock_transcript(self):
        """
        Tạo dữ liệu giả lập chứa các events cần test.
        """
        data = [
            # --- NO EVENT ---
            (10.0, 15.0, "The teams are lining up on the pitch."),
            (15.0, 20.0, "Beautiful weather here at Old Trafford."),
            (20.0, 25.0, "United passing the ball around the back."),
            
            # --- GOAL ---
            (100.0, 105.0, "Rashford makes a run forward."),
            (105.0, 110.0, "He shoots! What a incredible strike!"),
            (110.0, 115.0, "It's a goal! Manchester United take the lead!"), # KEY
            
            # --- FOUL / YELLOW CARD ---
            (200.0, 205.0, "Oh that looks like a nasty challenge."),
            (205.0, 210.0, "The referee blows his whistle immediately."),
            (210.0, 215.0, "He's reaching into his pocket. It's a yellow card for Shaw."), # KEY
            
            # --- PENALTY ---
            (400.0, 405.0, "He dribbles into the box."),
            (405.0, 410.0, "Down he goes! The referee points to the spot!"),
            (410.0, 415.0, "Penalty kick given to Liverpool."), # KEY
            
            # --- RED CARD ---
            (500.0, 505.0, "That is a shocking tackle! Two footed!"),
            (505.0, 510.0, "The referee has no choice here."),
            (510.0, 515.0, "He's sent off! Red card! They are down to 10 men."), # KEY
            
            # --- SUBSTITUTION ---
            (600.0, 605.0, "There is activity on the bench."),
            (605.0, 610.0, "Looks like we are going to have a change."),
            (610.0, 615.0, "Mata is coming on to replace Bruno Fernandes."), # KEY
        ]
        
        segments = []
        for start, end, text in data:
            segments.append({'start': start, 'end': end, 'text': text})
        return segments

    def run_inference(self):
        segments = self.get_mock_transcript()
        
        # Create Sliding Windows
        windows = []
        for i in range(1, len(segments) - 1):
            prev = segments[i-1]
            curr = segments[i]
            next_ = segments[i+1]
            
            text = f"{prev['text']} {curr['text']} {next_['text']}"
            
            windows.append({
                'center_time': curr['start'],
                'text': text
            })
            
        print(f"\n🧪 Testing {len(windows)} windows...")
        
        # Predict
        for w in windows:
            inputs = self.tokenizer(
                w['text'], 
                return_tensors="pt", 
                truncation=True, 
                max_length=160
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            score, pred_id = torch.max(probs, dim=1)
            pred_label = self.id2label[pred_id.item()]
            score = score.item()
            
            icon = "⚪"
            if pred_label == "Goal": icon = "⚽"
            elif pred_label == "Red card": icon = "🟥"
            elif pred_label == "Yellow card": icon = "🟨"
            elif pred_label == "Penalty": icon = "⚡"
            elif pred_label == "Substitution": icon = "🔄"
            
            # Print simplified output
            print(f"{icon} {pred_label:<12} ({score:.2f}) | \"...{w['text'][-40:]}\"")

if __name__ == "__main__":
    # --- CONFIG ---
    # Update this path to where your model is saved.
    # Found model at: e:\USTH_ICT\B3\NLP\models\soccer_event_temporal
    MODEL_PATH = "./models/soccer_event_temporal" 
    
    # Resolve absolute path for clarity
    MODEL_PATH = os.path.abspath(MODEL_PATH)
    print(f"Target Model Path: {MODEL_PATH}")
    
    if os.path.exists(MODEL_PATH):
        try:
            tester = QuickTester(MODEL_PATH)
            tester.run_inference()
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"⚠️ Model path not found: {MODEL_PATH}")
        print("Please edit the script to point to your trained model folder.")
