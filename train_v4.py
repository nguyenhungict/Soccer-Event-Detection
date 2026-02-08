# ⚽ Soccer Event Detection - V4 SIMPLIFIED (4 CLASSES) - PENALTY VERSION
# ==============================================================================
# ✅ 4 Classes: No-Event, Goal, Card, Penalty
# ✅ Focus on HIGHLIGHTS (Goal, Card, Penalty)
# ✅ Substitution → No-Event (less important for highlights)
#
# Fixed Issues:
# 1. find_labels_file() - Proper path matching with fuzzy search
# 2. Don't skip matches without events (need No-Event samples)
# 3. Better weight calculation using sklearn
# 4. Auto-detect Colab paths
# 5. Added validation & error handling
# 6. Penalty instead of Substitution (more critical for highlights)
# ==============================================================================

import os
import glob
import gc
import re
import json
import random
import difflib
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# GPU SETUP
# ==============================================================================
print("🧹 Clearing GPU cache...")
torch.cuda.empty_cache()
gc.collect()
print("✅ GPU cleared\n")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
@dataclass
class Config:
    # Paths
    dataset_root: str = "/content/drive/MyDrive/NLP_Soccer_Temporal/dataset/sn-echoes/Dataset"
    soccernet_labels_dir: str = "/content/drive/MyDrive/NLP_Soccer_Temporal/dataset/soccernet"
    output_dir: str = "/content/drive/MyDrive/NLP_Soccer_Temporal/models_v4_simplified"

    # Model
    model_name: str = "xlm-roberta-base"
    max_length: int = 160

    # Training
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_steps: int = 500
    weight_decay: float = 0.01

    # Balance (33% Events - 67% No-Event)
    no_event_multiplier: float = 2.0

    # Optimization
    max_matches_to_use: int = None

    # Temporal
    reaction_lag_start: int = 1
    reaction_lag_end: int = 6

    # Whisper versions
    whisper_folders: List[str] = None
    event_classes: List[str] = None

    def __post_init__(self):
        # Whisper versions
        self.whisper_folders = ["whisper_v1_en", "whisper_v2_en"]

        # 4 Classes (Highlight-focused)
        self.event_classes = ["No-Event", "Goal", "Card", "Penalty"]
        self.label2id = {label: idx for idx, label in enumerate(self.event_classes)}
        self.id2label = {idx: label for idx, label in enumerate(self.event_classes)}

        # Auto-detect Colab
        if '/content/drive' in os.getcwd():
            base = "/content/drive/MyDrive/NLP_Soccer_Temporal"
            self.dataset_root = f"{base}/dataset/sn-echoes/Dataset"
            self.soccernet_labels_dir = f"{base}/dataset/soccernet"
            self.output_dir = f"{base}/models_v4_simplified"
            print("✅ Auto-detected Colab environment")

config = Config()
print("="*70)
print("✅ Config V4 Loaded - PENALTY VERSION")
print("="*70)
print(f"Classes: {config.event_classes}")
print(f"Focus: Highlights (Goal, Card, Penalty)")
print(f"Dataset: {config.dataset_root}")
print(f"Output:  {config.output_dir}")
print("="*70 + "\n")

# ==============================================================================
# LABEL MAPPING
# ==============================================================================
def map_label_v4(original_label: str) -> str:
    """
    Map 6 original classes to 4 simplified classes:
    - Goal → Goal
    - Yellow/Red card → Card
    - Penalty → Penalty ✅ CRITICAL HIGHLIGHT EVENT
    - Substitution → No-Event (tactical, less important for highlights)
    - Others → No-Event
    """
    if original_label == "Goal":
        return "Goal"
    elif original_label in ["Yellow card", "Red card"]:
        return "Card"
    elif original_label == "Penalty":
        return "Penalty"
    else:  # Substitution, Offside, etc. → No-Event
        return "No-Event"

# ==============================================================================
# LABEL LOADER
# ==============================================================================
class SoccerNetLabelLoader:
    def __init__(self, config: Config):
        self.config = config

    def parse_game_time(self, game_time: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse '1 - 17:31' to (half=1, seconds=1051)"""
        try:
            parts = game_time.split(' - ')
            half = int(parts[0])
            m, s = map(int, parts[1].split(':'))
            return half, m * 60 + s
        except:
            return None, None

    def load_labels(self, labels_path: str) -> List[Dict]:
        """
        Load labels from SoccerNet Labels-v2.json
        Returns: List of {'half': int, 'time': int, 'label': str}
        """
        if not os.path.exists(labels_path):
            return []

        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            events = []
            for ann in data.get('annotations', []):
                label_orig = ann.get('label', '')

                # ✅ Map to V4 classes
                mapped_label = map_label_v4(label_orig)

                # ✅ Only store actual events (No-Event created in align())
                if mapped_label != "No-Event":
                    half, seconds = self.parse_game_time(ann.get('gameTime', ''))
                    if half is not None:
                        events.append({
                            'half': half,
                            'time': seconds,
                            'label': mapped_label
                        })

            return events

        except Exception as e:
            print(f"⚠️ Error loading {labels_path}: {e}")
            return []

    def find_labels_file(self, match_folder: str) -> Optional[str]:
        """
        Find Labels-v2.json for a transcript folder.

        Structure:
          Transcripts: dataset/whisper_v1_en/england_epl/2016-2017/Match_Name/
          Labels:      dataset/soccernet/england_epl/2016-2017/Match_Name/Labels-v2.json
        """
        try:
            parts = Path(match_folder).parts

            if len(parts) < 3:
                return None

            # Extract league/season/match (last 3 parts)
            league = parts[-3]
            season = parts[-2]
            match_name = parts[-1]

            # Try exact match first
            labels_path = os.path.join(
                self.config.soccernet_labels_dir,
                league,
                season,
                match_name,
                "Labels-v2.json"
            )

            if os.path.exists(labels_path):
                return labels_path

            # ✅ Fallback: Fuzzy match (match names might differ slightly)
            base_path = os.path.join(self.config.soccernet_labels_dir, league, season)

            if not os.path.exists(base_path):
                return None

            available_matches = [
                d for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d))
            ]

            # Fuzzy search
            matches = difflib.get_close_matches(match_name, available_matches, n=1, cutoff=0.3)

            if matches:
                fuzzy_path = os.path.join(base_path, matches[0], "Labels-v2.json")
                if os.path.exists(fuzzy_path):
                    return fuzzy_path

            return None

        except Exception as e:
            return None

# ==============================================================================
# TRANSCRIPT LOADER
# ==============================================================================
class TranscriptLoader:
    def __init__(self, config: Config):
        self.config = config

    def load_transcript(self, file_path: str) -> List[Dict]:
        """Load segments from ASR JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = []
        if 'segments' in data:
            for k, v in data['segments'].items():
                segments.append({
                    'start': float(v[0]),
                    'end': float(v[1]),
                    'text': str(v[2]).strip()
                })

        return sorted(segments, key=lambda x: x['start'])

    def create_windows(self, segments: List[Dict]) -> List[Dict]:
        """Create 3-segment context windows"""
        windows = []
        for i in range(1, len(segments) - 1):
            text = f"{segments[i-1]['text']} {segments[i]['text']} {segments[i+1]['text']}"
            windows.append({
                'text': text.strip(),
                'start': segments[i-1]['start'],
                'center': segments[i]['start'],
                'end': segments[i+1]['end']
            })
        return windows

    def load_all(self) -> List[Dict]:
        """Scan recursively for all ASR files"""
        print(f"📂 Scanning: {self.config.dataset_root}")

        pattern = os.path.join(self.config.dataset_root, "**", "*_asr.json")
        files = glob.glob(pattern, recursive=True)

        if self.config.max_matches_to_use:
            files = files[:self.config.max_matches_to_use]
            print(f"⚠️ Limited to {len(files)} matches")

        matches = []
        for f in tqdm(files, desc="Loading transcripts"):
            try:
                # Extract half from filename (1_asr.json → 1)
                base = os.path.basename(f)
                half = int(base.split('_')[0])

                matches.append({
                    'file': f,
                    'half': half,
                    'folder': os.path.dirname(f)
                })
            except:
                continue

        print(f"✅ Found {len(matches)} transcript files\n")
        return matches

# ==============================================================================
# TEMPORAL ALIGNER
# ==============================================================================
class TemporalAligner:
    def __init__(self, config: Config):
        self.config = config

    def align(self, windows: List[Dict], events: List[Dict]) -> List[Dict]:
        """
        Align windows with events using temporal lag.
        Windows without matching events get "No-Event" label.
        """
        aligned = []

        for w in windows:
            label = "No-Event"
            center = w['center']

            # Check if window matches any event
            for e in events:
                # Apply commentator reaction lag
                event_start = e['time'] + self.config.reaction_lag_start
                event_end = e['time'] + self.config.reaction_lag_end

                if event_start <= center <= event_end:
                    label = e['label']
                    break

            aligned.append({**w, 'label': label})

        return aligned

# ==============================================================================
# CLASS BALANCER
# ==============================================================================
class ClassBalancer:
    def __init__(self, config: Config):
        self.config = config

    def balance(self, windows: List[Dict], oversample=True, min_rare=500) -> List[Dict]:
        """
        Balance dataset:
        1. Oversample rare events to min_rare
        2. Downsample No-Event to multiplier * events
        """
        # Group by label
        groups = defaultdict(list)
        for w in windows:
            groups[w['label']].append(w)

        balanced = []
        stats = {}

        # Process event classes
        for label in config.event_classes:
            if label == "No-Event":
                continue

            if label not in groups:
                continue

            items = groups[label]
            original_count = len(items)

            # Oversample if needed
            if oversample and len(items) < min_rare:
                target = min_rare
                mult = target // len(items)
                rem = target % len(items)
                items = items * mult + random.sample(items, rem)
                stats[label] = (original_count, len(items))

            balanced.extend(items)

        # Process No-Event
        n_events = len(balanced)
        n_target_no_event = int(n_events * self.config.no_event_multiplier)

        if "No-Event" in groups:
            no_event_items = groups["No-Event"]

            if len(no_event_items) > n_target_no_event:
                no_event_items = random.sample(no_event_items, n_target_no_event)

            balanced.extend(no_event_items)

        random.shuffle(balanced)

        # Print stats
        total = len(balanced)
        print(f"\n📊 Balance Results:")
        print("-" * 60)

        if stats:
            print("Oversampled:")
            for label, (before, after) in stats.items():
                print(f"  {label:<15}: {before:5,d} → {after:5,d}")
            print()

        final_counts = defaultdict(int)
        for w in balanced:
            final_counts[w['label']] += 1

        print("Final Distribution:")
        for label in config.event_classes:
            count = final_counts[label]
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {label:<15}: {count:7,d} ({pct:5.1f}%)")

        print("-" * 60)
        print(f"Total: {total:,}\n")

        return balanced

# ==============================================================================
# DATASET & METRICS
# ==============================================================================
class DatasetV4(Dataset):
    def __init__(self, data: List[Dict], tokenizer, config: Config):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )

        label_id = self.config.label2id[item['label']]

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """
    Compute metrics with focus on ALL highlight events (Goal, Card, Penalty)
    """
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids

    # Overall metrics
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)

    # ✅ F1 for ALL highlight classes (Goal, Card, Penalty)
    # Label IDs: 0=No-Event, 1=Goal, 2=Card, 3=Penalty
    highlight_labels = [1, 2, 3]  # All event classes
    f1_highlights = f1_score(
        labels,
        predictions,
        labels=highlight_labels,
        average='macro',
        zero_division=0
    )

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_highlights': f1_highlights  # Main metric: optimize for all highlights
    }

# ==============================================================================
# WEIGHTED TRAINER
# ==============================================================================
class WeightedTrainer(Trainer):
    def __init__(self, weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(
            weight=self.weights.to(logits.device) if self.weights is not None else None
        )

        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

# ==============================================================================
# MAIN TRAINING
# ==============================================================================
def main():
    # Mount Drive (Colab)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Colab Drive mounted\n")
    except:
        pass

    # Initialize
    t_loader = TranscriptLoader(config)
    l_loader = SoccerNetLabelLoader(config)
    aligner = TemporalAligner(config)

    # 1. Load all matches
    print("="*70)
    print("[1/6] LOADING MATCHES")
    print("="*70)

    raw_files = t_loader.load_all()

    if not raw_files:
        print("❌ No matches found! Check dataset path.")
        return

    # 2. Split by match
    print("="*70)
    print("[2/6] SPLITTING TRAIN/VAL")
    print("="*70)

    random.shuffle(raw_files)
    split_idx = int(len(raw_files) * 0.85)
    train_files = raw_files[:split_idx]
    val_files = raw_files[split_idx:]

    print(f"Train matches: {len(train_files)}")
    print(f"Val matches:   {len(val_files)}\n")

    # 3. Process data
    def prepare_data(files: List[Dict], is_train: bool) -> List[Dict]:
        """Process matches into labeled windows"""
        windows = []

        desc = "Processing Train" if is_train else "Processing Val"

        for match in tqdm(files, desc=desc):
            # Find labels
            lbl_path = l_loader.find_labels_file(match['folder'])

            if not lbl_path:
                # Skip matches without labels
                continue

            # Load events
            events = l_loader.load_labels(lbl_path)
            events = [e for e in events if e['half'] == match['half']]

            # ✅ DON'T skip even if no events - we need No-Event samples!
            # Load transcript
            segments = t_loader.load_transcript(match['file'])
            wins = t_loader.create_windows(segments)

            # Align (creates No-Event labels automatically)
            aligned = aligner.align(wins, events)
            windows.extend(aligned)

        # Balance
        balancer = ClassBalancer(config)
        min_rare = 500 if is_train else 0

        return balancer.balance(
            windows,
            oversample=is_train,
            min_rare=min_rare
        )

    print("="*70)
    print("[3/6] PREPARING TRAIN DATA")
    print("="*70)
    train_data = prepare_data(train_files, is_train=True)

    print("="*70)
    print("[4/6] PREPARING VAL DATA")
    print("="*70)
    val_data = prepare_data(val_files, is_train=False)

    print(f"\n✅ Train size: {len(train_data):,}")
    print(f"✅ Val size:   {len(val_data):,}\n")

    # 4. Calculate class weights
    print("="*70)
    print("[5/6] CALCULATING WEIGHTS")
    print("="*70)

    train_labels = [config.label2id[x['label']] for x in train_data]
    unique_labels = np.unique(train_labels)

    # Use sklearn
    class_weights_raw = compute_class_weight(
        'balanced',
        classes=unique_labels,
        y=train_labels
    )

    # Soften weights
    class_weights = class_weights_raw ** 0.5

    # Create full weight tensor
    weights_list = []
    print("\nClass Weights:")
    print("-" * 60)

    for i in range(4):
        if i in unique_labels:
            idx = list(unique_labels).index(i)
            w = class_weights[idx]
        else:
            w = 1.0

        weights_list.append(w)
        count = np.sum(np.array(train_labels) == i)
        print(f"  {config.event_classes[i]:<15}: {w:5.2f} (n={count:,})")

    print("-" * 60 + "\n")

    weights_tensor = torch.FloatTensor(weights_list)

    # 5. Train
    print("="*70)
    print("[6/6] TRAINING")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_ds = DatasetV4(train_data, tokenizer, config)
    val_ds = DatasetV4(val_data, tokenizer, config)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=4,
        id2label=config.id2label,
        label2id=config.label2id
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,

        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="f1_highlights",  # Optimize for Goal/Card/Penalty
        greater_is_better=True,

        # ✅ FIX: Disable fused optimizer (not compatible with XLA/TPU)
        optim="adamw_torch",  # Use standard PyTorch Adam instead of fused

        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = WeightedTrainer(
        weights=weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train!
    print("\n🚀 Starting training...\n")
    trainer.train()

    # Save
    final_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    # Save metadata
    with open(os.path.join(final_path, "training_params.json"), "w") as f:
        json.dump({
            "classes": config.event_classes,
            "id2label": config.id2label,
            "label2id": config.label2id
        }, f, indent=2)

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Model saved to: {final_path}")
    print("="*70)

if __name__ == "__main__":
    main()
