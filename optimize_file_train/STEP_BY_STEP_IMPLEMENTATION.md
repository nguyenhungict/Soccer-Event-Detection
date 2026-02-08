# STEP_BY_STEP_IMPLEMENTATION.md
# 🔧 Hướng dẫn từng bước FIX và RETRAIN model

## 📋 CHUẨN BỊ

### 1. Backup
```bash
# Download notebook hiện tại
# Save model cũ (nếu đã train)
# Screenshot kết quả cũ để so sánh
```

### 2. Hiểu vấn đề
```
HIỆN TẠI:
- No-Event: 158,833 (91.8%) ← QUÁ CAO
- Events: 14,124 (8.2%)
- F1 Macro: 0.46 ← KÉM

MỤC TIÊU:
- No-Event: ~40-50%
- Events: ~50-60%
- F1 Macro: >0.65
```

---

## 🔨 BƯỚC 1: SỬA CONFIG

**Location:** Cell 7 (dòng ~342-390)

### Tìm dòng này:
```python
no_event_keep_ratio: float = 0.15
```

### Đổi thành:
```python
# ✅ FIXED: Downsample No-Event dựa trên số events
no_event_multiplier: int = 2  # No-Event = 2x events

# Optional: Giới hạn matches để train nhanh
max_matches_to_use: int = 300  # Set None để dùng hết

# Giảm epochs cho iteration đầu
num_epochs: int = 3  # Từ 5 xuống 3

# Giảm batch size
batch_size: int = 8  # Từ 16 xuống 8

# Thêm gradient accumulation
gradient_accumulation_steps: int = 4
```

### Code đầy đủ:
```python
@dataclass
class Config:
    dataset_root: str = "./dataset/sn-echoes/Dataset"
    whisper_folders: List[str] = None
    soccernet_labels_dir: str = "./dataset/soccernet"
    output_dir: str = "./models/soccer_event_temporal"
    
    reaction_lag_start: int = 1
    reaction_lag_end: int = 6
    context_window_size: int = 3
    
    # ✅ FIXED
    no_event_multiplier: int = 2
    max_matches_to_use: int = 300  # None for full dataset
    
    model_name: str = "xlm-roberta-base"
    max_length: int = 160
    
    # ✅ OPTIMIZED
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    train_val_split: float = 0.8
    gradient_accumulation_steps: int = 4
    
    event_classes: List[str] = None
    
    def __post_init__(self):
        if self.whisper_folders is None:
            self.whisper_folders = ["whisper_v1_en", "whisper_v2_en"]
        if self.event_classes is None:
            self.event_classes = [
                "No-Event", "Goal", "Yellow card", "Red card", "Substitution", "Penalty"
            ]
        self.label2id = {label: idx for idx, label in enumerate(self.event_classes)}
        self.id2label = {idx: label for idx, label in enumerate(self.event_classes)}

config = Config()
print("✅ Config loaded (Fixed Version)")
print(f"   Balance: No-Event = {config.no_event_multiplier}x Events")
if config.max_matches_to_use:
    print(f"   Limited: {config.max_matches_to_use} matches")
```

---

## 🔨 BƯỚC 2: SỬA CLASS BALANCER

**Location:** Cell ~11 (ClassBalancer class)

### THAY THẾ TOÀN BỘ CLASS:

```python
class ClassBalancer:
    def __init__(self, config):
        self.config = config
    
    def balance_dataset(self, windows, oversample_rare=True, min_rare_samples=500):
        # Tách theo class
        class_windows = defaultdict(list)
        for window in windows:
            class_windows[window['label']].append(window)
        
        # Thống kê events
        event_counts = {label: len(samples) 
                       for label, samples in class_windows.items() 
                       if label != 'No-Event'}
        
        if not event_counts:
            return windows
        
        print(f"\n📊 Original Event Distribution:")
        for label, count in sorted(event_counts.items(), key=lambda x: -x[1]):
            print(f"  {label:20s}: {count:6,d}")
        
        balanced = []
        oversampled_stats = {}
        
        # Step 1: Oversample rare events
        for label, samples in class_windows.items():
            if label == 'No-Event':
                continue
            
            count = len(samples)
            
            if oversample_rare and count < min_rare_samples:
                target = min_rare_samples
                num_copies = target // count
                remainder = target % count
                
                oversampled = samples * num_copies
                if remainder > 0:
                    oversampled.extend(random.sample(samples, remainder))
                
                balanced.extend(oversampled)
                oversampled_stats[label] = (count, len(oversampled))
            else:
                balanced.extend(samples)
        
        # Step 2: Downsample No-Event
        num_events = len(balanced)
        no_event_samples = class_windows['No-Event']
        
        multiplier = getattr(self.config, 'no_event_multiplier', 2)
        target_no_event = num_events * multiplier
        num_to_keep = min(target_no_event, len(no_event_samples))
        
        kept_no_event = random.sample(no_event_samples, num_to_keep)
        balanced.extend(kept_no_event)
        random.shuffle(balanced)
        
        # Statistics
        total = len(balanced)
        print(f"\n🎯 BALANCING RESULTS:")
        print("="*60)
        
        if oversampled_stats:
            print("📈 Oversampled:")
            for label, (before, after) in oversampled_stats.items():
                print(f"  {label:20s}: {before:6,d} → {after:6,d}")
            print()
        
        print(f"📊 Final:")
        print(f"  Events:   {num_events:,d} ({num_events/total*100:.1f}%)")
        print(f"  No-Event: {len(kept_no_event):,d} ({len(kept_no_event)/total*100:.1f}%)")
        print(f"  Total:    {total:,d}")
        
        final_counts = defaultdict(int)
        for w in balanced:
            final_counts[w['label']] += 1
        
        print(f"\n📋 Per-class:")
        for label, count in sorted(final_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            print(f"  {label:20s}: {count:7,d} ({pct:5.1f}%)")
        print("="*60)
        
        no_event_ratio = len(kept_no_event) / total
        if 0.3 <= no_event_ratio <= 0.6:
            print("✅ Balance looks good!")
        else:
            print("⚠️ WARNING: Check no_event_multiplier")
        
        return balanced

print("✅ ClassBalancer (Fixed v2) defined")
```

---

## 🔨 BƯỚC 3: THÊM CLASS WEIGHTS

**Location:** Cell MỚI, thêm SAU cell load model, TRƯỚC cell training

### Thêm cell MỚI:

```python
# ============================================================================
# COMPUTE CLASS WEIGHTS
# ============================================================================

import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

print("\n💪 Computing class weights...")

# Get labels from training data
train_labels = [config.label2id[w['label']] for w in train_data]
unique_labels = np.unique(train_labels)

# Compute balanced weights
class_weights = compute_class_weight(
    'balanced',
    classes=unique_labels,
    y=train_labels
)

print("\n📊 Class Weights:")
print("-" * 50)
for idx, weight in enumerate(class_weights):
    label = config.id2label[idx]
    count = train_labels.count(idx)
    print(f"  {label:20s}: {weight:6.2f} (n={count:,})")
print("-" * 50)

# Convert to tensor
class_weights_tensor = torch.FloatTensor(class_weights)

# ============================================================================
# WEIGHTED TRAINER
# ============================================================================

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        print("✅ WeightedLossTrainer initialized")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )
        else:
            loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                       labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

print("✅ Class weights computed")
```

---

## 🔨 BƯỚC 4: SỬA TRAINING ARGUMENTS

**Location:** Cell 8 (Training cell)

### Tìm và sửa:

```python
training_args = TrainingArguments(
    output_dir=config.output_dir,
    num_train_epochs=config.num_epochs,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size * 2,
    learning_rate=config.learning_rate,
    warmup_steps=config.warmup_steps,
    weight_decay=config.weight_decay,
    
    logging_dir=f"{config.output_dir}/logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    
    load_best_model_at_end=True,
    
    # ✅ CHANGED: f1_weighted → f1_macro
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    
    # ✅ ADDED: Gradient accumulation
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    
    fp16=torch.cuda.is_available(),
    report_to="none",
)
```

### Đổi Trainer:

```python
# ✅ CHANGED: Trainer → WeightedLossTrainer
trainer = WeightedLossTrainer(
    class_weights=class_weights_tensor,  # ✅ Add class weights
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("✅ Trainer ready with weighted loss")
```

---

## 🔨 BƯỚC 5: THÊM CHECKPOINT RESUME (Optional nhưng recommended)

**Location:** Trong cell training, TRƯỚC trainer.train()

### Thêm code này:

```python
# ============================================================================
# CHECKPOINT RESUME
# ============================================================================

def find_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    
    checkpoint_steps = []
    for cp in checkpoints:
        try:
            step = int(os.path.basename(cp).split('-')[1])
            checkpoint_steps.append((step, cp))
        except:
            continue
    
    if checkpoint_steps:
        checkpoint_steps.sort(reverse=True)
        return checkpoint_steps[0][1]
    return None

# Check for checkpoint
checkpoint_path = find_latest_checkpoint(config.output_dir)

if checkpoint_path:
    print(f"✅ Found checkpoint: {checkpoint_path}")
    print("   Will resume training...")
else:
    print("ℹ️ No checkpoint found. Starting fresh.")

# ============================================================================
# TRAIN
# ============================================================================

print("\n🚀 Starting training...")
print("="*70)

if checkpoint_path:
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    trainer.train()

print("\n✅ Training complete!")
```

---

## 🔨 BƯỚC 6: OPTIONAL - GIỚI HẠN SỐ MATCHES

**Location:** Cell load transcripts (Section 7)

### Sửa trong TranscriptLoader:

```python
def load_all_transcripts(self) -> List[Dict]:
    all_matches = []
    match_folders = self.get_all_match_folders()
    print(f"Found {len(match_folders)} match folders")
    
    # ✅ ADDED: Limit matches
    if hasattr(self.config, 'max_matches_to_use') and self.config.max_matches_to_use:
        match_folders = match_folders[:self.config.max_matches_to_use]
        print(f"⚠️ Limited to {len(match_folders)} matches for faster training")
    
    for match_folder in tqdm(match_folders, desc="Loading transcripts"):
        # ... rest of code
```

---

## ✅ VERIFICATION CHECKLIST

Sau khi sửa, CHECK các điểm sau TRƯỚC KHI TRAIN:

### [ ] Config
- [ ] `no_event_multiplier = 2` (không phải no_event_keep_ratio)
- [ ] `batch_size = 8`
- [ ] `num_epochs = 3`
- [ ] `gradient_accumulation_steps = 4`
- [ ] `max_matches_to_use = 300` (hoặc None cho full)

### [ ] ClassBalancer
- [ ] Balance logic dùng multiplier, không phải ratio
- [ ] Có oversample cho rare events
- [ ] Print ra statistics đầy đủ

### [ ] Training Setup
- [ ] Class weights được compute
- [ ] WeightedLossTrainer được dùng
- [ ] `metric_for_best_model = "f1_macro"`
- [ ] Có checkpoint resume logic

### [ ] Dataset Loading
- [ ] Có giới hạn matches (nếu muốn train nhanh)
- [ ] Load thành công transcripts và labels

---

## 🚀 CHẠY TRAINING

### 1. Run từng cell từ đầu
```python
# Cell 1: GPU check ✓
# Cell 2: Install dependencies ✓
# Cell 3: Mount Drive ✓
# Cell 4: Download labels (skip nếu đã có) ✓
# Cell 5: Check dataset ✓
# Cell 6: Imports ✓
# Cell 7: Config (FIXED) ✓
# Cells 8-12: Helper classes (FIXED ClassBalancer) ✓
# Cell 13: Load transcripts ✓
# Cell 14: Temporal alignment ✓
# Cell 15: Balance (CHECK OUTPUT!) ✓
# Cell 16: Split data ✓
# Cell 17: Load model ✓
# Cell NEW: Compute class weights ✓
# Cell 18: Training (FIXED) ✓
```

### 2. Monitor sau Cell 15 (Balance)

**PHẢI THẤY:**
```
📊 Final:
  Events:   ~14,000-18,000 (50-60%)  ← QUAN TRỌNG!
  No-Event: ~14,000-28,000 (40-50%)  ← QUAN TRỌNG!
  Total:    ~28,000-36,000

✅ Balance looks good!
```

**NẾU THẤY:**
```
No-Event: 91.8%  ← VẪN SAI!
```
→ STOP, kiểm tra lại ClassBalancer

### 3. Monitor sau Cell Class Weights

**PHẢI THẤY:**
```
📊 Class Weights:
  No-Event:  0.50-1.00
  Goal:      2.00-4.00  ← Higher than No-Event
  Penalty:   10.00-30.00  ← Highest
```

### 4. Monitor training logs

**Good signs:**
```
Epoch 1/3: Loss giảm dần
Eval: F1 Macro tăng dần (0.50 → 0.60 → 0.70)
GPU usage: 70-90%
```

**Bad signs:**
```
Loss không giảm hoặc tăng → Check learning rate
GPU usage: <50% → Check batch size
OOM error → Giảm batch_size xuống 4
```

---

## 📊 KẾT QUẢ MONG ĐỢI

### Sau Balance:
```
TRƯỚC:
  No-Event: 158,833 (91.8%)  ❌

SAU:
  No-Event: ~28,000 (40-50%)  ✅
  Events: ~28,000 (50-60%)  ✅
```

### Sau Training:
```
TRƯỚC:
  F1 Macro: 0.46
  Goal F1: 0.43
  Penalty F1: 0.21

SAU (Target):
  F1 Macro: 0.65-0.75  ✅
  Goal F1: 0.65-0.75  ✅
  Penalty F1: 0.40-0.55  ✅
```

### Training Time:
```
300 matches: 30-40 phút
1420 matches: ~2 giờ
```

---

## ⚠️ TROUBLESHOOTING

### Vấn đề 1: No-Event vẫn 90%+
**Nguyên nhân:** ClassBalancer không được sửa đúng
**Fix:** Copy lại code ClassBalancer từ file `ClassBalancer_Fixed_v2.py`

### Vấn đề 2: OOM (Out of Memory)
**Nguyên nhân:** Batch size quá lớn
**Fix:** 
```python
batch_size: int = 4  # Giảm xuống 4
gradient_accumulation_steps: int = 8  # Tăng lên 8
```

### Vấn đề 3: Training quá chậm
**Fix:**
```python
max_matches_to_use: int = 100  # Giảm xuống 100
num_epochs: int = 2
```

### Vấn đề 4: F1 Macro không cải thiện
**Thử:**
1. Tăng `no_event_multiplier` từ 2 → 1
2. Tăng `min_rare_samples` từ 500 → 800
3. Thử Focal Loss thay vì Weighted CE

### Vấn đề 5: Colab disconnect
**Fix:** 
- Đã có checkpoint resume → Chạy lại, nó sẽ tự resume
- Nếu không có checkpoint → Phải train lại từ đầu

---

## 📞 SUPPORT

Nếu gặp vấn đề, cần check:
1. **Label distribution sau balance** - Paste output Cell 15
2. **Class weights** - Paste output cell compute weights
3. **Training logs** - Screenshot epoch 1-2
4. **Evaluation results** - Classification report

Good luck! 🚀
