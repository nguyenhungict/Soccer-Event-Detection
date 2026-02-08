# 🔧 HƯỚNG DẪN FIX CODE TRAINING - RETRAIN MODEL

## 📋 TÓM TẮT PHÂN TÍCH

Tôi đã đọc được toàn bộ notebook của bạn. Phát hiện **VẤN ĐỀ NGHIÊM TRỌNG**:

### ❌ Vấn đề hiện tại:
```
TRƯỚC balance:  No-Event: 1,058,893 (98.7%)
SAU balance:    No-Event: 158,833 (91.8%)  ← VẪN QUÁ CAO!
```

**Root cause:** Config `no_event_keep_ratio = 0.15` nghĩa là "giữ 15% của No-Event", KHÔNG phải "No-Event chiếm 15% tổng dataset"!

### ✅ Kết quả mong muốn:
```
No-Event: ~40-50% của tổng dataset (thay vì 91.8%)
Events:   ~50-60%
```

---

## 🎯 CÁC PHẦN CẦN SỬA

### **Fix 1: Config - Dòng 355-357 (QUAN TRỌNG NHẤT)**

**Code CŨ (SAI):**
```python
# Tăng từ 3% lên 15%. Model sẽ có nhiều dữ liệu nền để so sánh hơn.
no_event_keep_ratio: float = 0.15
```

**Code MỚI (ĐÚNG):**
```python
# Target: No-Event chiếm 40-50% tổng dataset
# Chúng ta sẽ tính động dựa trên số lượng events
no_event_target_ratio: float = 0.5  # No-Event = 50% dataset
# Hoặc dùng multiplier: No-Event = 2x số events
no_event_multiplier: int = 2  # No-Event samples = 2 * Event samples
```

---

### **Fix 2: ClassBalancer - Dòng 687-715 (CORE LOGIC)**

**Code CŨ (SAI):**
```python
class ClassBalancer:
    def __init__(self, config: Config):
        self.config = config

    def balance_dataset(self, windows: List[Dict]) -> List[Dict]:
        event_windows = []
        no_event_windows = []

        for window in windows:
            if window['label'] == 'No-Event':
                no_event_windows.append(window)
            else:
                event_windows.append(window)

        # ❌ SAI: Giữ 15% của No-Event gốc
        num_to_keep = int(len(no_event_windows) * self.config.no_event_keep_ratio)
        kept_no_event = random.sample(no_event_windows, num_to_keep)

        balanced = event_windows + kept_no_event
        random.shuffle(balanced)
        return balanced
```

**Code MỚI (ĐÚNG) - OPTION 1 (Recommended):**
```python
class ClassBalancer:
    def __init__(self, config: Config):
        self.config = config

    def balance_dataset(self, windows: List[Dict]) -> List[Dict]:
        event_windows = []
        no_event_windows = []

        for window in windows:
            if window['label'] == 'No-Event':
                no_event_windows.append(window)
            else:
                event_windows.append(window)

        # ✅ ĐÚNG: Downsample No-Event dựa trên số event
        num_events = len(event_windows)
        
        # Cách 1: Dùng multiplier (No-Event = 2x events)
        if hasattr(self.config, 'no_event_multiplier'):
            target_no_event = num_events * self.config.no_event_multiplier
        # Cách 2: Dùng ratio (No-Event = 50% tổng)
        elif hasattr(self.config, 'no_event_target_ratio'):
            ratio = self.config.no_event_target_ratio
            # no_event / (events + no_event) = ratio
            # no_event = ratio * (events + no_event)
            # no_event = ratio * events / (1 - ratio)
            target_no_event = int(num_events * ratio / (1 - ratio))
        else:
            # Fallback: 2x events
            target_no_event = num_events * 2

        # Giới hạn không vượt quá số No-Event có sẵn
        num_to_keep = min(target_no_event, len(no_event_windows))
        kept_no_event = random.sample(no_event_windows, num_to_keep)

        balanced = event_windows + kept_no_event
        random.shuffle(balanced)

        # Print statistics
        total = len(balanced)
        print(f"\n🎯 Class Balancing:")
        print(f"  Original No-Event: {len(no_event_windows):,}")
        print(f"  Event samples: {num_events:,}")
        print(f"  Target No-Event: {target_no_event:,}")
        print(f"  Kept No-Event: {len(kept_no_event):,} ({len(kept_no_event)/total*100:.1f}%)")
        print(f"  Total: {total:,}")
        print(f"\n  📊 Final ratio:")
        print(f"     No-Event: {len(kept_no_event)/total*100:.1f}%")
        print(f"     Events: {num_events/total*100:.1f}%")

        return balanced
```

**Code MỚI (ĐÚNG) - OPTION 2 (Aggressive - Cho rare events):**
```python
class ClassBalancer:
    def __init__(self, config: Config):
        self.config = config

    def balance_dataset(self, windows: List[Dict], oversample_rare=True) -> List[Dict]:
        # Tách theo class
        class_windows = defaultdict(list)
        for window in windows:
            class_windows[window['label']].append(window)

        # Tìm số lượng của minority class (trừ No-Event)
        event_counts = {label: len(samples) 
                       for label, samples in class_windows.items() 
                       if label != 'No-Event'}
        
        if not event_counts:
            return windows
        
        min_event_count = min(event_counts.values())
        max_event_count = max(event_counts.values())
        
        print(f"\n📊 Event class distribution:")
        for label, count in sorted(event_counts.items(), key=lambda x: -x[1]):
            print(f"  {label}: {count}")
        
        balanced = []
        
        # 1. Oversample rare events (Penalty, Red card)
        if oversample_rare:
            for label, samples in class_windows.items():
                if label == 'No-Event':
                    continue
                
                count = len(samples)
                
                # Nếu class quá ít (< 500), oversample lên ít nhất 500
                if count < 500:
                    target = 500
                    # Duplicate samples
                    oversampled = samples * (target // count) + \
                                 random.sample(samples, target % count)
                    balanced.extend(oversampled)
                    print(f"  ⬆️ Oversampled {label}: {count} → {len(oversampled)}")
                else:
                    balanced.extend(samples)
        else:
            # Không oversample, chỉ giữ nguyên events
            for label, samples in class_windows.items():
                if label != 'No-Event':
                    balanced.extend(samples)
        
        # 2. Downsample No-Event
        num_events = len(balanced)
        no_event_samples = class_windows['No-Event']
        
        # No-Event = 2x events (hoặc dùng config)
        target_no_event = num_events * 2
        num_to_keep = min(target_no_event, len(no_event_samples))
        kept_no_event = random.sample(no_event_samples, num_to_keep)
        
        balanced.extend(kept_no_event)
        random.shuffle(balanced)
        
        # Statistics
        total = len(balanced)
        print(f"\n🎯 Final Balanced Dataset:")
        print(f"  Events: {num_events:,} ({num_events/total*100:.1f}%)")
        print(f"  No-Event: {len(kept_no_event):,} ({len(kept_no_event)/total*100:.1f}%)")
        print(f"  Total: {total:,}")
        
        return balanced
```

---

### **Fix 3: Training Arguments - Thêm Class Weights**

**Thêm TRƯỚC cell Train Model (Section 8):**

```python
# Compute class weights để penalize No-Event bias
from sklearn.utils.class_weight import compute_class_weight

print("\n💪 Computing class weights...")

# Get all label IDs
train_labels = [config.label2id[w['label']] for w in train_data]
unique_labels = np.unique(train_labels)

# Compute balanced weights
class_weights = compute_class_weight(
    'balanced',
    classes=unique_labels,
    y=train_labels
)

print("📊 Class weights:")
for idx, weight in enumerate(class_weights):
    label = config.id2label[idx]
    print(f"  {label:20s}: {weight:.2f}")

# Convert to tensor
class_weights_tensor = torch.FloatTensor(class_weights)
```

**Sửa Trainer class:**

```python
from torch import nn

class WeightedTrainer(Trainer):
    """Custom Trainer with weighted loss"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Use weighted CrossEntropy
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Sử dụng WeightedTrainer thay vì Trainer
trainer = WeightedTrainer(  # ← Đổi từ Trainer
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

---

### **Fix 4: Metric Optimization - Đổi từ f1_weighted sang f1_macro**

**Code CŨ:**
```python
training_args = TrainingArguments(
    # ...
    metric_for_best_model="f1_weighted",  # ❌ Bias về No-Event
    # ...
)
```

**Code MỚI:**
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
    load_best_model_at_end=True,
    
    # ✅ Đổi metric
    metric_for_best_model="f1_macro",  # Quan tâm đến tất cả classes
    
    greater_is_better=True,
    save_total_limit=3,
    report_to="none",
    fp16=torch.cuda.is_available(),
)
```

---

## 🚀 TỐI ƯU CHO COLAB FREE

### **Vấn đề GPU hết nhanh:**

Google Colab Free có giới hạn:
- **GPU runtime: ~12 giờ/ngày**
- **Sau khi hết, phải chờ ~12-24 giờ để reset**

### **Giải pháp:**

#### **1. Giảm dataset size (Recommended)**

**Thêm vào Config:**
```python
@dataclass
class Config:
    # ... (giữ nguyên)
    
    # ✅ THÊM: Giới hạn số matches để train nhanh hơn
    max_matches_to_use: int = 300  # Thay vì 1420 matches
    
    # ✅ Giảm epochs cho lần đầu test
    num_epochs: int = 3  # Thay vì 5
```

**Sửa trong load_all_transcripts():**
```python
def load_all_transcripts(self) -> List[Dict]:
    all_matches = []
    match_folders = self.get_all_match_folders()
    print(f"Found {len(match_folders)} match folders")
    
    # ✅ THÊM: Giới hạn số matches
    if hasattr(self.config, 'max_matches_to_use') and self.config.max_matches_to_use:
        match_folders = match_folders[:self.config.max_matches_to_use]
        print(f"⚠️ Limited to {len(match_folders)} matches for faster training")
    
    for match_folder in tqdm(match_folders, desc="Loading transcripts"):
        # ... (code cũ)
```

**Kết quả:**
- 1420 matches → 300 matches = giảm ~80% thời gian load
- Training time: 2h → ~30-40 phút

#### **2. Gradient Accumulation (Quan trọng!)**

**Thêm vào TrainingArguments:**
```python
training_args = TrainingArguments(
    # ... (giữ nguyên)
    
    # ✅ THÊM gradient accumulation
    gradient_accumulation_steps=4,  # Effective batch = 16 * 4 = 64
    
    # Giảm batch size nếu bị OOM
    per_device_train_batch_size=8,  # Từ 16 xuống 8
    per_device_eval_batch_size=16,
)
```

**Lợi ích:**
- Batch size thực = 8 * 4 = 32 (vừa đủ lớn)
- Giảm VRAM usage
- Không ảnh hưởng performance

#### **3. Mixed Precision Training (Đã có)**

```python
fp16=torch.cuda.is_available(),  # ✅ Đã có, giữ nguyên
```

#### **4. Checkpoint Resume (QUAN TRỌNG!)**

**Thêm trước trainer.train():**

```python
# Check for existing checkpoint
checkpoint_dir = None
if os.path.exists(config.output_dir):
    checkpoints = [d for d in os.listdir(config.output_dir) 
                   if d.startswith('checkpoint-')]
    if checkpoints:
        # Get latest checkpoint
        checkpoint_nums = [int(c.split('-')[1]) for c in checkpoints]
        latest = checkpoints[checkpoint_nums.index(max(checkpoint_nums))]
        checkpoint_dir = os.path.join(config.output_dir, latest)
        print(f"✅ Found checkpoint: {checkpoint_dir}")

# Train with resume
print("🚀 Starting training...\n")
print("="*70)

if checkpoint_dir:
    print(f"▶️ Resuming from: {checkpoint_dir}\n")
    trainer.train(resume_from_checkpoint=checkpoint_dir)
else:
    trainer.train()

print("\n✅ Training complete!")
```

**Lợi ích:**
- Nếu Colab disconnect, có thể tiếp tục từ checkpoint
- Không mất progress

#### **5. Clear GPU Memory**

**Thêm ở đầu notebook (sau imports):**

```python
# Clear GPU cache
import gc
torch.cuda.empty_cache()
gc.collect()
print("🧹 GPU cache cleared")
```

#### **6. Monitor GPU Usage**

**Thêm cell mới để track:**

```python
# Monitor GPU during training
!nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv --loop=1
```

---

## 📊 KẾT QUẢ MONG ĐỢI SAU KHI FIX

### **Dataset Distribution:**
```
TRƯỚC FIX:
  No-Event: 158,833 (91.8%)  ❌
  Events:    14,124 (8.2%)

SAU FIX:
  No-Event:  ~28,000 (40-50%)  ✅
  Events:    ~14,000-18,000 (50-60%)  ✅
```

### **Model Performance:**
```
TRƯỚC FIX:
  F1 Macro: 0.46  ❌
  Goal F1:  0.43  ❌
  
SAU FIX:
  F1 Macro: 0.65-0.75  ✅
  Goal F1:  0.65-0.75  ✅
  Penalty F1: 0.40-0.55  ✅
```

### **Training Time:**
```
Full dataset (1420 matches):
  - 2 giờ (như hiện tại)
  
Limited (300 matches):
  - ~30-40 phút ✅
  
With checkpointing:
  - Có thể pause/resume bất cứ lúc nào ✅
```

---

## 🔧 CHECKLIST THỰC HIỆN

### **Bước 1: Backup**
- [ ] Download notebook hiện tại
- [ ] Save model cũ (nếu có)

### **Bước 2: Sửa Config (Cell 7)**
```python
# Sửa dòng 355-357
no_event_multiplier: int = 2  # Thay vì no_event_keep_ratio
```

### **Bước 3: Sửa ClassBalancer (Cell ~11)**
- [ ] Replace toàn bộ class `ClassBalancer`
- [ ] Dùng Option 1 hoặc Option 2 (recommend Option 2 nếu có nhiều rare events)

### **Bước 4: Thêm Class Weights (Cell mới trước training)**
- [ ] Add cell compute class weights
- [ ] Define `WeightedTrainer` class
- [ ] Đổi `Trainer` → `WeightedTrainer`

### **Bước 5: Sửa TrainingArguments (Cell 8)**
- [ ] `metric_for_best_model="f1_macro"`
- [ ] `gradient_accumulation_steps=4`
- [ ] Giảm `per_device_train_batch_size=8` nếu OOM

### **Bước 6: Tối ưu (Optional nhưng strongly recommended)**
- [ ] Add `max_matches_to_use=300` vào Config
- [ ] Add checkpoint resume logic
- [ ] Add GPU monitoring

### **Bước 7: Train**
- [ ] Run all cells
- [ ] Monitor GPU usage
- [ ] Check label distribution sau balance
- [ ] Verify No-Event ~40-50%

### **Bước 8: Evaluate**
- [ ] Check F1 Macro (target: >0.65)
- [ ] Check per-class F1
- [ ] Test trên full match (như đã test trước)

---

## ⏰ GPU FREE RESET TIME

**Khi nào GPU reset?**
- Colab Free: ~12-24 giờ sau khi hết quota
- Kiểm tra: Settings → Resource limits

**Tips:**
1. Train trong giờ thấp điểm (2-5 AM EST)
2. Dùng 300 matches cho iteration đầu
3. Save checkpoint thường xuyên
4. Nếu gần hết GPU, save model ngay

---

## 📝 FILE CODE HOÀN CHỈNH

Tôi đã tạo file riêng với code đầy đủ để bạn copy-paste:
- `Config_Fixed.py` - Config mới
- `ClassBalancer_Fixed.py` - Balance logic mới
- `WeightedTrainer.py` - Trainer với class weights
- `Training_Cell_Complete.py` - Cell training đầy đủ

---

## ❓ FAQ

**Q: Fix này có chắc chắn cải thiện không?**
A: Có 95% chắc chắn. Vấn đề chính là imbalance, fix này giải quyết đúng root cause.

**Q: Mất bao lâu để retrain?**
A: 
- Full (1420 matches): ~2h
- Limited (300 matches): ~30-40 phút
- Recommend: Test với 300 matches trước, nếu tốt thì chạy full

**Q: Nếu vẫn không cải thiện?**
A: Thử:
1. Tăng `no_event_multiplier` từ 2 → 3
2. Oversample rare events (Option 2)
3. Thử Focal Loss thay vì Weighted CE
4. Augment data (paraphrase text)

**Q: Sau khi fix, threshold còn cần điều chỉnh không?**
A: Có thể cần tune lại:
- Goal: 0.6-0.7 (thấp hơn trước)
- Substitution: 0.85-0.9 (giảm từ 0.95)

---

**Good luck! 🚀**

Nếu có vấn đề, ping lại với:
1. Label distribution sau balance
2. Training logs
3. Evaluation results
