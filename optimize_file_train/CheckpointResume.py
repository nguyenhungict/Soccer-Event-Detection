# CheckpointResume.py
# ✅ Code để resume training từ checkpoint nếu Colab disconnect
# Thêm vào cell training (Section 8), TRƯỚC trainer.train()

import os
import glob


def find_latest_checkpoint(output_dir):
    """
    Tìm checkpoint mới nhất trong output directory
    
    Args:
        output_dir: Directory chứa checkpoints
    
    Returns:
        str or None: Path to latest checkpoint, or None if not found
    """
    if not os.path.exists(output_dir):
        return None
    
    # Tìm tất cả checkpoints (format: checkpoint-{step})
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    
    if not checkpoints:
        return None
    
    # Extract step numbers
    checkpoint_steps = []
    for cp in checkpoints:
        try:
            step = int(os.path.basename(cp).split('-')[1])
            checkpoint_steps.append((step, cp))
        except (IndexError, ValueError):
            continue
    
    if not checkpoint_steps:
        return None
    
    # Sort by step và lấy checkpoint mới nhất
    checkpoint_steps.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_checkpoint = checkpoint_steps[0]
    
    return latest_checkpoint


def get_checkpoint_info(checkpoint_dir):
    """
    Lấy thông tin về checkpoint
    
    Returns:
        dict: Checkpoint information
    """
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return None
    
    # Parse step from dirname
    try:
        step = int(os.path.basename(checkpoint_dir).split('-')[1])
    except (IndexError, ValueError):
        step = None
    
    # Check files
    has_model = os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin"))
    has_config = os.path.exists(os.path.join(checkpoint_dir, "config.json"))
    has_optimizer = os.path.exists(os.path.join(checkpoint_dir, "optimizer.pt"))
    has_scheduler = os.path.exists(os.path.join(checkpoint_dir, "scheduler.pt"))
    
    return {
        'path': checkpoint_dir,
        'step': step,
        'has_model': has_model,
        'has_config': has_config,
        'has_optimizer': has_optimizer,
        'has_scheduler': has_scheduler,
        'complete': all([has_model, has_config, has_optimizer, has_scheduler])
    }


def clean_incomplete_checkpoints(output_dir):
    """
    Xóa checkpoints không hoàn chỉnh (bị corrupt khi disconnect)
    """
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    
    cleaned = 0
    for cp in checkpoints:
        info = get_checkpoint_info(cp)
        if info and not info['complete']:
            print(f"🗑️ Removing incomplete checkpoint: {os.path.basename(cp)}")
            import shutil
            shutil.rmtree(cp)
            cleaned += 1
    
    if cleaned > 0:
        print(f"✅ Cleaned {cleaned} incomplete checkpoint(s)")
    
    return cleaned


def setup_checkpoint_resume(config):
    """
    Setup checkpoint resume với validation
    
    Returns:
        str or None: Checkpoint path to resume from
    """
    print("\n🔄 Checking for existing checkpoints...")
    print("="*60)
    
    # Clean incomplete checkpoints first
    clean_incomplete_checkpoints(config.output_dir)
    
    # Find latest
    checkpoint_dir = find_latest_checkpoint(config.output_dir)
    
    if checkpoint_dir:
        info = get_checkpoint_info(checkpoint_dir)
        
        if info and info['complete']:
            print(f"✅ Found complete checkpoint:")
            print(f"   Path: {checkpoint_dir}")
            print(f"   Step: {info['step']:,}")
            print(f"   Files: Model ✓ | Config ✓ | Optimizer ✓ | Scheduler ✓")
            print()
            
            # Estimate progress
            if info['step']:
                # Rough estimation (depends on total steps)
                print(f"📊 Training will resume from step {info['step']:,}")
            
            print("="*60)
            return checkpoint_dir
        else:
            print("⚠️ Found checkpoint but it's incomplete. Starting fresh.")
            print("="*60)
            return None
    else:
        print("ℹ️ No checkpoint found. Starting from scratch.")
        print("="*60)
        return None


# ============================================================================
# USAGE IN NOTEBOOK - TRAINING CELL
# ============================================================================
"""
# REPLACE your training cell with this:

# Setup checkpoint resume
checkpoint_path = setup_checkpoint_resume(config)

# Training arguments (same as before)
training_args = TrainingArguments(
    output_dir=config.output_dir,
    num_train_epochs=config.num_epochs,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size * 2,
    learning_rate=config.learning_rate,
    warmup_steps=config.warmup_steps,
    weight_decay=config.weight_decay,
    
    # Logging & Checkpointing
    logging_dir=f"{config.output_dir}/logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,  # Keep only 3 latest checkpoints
    
    # Metrics
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",  # ✅ Changed from f1_weighted
    greater_is_better=True,
    
    # Performance
    gradient_accumulation_steps=4,  # ✅ Added
    fp16=torch.cuda.is_available(),
    
    report_to="none",
)

# Setup trainer (with class weights if available)
if 'class_weights' in globals():
    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

print("✅ Trainer ready")

# Train with resume
print("\n🚀 Starting training...")
print("="*70)

if checkpoint_path:
    print(f"▶️ RESUMING from: {checkpoint_path}\n")
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    print("▶️ STARTING fresh training\n")
    trainer.train()

print("\n✅ Training complete!")
"""


# ============================================================================
# ADVANCED: Auto-save on disconnect detection
# ============================================================================

class DisconnectSafeTrainer(Trainer):
    """
    Trainer that saves checkpoint more frequently for Colab
    """
    
    def __init__(self, *args, save_every_n_steps=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_every_n_steps = save_every_n_steps
        self._last_save_step = 0
    
    def training_step(self, *args, **kwargs):
        """Override to add frequent checkpointing"""
        loss = super().training_step(*args, **kwargs)
        
        # Save every N steps
        if self.state.global_step - self._last_save_step >= self.save_every_n_steps:
            self.save_model()
            self._last_save_step = self.state.global_step
            print(f"💾 Auto-saved checkpoint at step {self.state.global_step}")
        
        return loss


# Usage:
"""
trainer = DisconnectSafeTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    save_every_n_steps=100,  # Save every 100 steps
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
"""


# ============================================================================
# MONITORING
# ============================================================================

def print_gpu_usage():
    """Print current GPU usage (Colab specific)"""
    try:
        import subprocess
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        
        memory_used, memory_total, gpu_util = result.strip().split(',')
        
        print(f"\n🖥️ GPU Status:")
        print(f"  Memory: {memory_used.strip()}MB / {memory_total.strip()}MB")
        print(f"  Utilization: {gpu_util.strip()}%")
        
    except Exception as e:
        print(f"⚠️ Could not get GPU info: {e}")


def estimate_remaining_time(trainer):
    """Estimate remaining training time"""
    if not hasattr(trainer, 'state') or not trainer.state:
        return None
    
    current_step = trainer.state.global_step
    max_steps = trainer.state.max_steps
    
    if current_step == 0 or max_steps == 0:
        return None
    
    # Estimate based on current progress
    progress = current_step / max_steps
    # This is rough, actual time varies
    
    return {
        'current_step': current_step,
        'max_steps': max_steps,
        'progress': progress * 100
    }


# Usage:
"""
# After training starts, in a separate cell:
info = estimate_remaining_time(trainer)
if info:
    print(f"Progress: {info['progress']:.1f}% ({info['current_step']}/{info['max_steps']})")

print_gpu_usage()
"""
