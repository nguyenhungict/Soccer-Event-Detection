# Config_Fixed.py
# ✅ Config với balance strategy mới
# Thay thế cell Config (cell 7, dòng ~342-390) trong notebook

from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    """
    Configuration cho Soccer Event Detection với temporal alignment
    
    KEY CHANGES:
    - no_event_multiplier: Downsample No-Event = multiplier * events
    - max_matches_to_use: Giới hạn số matches để train nhanh (optional)
    - num_epochs: Giảm xuống 3 cho iteration đầu
    """
    
    # ==================== PATHS ====================
    dataset_root: str = "./dataset/sn-echoes/Dataset"
    whisper_folders: List[str] = None
    soccernet_labels_dir: str = "./dataset/soccernet"
    output_dir: str = "./models/soccer_event_temporal"
    
    # ==================== TEMPORAL ALIGNMENT ====================
    reaction_lag_start: int = 1  # Commentator reaction bắt đầu sau 1s
    reaction_lag_end: int = 6    # Kết thúc sau 6s
    context_window_size: int = 3  # 3 segments = prev + current + next
    
    # ==================== BALANCE STRATEGY (✅ FIXED) ====================
    # TRƯỚC: no_event_keep_ratio = 0.15 (giữ 15% No-Event)
    # SAU: no_event_multiplier = 2 (No-Event = 2x events)
    
    no_event_multiplier: int = 2  
    # Nghĩa là: No-Event samples = 2 * Event samples
    # Ví dụ: 14,000 events → 28,000 No-Events → Ratio ~40-50% ✅
    
    # Alternative: Dùng target ratio
    # no_event_target_ratio: float = 0.4  # No-Event = 40% total
    
    # ==================== MODEL ====================
    model_name: str = "xlm-roberta-base"
    max_length: int = 160  # Token length
    
    # ==================== TRAINING ====================
    batch_size: int = 8  # Giảm từ 16 → 8 để tiết kiệm VRAM
    learning_rate: float = 2e-5
    num_epochs: int = 3  # Giảm từ 5 → 3 cho iteration đầu
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    train_val_split: float = 0.8
    
    # ==================== OPTIMIZATION (✅ NEW) ====================
    # Giới hạn số matches để train nhanh hơn (optional)
    max_matches_to_use: int = None  # None = use all, 300 = limit to 300
    
    # Gradient accumulation để mô phỏng batch lớn với VRAM nhỏ
    gradient_accumulation_steps: int = 4  # Effective batch = 8 * 4 = 32
    
    # ==================== EVENT CLASSES ====================
    event_classes: List[str] = None
    
    def __post_init__(self):
        """Initialize derived attributes"""
        
        # Default whisper folders
        if self.whisper_folders is None:
            self.whisper_folders = ["whisper_v1_en", "whisper_v2_en"]
        
        # Event classes
        if self.event_classes is None:
            self.event_classes = [
                "No-Event", 
                "Goal", 
                "Yellow card", 
                "Red card", 
                "Substitution", 
                "Penalty"
            ]
        
        # Label mappings
        self.label2id = {label: idx for idx, label in enumerate(self.event_classes)}
        self.id2label = {idx: label for idx, label in enumerate(self.event_classes)}
        
        # Validation
        if self.no_event_multiplier < 1:
            raise ValueError("no_event_multiplier must be >= 1")
        
        if self.no_event_multiplier > 5:
            print("⚠️ WARNING: no_event_multiplier > 5 may cause imbalance")


# ============================================================================
# USAGE IN NOTEBOOK
# ============================================================================
# Replace the config cell with:
"""
config = Config()
print(f"✅ Config loaded (Fixed Version)")
print(f"   Balance strategy: No-Event = {config.no_event_multiplier}x Events")
if config.max_matches_to_use:
    print(f"   ⚠️ Limited to {config.max_matches_to_use} matches")
"""


# ============================================================================
# QUICK PRESETS
# ============================================================================

def get_fast_config():
    """Config for quick iteration (30-40 min)"""
    return Config(
        max_matches_to_use=300,
        num_epochs=2,
        no_event_multiplier=2
    )

def get_full_config():
    """Config for full training (2h)"""
    return Config(
        max_matches_to_use=None,
        num_epochs=5,
        no_event_multiplier=2
    )

def get_aggressive_config():
    """Config for rare event detection"""
    return Config(
        max_matches_to_use=None,
        num_epochs=5,
        no_event_multiplier=1  # More focus on events
    )


# Example usage:
# config = get_fast_config()  # For testing
# config = get_full_config()  # For production
