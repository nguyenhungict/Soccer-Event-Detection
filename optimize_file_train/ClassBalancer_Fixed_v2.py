# ClassBalancer_Fixed_v2.py
# ✅ AGGRESSIVE BALANCING với Oversample cho rare events
# Copy code này thay thế cell ClassBalancer (cell ~11) trong notebook

from collections import defaultdict
import random

class ClassBalancer:
    """
    Balanced dataset với chiến lược:
    1. Oversample rare events (Penalty, Red card) lên ít nhất 500 samples
    2. Giữ nguyên common events (Goal, Yellow card, Substitution)
    3. Downsample No-Event = 2x tổng số events
    
    Kết quả: No-Event ~40-50%, Events ~50-60%
    """
    
    def __init__(self, config):
        self.config = config
    
    def balance_dataset(self, windows, oversample_rare=True, min_rare_samples=500):
        """
        Balance dataset with aggressive oversampling
        
        Args:
            windows: List of labeled windows
            oversample_rare: Whether to oversample rare events
            min_rare_samples: Minimum samples for rare classes
        
        Returns:
            Balanced list of windows
        """
        # Tách theo class
        class_windows = defaultdict(list)
        for window in windows:
            class_windows[window['label']].append(window)
        
        # Thống kê event classes (trừ No-Event)
        event_counts = {label: len(samples) 
                       for label, samples in class_windows.items() 
                       if label != 'No-Event'}
        
        if not event_counts:
            print("⚠️ No events found! Returning original windows.")
            return windows
        
        print(f"\n📊 Original Event Distribution:")
        for label, count in sorted(event_counts.items(), key=lambda x: -x[1]):
            print(f"  {label:20s}: {count:6,d}")
        
        balanced = []
        oversampled_stats = {}
        
        # Step 1: Handle event classes
        for label, samples in class_windows.items():
            if label == 'No-Event':
                continue
            
            count = len(samples)
            
            # Oversample rare events
            if oversample_rare and count < min_rare_samples:
                target = min_rare_samples
                
                # Duplicate với randomization
                num_copies = target // count
                remainder = target % count
                
                oversampled = []
                
                # Full copies
                for _ in range(num_copies):
                    oversampled.extend(samples)
                
                # Partial copy
                if remainder > 0:
                    oversampled.extend(random.sample(samples, remainder))
                
                balanced.extend(oversampled)
                oversampled_stats[label] = (count, len(oversampled))
                
            else:
                balanced.extend(samples)
        
        # Step 2: Downsample No-Event
        num_events = len(balanced)
        no_event_samples = class_windows['No-Event']
        
        # No-Event = multiplier * events
        if hasattr(self.config, 'no_event_multiplier'):
            multiplier = self.config.no_event_multiplier
        else:
            multiplier = 2  # Default
        
        target_no_event = num_events * multiplier
        num_to_keep = min(target_no_event, len(no_event_samples))
        
        kept_no_event = random.sample(no_event_samples, num_to_keep)
        balanced.extend(kept_no_event)
        
        # Shuffle
        random.shuffle(balanced)
        
        # Print statistics
        total = len(balanced)
        
        print(f"\n🎯 BALANCING RESULTS:")
        print("="*60)
        
        if oversampled_stats:
            print("📈 Oversampled classes:")
            for label, (before, after) in oversampled_stats.items():
                print(f"  {label:20s}: {before:6,d} → {after:6,d} (+{after-before:,d})")
            print()
        
        print(f"📊 Final distribution:")
        print(f"  Total events: {num_events:,d} ({num_events/total*100:.1f}%)")
        print(f"  No-Event:     {len(kept_no_event):,d} ({len(kept_no_event)/total*100:.1f}%)")
        print(f"  Total:        {total:,d}")
        
        print(f"\n📋 Per-class breakdown:")
        final_counts = defaultdict(int)
        for w in balanced:
            final_counts[w['label']] += 1
        
        for label, count in sorted(final_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            print(f"  {label:20s}: {count:7,d} ({pct:5.1f}%)")
        
        print("="*60)
        
        # Validation
        no_event_ratio = len(kept_no_event) / total
        if no_event_ratio > 0.6:
            print("⚠️ WARNING: No-Event still high (>60%). Consider:")
            print("   1. Increasing no_event_multiplier")
            print("   2. Decreasing min_rare_samples")
        elif no_event_ratio < 0.3:
            print("⚠️ WARNING: No-Event too low (<30%). Model may overfit to events.")
        else:
            print("✅ Balance looks good! No-Event ratio: 30-60%")
        
        return balanced


# ============================================================================
# USAGE IN NOTEBOOK
# ============================================================================
# Replace the existing ClassBalancer cell with this code:
"""
print("✅ ClassBalancer (Fixed v2 - Aggressive) defined")
"""

# In the balancing cell, use:
"""
print("\n[3/5] Balancing dataset...")
balancer = ClassBalancer(config)
all_labeled_windows = balancer.balance_dataset(
    all_labeled_windows,
    oversample_rare=True,      # Enable oversampling
    min_rare_samples=500       # Target 500+ for rare events
)
"""
