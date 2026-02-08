# WeightedTrainer.py
# ✅ Custom Trainer với class weights để penalize No-Event bias
# Thêm cell MỚI vào notebook, TRƯỚC cell training (Section 8)

import torch
import torch.nn as nn
import numpy as np
from transformers import Trainer
from sklearn.utils.class_weight import compute_class_weight


# ============================================================================
# STEP 1: Compute Class Weights
# ============================================================================

def compute_balanced_weights(train_data, config):
    """
    Compute balanced class weights
    
    Returns:
        torch.FloatTensor: Class weights for CrossEntropyLoss
    """
    print("\n💪 Computing class weights...")
    
    # Get all label IDs from training data
    train_labels = [config.label2id[w['label']] for w in train_data]
    unique_labels = np.unique(train_labels)
    
    # Compute balanced weights using sklearn
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_labels,
        y=train_labels
    )
    
    # Print weights
    print("\n📊 Class Weights (Higher = More Important):")
    print("-" * 50)
    for idx, weight in enumerate(class_weights):
        label = config.id2label[idx]
        count = train_labels.count(idx)
        print(f"  {label:20s}: {weight:6.2f} (n={count:,})")
    print("-" * 50)
    
    # Explanation
    no_event_weight = class_weights[config.label2id['No-Event']]
    avg_event_weight = np.mean([w for i, w in enumerate(class_weights) 
                                if config.id2label[i] != 'No-Event'])
    
    print(f"\n📈 Weight Analysis:")
    print(f"  No-Event weight:      {no_event_weight:.2f}")
    print(f"  Avg Event weight:     {avg_event_weight:.2f}")
    print(f"  Ratio (Event/No-Evt): {avg_event_weight/no_event_weight:.2f}x")
    print(f"\n  → Model sẽ penalize No-Event errors {avg_event_weight/no_event_weight:.1f}x so với Event errors")
    
    # Convert to tensor
    weights_tensor = torch.FloatTensor(class_weights)
    
    return weights_tensor


# ============================================================================
# STEP 2: Custom Trainer with Weighted Loss
# ============================================================================

class WeightedLossTrainer(Trainer):
    """
    Custom Trainer that uses weighted CrossEntropyLoss
    
    This helps combat class imbalance by penalizing mistakes
    on minority classes (events) more than majority class (No-Event)
    """
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
        if self.class_weights is not None:
            print(f"✅ WeightedLossTrainer initialized with class weights")
        else:
            print("⚠️ No class weights provided, using standard CrossEntropyLoss")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to use weighted CrossEntropyLoss
        """
        # Extract labels
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Compute weighted loss
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )
        else:
            loss_fct = nn.CrossEntropyLoss()
        
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), 
            labels.view(-1)
        )
        
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# ALTERNATIVE: Focal Loss (Even more aggressive)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for hard example mining
    
    FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    
    - gamma: Focus on hard examples (default 2.0)
    - alpha: Class weighting (default 0.25)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class labels
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLossTrainer(Trainer):
    """Trainer using Focal Loss"""
    
    def __init__(self, alpha=0.25, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        print(f"✅ FocalLossTrainer initialized (alpha={alpha}, gamma={gamma})")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss = self.focal_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# USAGE IN NOTEBOOK
# ============================================================================
"""
# ADD THIS CELL BEFORE TRAINING (after loading model, before trainer setup)

# Compute class weights
class_weights = compute_balanced_weights(train_data, config)

# Setup trainer with weights
trainer = WeightedLossTrainer(
    class_weights=class_weights,  # ✅ Add class weights
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("✅ Trainer ready with weighted loss")
"""


# ============================================================================
# COMPARISON
# ============================================================================
"""
STANDARD TRAINER:
- All classes treated equally
- Model tends to predict No-Event (majority class)
- Low F1 for events

WEIGHTED TRAINER:
- Rare classes weighted higher
- Model penalized more for missing events
- Better F1 for minority classes

FOCAL LOSS TRAINER:
- Even more aggressive
- Focuses on hard-to-classify examples
- Best for very imbalanced data
- May be overkill if weighted loss works

RECOMMENDATION: Try Weighted first, Focal if still not good
"""
