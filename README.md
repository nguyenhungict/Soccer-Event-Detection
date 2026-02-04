# Soccer Event Detection from Commentary Transcripts

## Project Overview
An automated system for detecting key soccer events (Goals, Yellow/Red cards, Penalties, Substitutions) from audio commentary transcripts using transformer-based deep learning. The model leverages temporal alignment and imbalanced learning techniques to identify high-impact moments within 90-minute matches.

## Key Features
- **Temporal Alignment Engine**: Synchronized commentary transcripts with ground-truth SoccerNet labels using a 5-second reaction lag window (T+1 to T+6 seconds), accounting for the natural delay in live sports commentary.
- **Class Balancing Strategy**:
  - Applied **Strategic Undersampling** to the "No-Event" majority class, retaining only 15% of background segments to increase the training signal for rare, critical events.
  - Improved dataset balance from 98:2 (Event:No-Event) to approximately 50:50.
- **Fuzzy String Matching**: Developed a robust data loader using Python's `difflib` to reconcile naming inconsistencies between SoccerNet and SoccerNet-Echoes datasets.
- **Transformer Architecture**: Fine-tuned **XLM-RoBERTa-base** for multilingual sequence classification, capitalizing on its pre-training to handle diverse player names and football terminology.

## Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/nguyenhungict/Soccer-Event-Detection-.git
   cd Soccer-Event-Detection-
   ```
2. **Download External Assets**:
   Due to file size limits, the `models/` and `dataset/` folders are hosted externally.
   - [Download Models](https://drive.google.com/drive/folders/1RFvh5l8u2bO-fcIRFb8b8XTi6S9PL3eK?usp=sharing) - Place in the root directory as `models/`.
   - [Download Dataset](https://drive.google.com/drive/folders/1hVSjPfSgQg_SMhiQUG_t0RR_6hlLM0BU?usp=sharing) - Place in the root directory as `dataset/`.
3. **Install Dependencies**:
   ```bash
   pip install transformers datasets scikit-learn torch SoccerNet
   ```

## Technology Stack
- **Languages**: Python
- **Frameworks**: PyTorch, HuggingFace Transformers
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Dataset**: SoccerNet-Echoes (500+ match transcripts), SoccerNet Action Spotting v2
- **Training Environment**: Google Colab (Tesla T4 GPU)

## Methodology

### 1. Data Preparation
- **Dataset**: SoccerNet-Echoes audio commentary transcripts aligned with 500+ match event annotations from SoccerNet Action Spotting v2.
- **Preprocessing Pipeline**:
  - Implemented a **3-sentence sliding window** approach to provide sufficient contextual information for the transformer.
  - Developed automated transcript-to-event alignment using **Fuzzy String Matching** (cutoff=0.3) to handle metadata discrepancies between datasets.
  - Applied temporal alignment with a 5-second reaction lag (1-6s after event) to sync commentary with ground-truth timestamps.

### 2. Training Strategy
- **Class Imbalance Handling**: 
  - Original distribution: 98% No-Event, 2% Events (extremely imbalanced).
  - Applied aggressive undersampling: retained only 15% of No-Event samples.
  - Final balanced distribution: ~50% Events, ~50% No-Event.
- **Model Architecture**: XLM-RoBERTa-base (278M parameters) with sequence classification head.
- **Hyperparameters**:
  - Batch size: 16 | Learning rate: 2e-5  
  - Epochs: 5 | Max sequence length: 160 tokens  
  - Optimizer: AdamW | Weight decay: 0.01
- **Training Duration**: ~2 hours on Tesla T4 GPU (Google Colab).

### 3. Evaluation Results
The model was evaluated on a held-out validation set (20% of total data) representing unseen matches:

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 87.0% |
| **F1 Macro (All Classes)** | **0.46** |
| **F1 Weighted** | 0.88 |
| Precision (Weighted) | 0.91 |
| Recall (Weighted) | 0.87 |

### Per-Class Performance:

| Event Type | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **No-Event** | 0.96 | 0.90 | 0.93 | 31,767 |
| **Goal** | 0.34 | 0.61 | **0.43** | 709 |
| **Yellow Card** | 0.35 | 0.54 | 0.43 | 861 |
| **Red Card** | 0.32 | 0.52 | 0.40 | 23 |
| **Substitution** | 0.28 | 0.50 | 0.36 | 1,147 |
| **Penalty** | 0.19 | 0.24 | 0.21 | 85 |

**Key Insights**:
- The model achieves **61% recall for Goal detection**, successfully capturing over half of all scoring events from commentary alone.
- High accuracy (87%) is primarily driven by reliable No-Event classification.
- The Macro F1 of 0.46 reflects the significant challenge of rare event detection in highly imbalanced sports data.

## Usage

### Inference on New Match Transcripts
To extract highlights from a match transcript:

```bash
python test_full_match.py
```

This script will:
1. Load a match transcript (JSON format from Whisper ASR).
2. Apply the same 3-sentence windowing and preprocessing.
3. Generate predictions with confidence scores for each event type.
4. Output a JSON file containing event timestamps and labels.

### Post-Processing (Threshold Tuning)
To optimize precision for production deployment:

```bash
python evaluate_result.py
```

By tuning decision thresholds per event class, the system can achieve significantly higher precision (up to 0.67 for Goals).

## Potential Model Improvements

Based on experimental iterations, the following strategies have shown promise for increasing model performance:

### 1. Advanced Loss Functions
- **Focal Loss** (γ=2.0, α=0.25): Down-weights easy examples and focuses training on hard-to-classify rare events. In experiments, this improved F1-Macro from 0.46 to 0.55+ during training.
- **Class-weighted Cross-Entropy**: Dynamically compute inverse frequency weights to penalize the model more heavily for misclassifying rare events.

### 2. Data Augmentation Strategies
- **Rare Event Oversampling**: Duplicate rare event samples (Red Card, Penalty) by 10x to increase their representation in training batches.
- **Synonym-based Text Augmentation**: Replace key action words with synonyms (e.g., "goal" → "scored", "strike") to improve model robustness to linguistic variations.
- **Back-Translation**: Use multilingual models to translate commentary to another language and back to create paraphrased training samples.

### 3. Post-Processing Optimization
- **Per-Class Threshold Tuning**: Optimize decision thresholds for each event type independently (e.g., Goal ≥ 0.7, Substitution ≥ 0.95).
  - Experimental results showed F1 improvement from 0.29 → **0.86** on test matches after threshold optimization.
- **Temporal Non-Maximum Suppression (NMS)**: Merge duplicate predictions within a 15-second window to reduce false positives.

### 4. Architecture Enhancements
- **Hierarchical Attention**: Add a secondary attention layer to weigh the importance of different sentences within the 3-sentence window.
- **Fine-tune on Domain-Specific Corpus**: Continue pre-training XLM-RoBERTa on a large corpus of sports commentary before task-specific fine-tuning.

### 5. Multimodal Fusion (Advanced)
- Integrate audio features (crowd noise peaks, referee whistles) and video frames (ball trajectory, player jerseys) to achieve 0.8+ F1 on all event types.
- This approach would require additional datasets like SoccerNet-v2 (video annotations).

## Acknowledgments
- **SoccerNet**: For providing the high-quality annotated dataset ([SoccerNet.org](https://www.soccer-net.org/)).
- **HuggingFace**: For the Transformers library and pre-trained models.
