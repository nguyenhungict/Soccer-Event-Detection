# Soccer Event Detection from Commentary Transcripts

## Project Overview
This project implements an automated system for detecting key events in soccer matches (Goals, Red Cards, Penalties, Substitutions) relying solely on audio commentary transcripts. The solution leverages a pre-trained multilingual transformer model (XLM-RoBERTa) fine-tuned on the SoccerNet-Echoes dataset.

The core objective was to address the challenge of extreme class imbalance in sports event detection, where 'No-Event' samples vastly outnumber rare events of interest.

## Key Features
- **Temporal Alignment**: Developed a custom aligner to map 1Hz commentary window segments to ground-truth timestamps, accounting for variable reaction lags (1-6 seconds).
- **Imbalanced Learning Strategy**: 
  - Implemented **Focal Loss** to down-weight easy negatives and focus training on hard-to-classify rare events.
  - Applied aggressive **Data Augmentation** (Oversampling rare classes by 10x) and **Undersampling** of the majority class (keeping only 10% of background noise).
- **Transformer Architecture**: Fine-tuned **XLM-RoBERTa-base** for sequence classification, chosen for its robustness in handling multilingual entities (player names, locations).
## Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/soccer-event-detection.git
   cd soccer-event-detection
   ```
2. **Download External Assets**:
   Due to file size limits, the `models/` and `dataset/` folders are hosted externally.
   - [Download Models](https://drive.google.com/drive/folders/1RFvh5l8u2bO-fcIRFb8b8XTi6S9PL3eK?usp=sharing) - Place in the root directory as `models/`.
   - [Download Dataset](https://drive.google.com/drive/folders/1hVSjPfSgQg_SMhiQUG_t0RR_6hlLM0BU?usp=sharing) - Place in the root directory as `dataset/`.
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Technology Stack
- **Languages**: Python
- **DL Frameworks**: PyTorch, HuggingFace Transformers
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Tools**: OpenAI Whisper (for ASR), Google Colab (Training environment)

## Methodology

### 1. Data Preparation
- **Dataset**: SoccerNet-Echoes (Audio transcripts aligned with event labels).
- **Preprocessing**: 
  - Sliding window approach (window size: 3 sentences) to capture context.
  - Transcript-to-event alignment using fuzzy matching of match metadata.

### 2. Model Optimization
- **Baseline**: Initial Cross-Entropy training yielded poor recall for rare events (F1 < 0.3).
- **Improvements**:
  - Integrated **Focal Loss** ($\gamma=2.0, \alpha=0.25$) to penalize misclassification of rare events.
  - Implemented class-specific decision thresholds during inference (e.g., Substitution threshold > 0.95 to reduce false positives).

### 3. Results
Evaluated on unseen English Premier League matches:

| Metric | Baseline | Optimized |
| :--- | :--- | :--- |
| **F1 Score** | 0.29 | **0.86** |
| Precision | 0.19 | 0.67 |
| Recall | 0.60 | >0.90 |

*Note: The optimized model successfully filters out background commentary noise, achieving high precision in Goal detection.*

## Usage

### Inference
To extract highlights from a match transcript:

```bash
python highlight_extraction.py --input match_transcript.json --model_path ./models/best_model
```

### Output
The system generates a JSON report containing event types, timestamps, and confidence scores, ready for video clipping integration.

## Future Work
- Integrate Multimodal Learning (Audio + Visual features) to further distinguish between 'Near Miss' and 'Goal'.
- Deploy model as a microservice using FastAPI and ONNX Runtime for real-time processing.
