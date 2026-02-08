# Soccer Event Detection from Commentary Transcripts (V4)

## Project Overview
An automated system for detecting high-impact soccer events (Goal, Card, Penalty) from audio commentary transcripts using transformer-based deep learning. The project has evolved from a general event classifier to a specialized, highly optimized model capable of running efficiently on local hardware while achieving state-of-the-art performance on highlight generation.

## Model Evolution: From V1 to V4

### Initial Version (V1/V2)
- **Scope**: Attempted to detect 5+ event classes including Substitutions.
- **Issues**: Extreme class imbalance (98% No-Event), causing the model to miss rare events like Red Cards and Penalties.
- **Hardware**: Heavy dependency on cloud GPUs (Tesla T4).
- **Performance**: High recall noise; F1-Score for highlights was low (< 0.40).

### Final Version (V4 - Optimized)
- **Refined Scope**: Focused strictly on **Highlights** (Goal, Card, Penalty). Removed "Substitution" to reduce noise.
- **Dataset Scale**: Trained on **4,752 matches** (SoccerNet-Echoes), split into 4,039 Train / 713 Val.
- **Advanced Balancing Strategy**:
  - Achieved a highly effective distribution for training:
    - **No-Event**: 66.7%
    - **Goal**: 14.0%
    - **Card**: 17.7%
    - **Penalty**: 1.6% (Heavily weighted w/ Class Weight 3.92)
- **Local Optimization**:
  - Runs on **4GB VRAM** (e.g., RTX 3050 Ti) using Gradient Accumulation (Steps=16).
  - Mixed Precision (FP16) training.

## Training Results
The V4 model was trained for 5 epochs with consistent performance gains:
- **Final Accuracy**: 72.4%
- **F1 Score (Highlights)**: **0.67** (Peak performance for detecting Goal/Card/Penalty)
- **Validation Loss**: Stabilized at 0.81, showing no signs of overfitting.

**Test Case: Chelsea vs Burnley (Whisper V2 English)**
| Metric | Value |
| :--- | :--- |
| **Precision** | **1.00** |
| **Recall** | **1.00** |
| **F1 Score** | **1.00** |
*Model detected 100% of events with zero false positives.*

**Test Case: Leicester vs Arsenal (High Scoring Match)**
| Metric | Value |
| :--- | :--- |
| **Precision** | **0.60** |
| **Recall** | **1.00** |
| **F1 Score** | **0.75** |
*Model successfully captured all goals in a chaotic 7-goal match.*

## Repository Structure
- `train_local_v4.py`: The main training script optimized for local execution.
- `test_full_match.py`: Inference script for generating highlights from JSON transcripts.
- `evaluate_result.py`: Evaluation tool for calculating Precision/Recall against ground truth.
- `models_v4/`: Directory for trained model artifacts.

## Installation & Usage

### 1. Environment Setup
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers scikit-learn accelerate
```

### 2. Training (Local V4)
To train the model on your local machine:
```bash
python train_local_v4.py
```

### 3. Running Inference (Test)
To generate highlights for a specific match:
1. Open `test_full_match.py` and update `TRANSCRIPT_FILE` to your target JSON.
2. Run:
```bash
python test_full_match.py
```
*Output will be saved to `highlights_v4.json`.*

### 4. Evaluation
To check accuracy against ground truth:
```bash
python evaluate_result.py
```

## Acknowledgments
- **SoccerNet**: For the annotated dataset.
- **HuggingFace**: For the Transformer models.
- **OpenAI Whisper**: For high-quality transcriptions using the V2 English model.
