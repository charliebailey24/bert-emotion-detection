# Emotion Detection in Text

All written work can be found in the `notebooks` directory. For a comprehensive project report, start with the `Final_Report.ipynb`.

A presentation of the project can be [found here](https://youtu.be/-QX3NMWf7rY?feature=shared). The presentation slides can be found in the `pdfs` directory.

Note: you will need to re-train the model. The trained model checkpoints were not checked in to version control. This takes ~20 minutes utilizing a Mac M3 GPU.

## Installation

Install project dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

The main script `BertEmotionDetection.py` in `src/` provides training and evaluation of the emotion detection model.

1. **Train & evaluate** (saves to `../trained_models`):

   ```bash
   python src/BertEmotionDetection.py
   ```

2. **Skip training & only evaluate** (loads latest checkpoint):

   ```
   python src/BertEmotionDetection.py --eval_only
   ```

3. **Evaluate a specific checkpoint**:

   ```
   python src/BertEmotionDetection.py --eval_only --checkpoint checkpoint-XXXX
   ```

## Requirements

All Python package requirements are listed in `requirements.txt`. Install them via:

```bash
pip install -r requirements.txt
```
