# MMS-300M Nepali Fine-tuning

This repository contains code for fine-tuning the MMS-300M model on Nepali speech data.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
   - Place your Nepali audio files (WAV format) in a directory
   - Create a transcript file (JSON or CSV) with the following format:
     - For JSON: `{"audio_file_id": "transcript text", ...}`
     - For CSV: Two columns with headers "file_id" and "text"

3. Process your dataset:
```bash
python prepare_dataset.py
```
Edit the paths in the script to point to your audio files and transcript file.

## Training

1. Start the training:
```bash
python train_mms_nepali.py
```

The training script will:
- Load the MMS-300M model
- Fine-tune it on your Nepali dataset
- Save checkpoints and the final model
- Log training progress to Weights & Biases

## Model Output

The fine-tuned model will be saved in:
- Checkpoints: `./mms-nepali-model/`
- Final model: `./mms-nepali-model-final/`

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- At least 16GB RAM
- Sufficient disk space for the dataset and model

## Notes

- The training script uses mixed precision training (fp16) for better performance
- Gradient accumulation is used to handle larger effective batch sizes
- The model is evaluated every 400 steps
- Training progress is logged to Weights & Biases for monitoring

## Customization

You can modify the training parameters in `train_mms_nepali.py`:
- Batch size
- Learning rate
- Number of epochs
- Evaluation frequency
- etc. 