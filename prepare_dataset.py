import os
import json
import torchaudio
from pathlib import Path
from datasets import Dataset, DatasetDict
import pandas as pd

def create_dataset(audio_dir: str, transcript_file: str, output_dir: str):
    """
    Create a dataset from audio files and transcripts.
    
    Args:
        audio_dir: Directory containing audio files
        transcript_file: Path to the transcript file (CSV or JSON)
        output_dir: Directory to save the processed dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load transcripts
    if transcript_file.endswith('.csv'):
        transcripts = pd.read_csv(transcript_file)
    else:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcripts = json.load(f)
    
    # Prepare dataset dictionary
    dataset_dict = {
        "audio_path": [],
        "text": [],
        "duration": []
    }
    
    # Process each audio file
    for audio_file in Path(audio_dir).glob("*.wav"):
        # Get corresponding transcript
        file_id = audio_file.stem
        transcript = transcripts.get(file_id, "")
        
        # Load audio to get duration
        waveform, sample_rate = torchaudio.load(str(audio_file))
        duration = waveform.shape[1] / sample_rate
        
        dataset_dict["audio_path"].append(str(audio_file))
        dataset_dict["text"].append(transcript)
        dataset_dict["duration"].append(duration)
    
    # Create dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split into train/validation sets
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Save the dataset
    dataset.save_to_disk(output_dir)
    
    return dataset

def main():
    # Example usage
    audio_dir = "path/to/your/audio/files"
    transcript_file = "path/to/your/transcripts.json"
    output_dir = "nepali_dataset"
    
    dataset = create_dataset(audio_dir, transcript_file, output_dir)
    print(f"Dataset created with {len(dataset['train'])} training samples and {len(dataset['test'])} validation samples")

if __name__ == "__main__":
    main() 