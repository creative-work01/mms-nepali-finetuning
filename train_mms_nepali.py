import os
import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    DataCollatorCTCWithPadding
)
from dataclasses import dataclass
from typing import Dict, List, Union
import wandb

# Initialize wandb for experiment tracking
wandb.init(project="mms-nepali-finetuning")

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

def main():
    # Load the MMS-300M model and processor
    model_name = "facebook/mms-300m"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Load your Nepali dataset
    # Replace this with your actual dataset loading code
    dataset = load_dataset("your_nepali_dataset_path")
    
    # Prepare the dataset
    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset["train"].column_names,
        batch_size=8,
        batched=True,
        fn_kwargs={"processor": processor},
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./mms-nepali-model",
        group_by_length=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        save_steps=400,
        eval_steps=400,
        logging_steps=400,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
    )

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model("./mms-nepali-model-final")

if __name__ == "__main__":
    main() 