###############################################################################
# Environment Setup & Imports
###############################################################################
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import huggingface_hub
from datasets import load_dataset
from transformers import (
    AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# Set visible CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# Log in to Hugging Face Hub
huggingface_hub.login()

###############################################################################
# Global Definitions & Helper Functions
###############################################################################
instruction = 'Convert the equation images to LaTeX equations.'

def convert_to_conversation(sample):
    """Convert a single dataset sample to a conversation (chat) format."""
    conversation = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': instruction},
                {'type': 'image', 'image': sample['image']}
            ]
        },
        {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': sample['text']}
            ]
        },
    ]
    return {'messages': conversation}


def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    """
    Extract and convert image data from conversation messages.
    Returns a list of PIL.Image objects in RGB mode.
    """
    image_inputs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for element in content:
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                image = element["image"] if "image" in element else element
                image_inputs.append(image.convert("RGB"))
    return image_inputs

###############################################################################
# Dataset Preparation
###############################################################################
# Load the dataset (here using a subset of LaTeX_OCR training data)
dataset_train = load_dataset('unsloth/LaTeX_OCR', split='train[:3000]')

# Convert dataset samples to conversation format
converted_dataset_train = [
    convert_to_conversation(sample) for sample in tqdm(dataset_train, total=len(dataset_train))
]

# For testing: display the first sample's original image
train_image = dataset_train[0]['image']

###############################################################################
# Model & Processor Initialization
###############################################################################
model_id = "google/gemma-3-4b-pt"  # Options: "google/gemma-3-4b-pt", "google/gemma-3-12b-pt", "google/gemma-3-27-pt"
processor_id = "google/gemma-3-4b-it"  # Corresponding processor model

# Define model initialization arguments
model_kwargs = dict(
    attn_implementation="flash_attention_2", 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Configure 4-bit quantization based on nf4 (enabled by default; comment out if not needed)
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)

# Load model and processor; if quantization is not required, comment out the quantization_config settings
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    use_auth_token=True,
    **model_kwargs
)
processor = AutoProcessor.from_pretrained(
    processor_id,
    use_auth_token=True
)

###############################################################################
# LoRA & Trainer Configuration
###############################################################################
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=8,
    bias="none",
    target_modules=[
        'down_proj', 'o_proj', 'k_proj', 'q_proj',
        'gate_proj', 'up_proj', 'v_proj'
    ],
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"],
)

args = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    num_train_epochs=1,  # Modify the number of epochs as needed
    learning_rate=2e-4,
    bf16=True,
    logging_steps=200,
    save_strategy='steps',
    save_steps=200,
    save_total_limit=2,
    optim='adamw_8bit',
    weight_decay=0.01,
    lr_scheduler_type='linear',
    seed=3407,
    output_dir='outputs',
    report_to='none',
    remove_unused_columns=False,
    dataset_text_field='',
    dataset_kwargs={'skip_prepare_dataset': True},
    max_seq_length=1024,
)

def collate_fn(examples):
    """
    Data collator: Encodes text and image pairs into batches.
    Masks tokens in the labels as needed to avoid loss computation on padding and image-related tokens.
    """
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = processor.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)
        texts.append(text.strip())
        images.append(image_inputs)
    
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()

    # Mask the padding tokens and image-related tokens in the labels by setting them to -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map["boi_token"])
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch

###############################################################################
# Trainer Initialization & Training
###############################################################################
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=converted_dataset_train,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

# Start training; the model will automatically be saved to Hugging Face Hub and the output directory
trainer.train()

# Save the final model to Hugging Face Hub
trainer.save_model()

# Clean up memory
del model
del trainer
torch.cuda.empty_cache()

