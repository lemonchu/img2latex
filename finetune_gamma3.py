import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7" 
import huggingface_hub
from datasets import load_dataset
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
from peft import LoraConfig
from trl import SFTConfig
from trl import SFTTrainer

huggingface_hub.login()
instruction = 'Convert the equation images to LaTeX equations.'
def convert_to_conversation(sample):
    conversation = [
        { 'role': 'user',
          'content' : [
            {'type' : 'text',  'text'  : instruction},
            {'type' : 'image', 'image' : sample['image']} ]
        },
        { 'role' : 'assistant',
          'content' : [
            {'type' : 'text',  'text'  : sample['text']} ]
        },
    ]
    return { 'messages' : conversation }


def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs

# Load dataset from the hub
#dataset = load_dataset("unsloth/LaTeX_OCR", split="train")
dataset_train = load_dataset('unsloth/LaTeX_OCR', split='train[:3000]')

train_image = dataset_train[0]['image']

converted_dataset_train = [
    convert_to_conversation(sample) \
    for sample in tqdm(dataset_train, total=len(dataset_train))
]

model_id = "google/gemma-3-4b-pt" # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`

# Define model init arguments
model_kwargs = dict(
    attn_implementation="flash_attention_2", 
    torch_dtype=torch.bfloat16,
    device_map="auto", # Let torch decide how to load the model
)

# BitsAndBytesConfig int-4 config
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)

# Load model and tokenizer
model = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-3-4b-pt",
    use_auth_token=True,
    **model_kwargs
)

processor = AutoProcessor.from_pretrained(
    "google/gemma-3-4b-it",
    use_auth_token=True
)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0,
    r=8,
    bias="none",
    target_modules=[
 'down_proj',
 'o_proj',
 'k_proj',
 'q_proj',
 'gate_proj',
 'up_proj',
 'v_proj'],
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=1, # For full training runs over the dataset.
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

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image_inputs = process_vision_info(example["messages"])
        text = processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        )
        texts.append(text.strip())
        images.append(image_inputs)

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
    labels = batch["input_ids"].clone()

    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=converted_dataset_train,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()

del model
del trainer
torch.cuda.empty_cache()

