import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig

# Select model id (options: "google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it")
model_id = "google/gemma-3-27b-it"
local_path = f"/mnt/moonfs/public-models-m2/{model_id}"

# Set the datatype for model weights
datatype = torch.bfloat16

# Define quantization configuration (4-bit quantization based on nf4 type)
# Uncomment the following lines to enable quantization
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=datatype
# )

# Load the Gemma3 model
# To enable quantization, uncomment "quantization_config=quantization_config"
model = Gemma3ForConditionalGeneration.from_pretrained(
    local_path,
    torch_dtype=datatype,
    device_map="auto",
    # quantization_config=quantization_config  # Uncomment to enable quantization
).eval()

# Load the processor associated with the model
processor = AutoProcessor.from_pretrained(local_path)

# Define the image path and the instruction for converting equation images to LaTeX
img_path = "./image/img.jpeg"
instruction = """
Convert the equation images to LaTeX equations.
"""

# Create chat messages including system and user roles
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": instruction}
        ]
    }
]

# Process the messages into model inputs; tokenization and formatting are handled by the processor
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

# Determine the length of the prompt tokens
input_len = inputs["input_ids"].shape[-1]

# Use the model to generate a response without computing gradients
with torch.inference_mode():
    generation = model.generate(
        **inputs,
        max_new_tokens=4096,
        do_sample=True,                # 启用采样，temperature/top_p/top_k 才会生效
        temperature=0.3,               # 控制生成多样性，常用范围 0.1~2.0
        top_p=0.95,                    # nucleus sampling
        top_k=50                       # top-k sampling
    )
    # Remove the prompt part from the generated output
    generation = generation[0][input_len:]

# Decode the generated tokens to a string, skipping special tokens
decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)