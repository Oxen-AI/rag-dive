# merge_lora_model.py
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import sys
import torch

if len(sys.argv) != 4:
    print("Usage: python merge_lora_model.py <base_model_name> <lora_dir> <output_dir>")
    exit()

device_map = {"": 0}
base_model_name = sys.argv[1]
lora_dir = sys.argv[2]

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

print("Loading model...")
model = AutoPeftModelForCausalLM.from_pretrained(lora_dir, device_map=device_map, torch_dtype=torch.bfloat16)


print("Merging model...")
model = model.merge_and_unload()

output_dir = sys.argv[3]
model.save_pretrained(output_dir)