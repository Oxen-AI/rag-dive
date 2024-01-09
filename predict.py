
# from peft import PeftModel
import sys
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments

base_model = sys.argv[1]
fine_tuned_model = sys.argv[2]

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model)

print(f"Loading model {fine_tuned_model}")

device_map = {"": 0}
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

ft_model = AutoModelForCausalLM.from_pretrained(
    fine_tuned_model,
    quantization_config=bnb_config,
    device_map=device_map,
)

context = "The meaning of life is to be happy."
question = "What is the meaning of life?"
prompt = f"""<text>
    <context>
        {context}
    </context>
    <question>
        {question}
    </question>
    <answer>

"""

print("Tokenzing...")
model_input = tokenizer(prompt, return_tensors="pt")
print(model_input)

print("Predict...")
ft_model.eval()
with torch.no_grad():
    print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, pad_token_id=2)[0], skip_special_tokens=True))
    