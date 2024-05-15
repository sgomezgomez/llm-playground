###########################################
## Dependencies
###########################################

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from helper import compute_module_sizes

###########################################
## Load model
###########################################

# Hugging Face token
os.environ['HF_TOKEN'] = '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'
model_name = 'EleutherAI/pythia-410m'
#model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
# Pytorch's default data type 
#torch.set_default_dtype(torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True, force_download=True)
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, force_download=True)

print(model)

model_size = compute_module_sizes(model)
print(f"The model size is {model_size[''] * 1e-9} GB")

input_text = "Hello, my name is "
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
print(f"Prompt: {input_text} \nModel response: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

model_int8 = torch.ao.quantization.quantize_dynamic(
    model,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)

print(model_int8)

model_size = compute_module_sizes(model_int8)
print(f"The quantized model size is {model_size[''] * 1e-9} GB")

outputs_int8 = model_int8.generate(input_ids, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
print(f"Prompt: {input_text} \nModel (int8) response: {tokenizer.decode(outputs_int8[0], skip_special_tokens=True)}")