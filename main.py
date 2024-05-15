###########################################
## Dependencies
###########################################

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from helper import compute_module_sizes

###########################################
## Config
###########################################
# Hugging Face token
os.environ['HF_TOKEN'] = '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'
# Model name
model_name = 'meta-llama/Llama-2-7b-chat-hf'
# Model file
model_path = './models/Llama-2-7b-chat-hf/'

###########################################
## Load model
###########################################
## Tokenizer
## From the Hugging Face Hub
###########################################
#tokenizer = AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True)
#tokenizer.save_pretrained(model_path)
## Stored locally
###########################################
tokenizer = AutoTokenizer.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=True)

###########################################
## Load model
###########################################
## Model
## From the Hugging Face Hub
###########################################
#model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, force_download=True)
#model.save_pretrained(model_path)
## Stored locally
###########################################
model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=True)
print(f"The model size is {compute_module_sizes(model)[''] * 1e-9} GB")
print(model)

###########################################
## Quantize Model
###########################################

model_int8 = torch.ao.quantization.quantize_dynamic(
    model,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8)
del model
print(f"The model size is {compute_module_sizes(model_int8)[''] * 1e-9} GB")
print(model)