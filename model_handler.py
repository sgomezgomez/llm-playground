import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import defaultdict
import re
from dotenv import load_dotenv
from utils import named_module_tensors, dtype_byte_size, log_text

class ModelHandler:

    def __init__(self, hf_model_name: str, log_model_name: str, model_path='', quantized_model_path=''):
        self.hf_model_name = hf_model_name
        self.log_model_name = log_model_name
        self.model_path = model_path
        self.quantized_model_path = quantized_model_path
        self.phi3_series = ['microsoft/Phi-3-mini-4k-instruct']
        self.use_quantized = False
        if quantized_model_path != '': self.use_quantized = True
        self.model = None
        self.tokenizer = None
        # Load environment variables from .env file
        env_path = './llm-playground/.env'
        load_dotenv(dotenv_path=env_path)
        self.token = os.getenv('HF_TOKEN')
        if not self.token:
            raise ValueError("Hugging Face token not found. Please set it in the .env file.")
    
    def compute_module_sizes(self) -> dict:
        """Compute the size of each submodule of a given model."""
        if not self.model:
            raise ValueError("Model is not loaded. Call download_model() first.")
        
        module_sizes = defaultdict(int)
        for name, tensor in named_module_tensors(self.model, recurse=True):
            size = tensor.numel() * dtype_byte_size(tensor.dtype)
            name_parts = name.split(".")
            for idx in range(len(name_parts) + 1):
                module_sizes[".".join(name_parts[:idx])] += size

        return module_sizes

    def download_model(self) -> None:
        """Download tokenizer and model from the Hugging Face Hub and save it locally."""
        
        if self.hf_model_name in self.phi3_series:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_name, 
                low_cpu_mem_usage=True, 
                force_download=True,
                trust_remote_code=True,
            )
            log_text(self.log_model_name, 'Tokenizer download complete')
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name, 
                low_cpu_mem_usage=True, 
                force_download=True,
                trust_remote_code=True,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_name, 
                low_cpu_mem_usage=True, 
                force_download=True,
            )
            log_text(self.log_model_name, 'Tokenizer download complete')
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name, 
                low_cpu_mem_usage=True, 
                force_download=True, 
            )
        log_text(self.log_model_name, 'Model download complete')
        self.save_model()
        log_text(self.log_model_name, 'Model saved locally')
        if self.use_quantized:
            self._quantize_linear_modules()
            log_text(self.log_model_name,'Model quantization complete')
    
    def load_model(self) -> None:
        """Load the tokenizer and model from the local path."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        log_text(self.log_model_name, 'Tokenizer local load complete')
        if self.use_quantized:
            self._load_quantized_model()
            log_text(self.log_model_name, 'Quantized model local load complete')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            log_text(self.log_model_name, 'Model local load complete')

    def save_model(self) -> None:
        """Save the tokenizer and model to the local path."""
        self.tokenizer.save_pretrained(self.model_path)
        self.model.save_pretrained(self.model_path)
    
    def _quantize_linear_modules(self, dtype=torch.qint8):
        """Quantize linear layers from model and save it locally."""
        if not self.model:
            raise ValueError("Model is not loaded. Call download_model() or load_model() first.")
        # Quantize linear layers from model
        self.model = torch.ao.quantization.quantize_dynamic(
            self.model,  # the original model
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype)
        # Save quantized model locally
        self._save_quantized_model()

    def _save_quantized_model(self):
        """Save quantized model locally."""
        if not self.model:
            raise ValueError("Quantized model is not available. Call quantize_linear() first.")
        # Save model config
        self.model.config.save_pretrained(self.quantized_model_path)
        # Save the quantized model's state_dict and model configuration
        torch.save(self.model.state_dict(), self.quantized_model_path + 'qmodel.pt')
        
    def _load_quantized_model(self, dtype=torch.qint8):
        """Load quantized model from file."""
        # Load the model configuration
        qconfig = AutoConfig.from_pretrained(self.quantized_model_path)
        # Initialize dummy model with configuration
        qmodel = AutoModelForCausalLM.from_config(qconfig)
        # Reconstruct quantized model
        self.model = torch.ao.quantization.quantize_dynamic(
            qmodel,  # the original model
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype)
        del qmodel
        
        # Load state dictionary
        qstate_dict = torch.load(self.quantized_model_path + 'qmodel.pt')
        self.model.load_state_dict(qstate_dict)
    
    def run_inference(self, input_text: str):
        prompt_template = self._load_prompt_template(input_text)
        input_ids = self.tokenizer(prompt_template, return_tensors="pt").input_ids
        #outputs = model.generate(input_ids, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        max_len = len(input_text)
        outputs = self.model.generate(input_ids, max_length=max_len, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def _load_prompt_template(self, input_text):
        new_input_text = input_text
        if self.hf_model_name in self.phi3_series:
            new_input_text = (f'<|user|>\n{input_text} <|end|>\n<|assistant|>')
        return new_input__text
    