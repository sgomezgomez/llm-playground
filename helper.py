###########################################
## Dependencies
###########################################
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

###########################################
## Helper functions
###########################################

def named_module_tensors(module, recurse=False):
    for named_parameter in module.named_parameters(recurse=recurse):
      name, val = named_parameter
      flag = True
      if hasattr(val,"_data") or hasattr(val,"_scale"):
        if hasattr(val,"_data"):
          yield name + "._data", val._data
        if hasattr(val,"_scale"):
          yield name + "._scale", val._scale
      else:
        yield named_parameter

    for named_buffer in module.named_buffers(recurse=recurse):
      yield named_buffer

def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.
    """
    import re
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8

def compute_module_sizes(model):
    """
    Compute the size of each submodule of a given model.
    """
    from collections import defaultdict
    module_sizes = defaultdict(int)
    for name, tensor in named_module_tensors(model, recurse=True):
      size = tensor.numel() * dtype_byte_size(tensor.dtype)
      name_parts = name.split(".")
      for idx in range(len(name_parts) + 1):
        module_sizes[".".join(name_parts[:idx])] += size

    return module_sizes

def download_model(model_name, model_path):
    """
    Download tokenizer and model from the Hugging Face Hub and save it locally.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True, force_download=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, force_download=True)
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
    
    return tokenizer, model

def load_model(model_path):    
    """
    Load tokenizer and model locally.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, local_files_only=True)
    
    return tokenizer, model

def quantize_linear(model, orig_model_path, quantized_model_path, dtype=torch.qint8):
    """
    Quantize linear layers from model and save it locally.
    """
    qmodel = torch.ao.quantization.quantize_dynamic(
        model,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype)
    
    torch.save({
        'model_state_dict': qmodel.state_dict(),
        'config': qmodel.config,
        'tokenizer': orig_model_path}, quantized_model_path)
    
    return qmodel

def load_quantized_model(quantized_model_path, orig_model_path):
    """
    Load quantized model locally.
    """
    tokenizer = AutoTokenizer.from_pretrained(orig_model_path, low_cpu_mem_usage=True, local_files_only=True)

    checkpoint = torch.load(quantized_model_path)
    # Load configuration correctly
    config_dict = checkpoint['config']
    config = AutoModelForCausalLM.config_class.from_dict(config_dict)
        # Initialize the model with the loaded configuration
    qmodel = AutoModelForCausalLM(config)
        # Load the state dictionary
    qmodel.load_state_dict(checkpoint['model_state_dict'])
    
    return tokenizer, qmodel