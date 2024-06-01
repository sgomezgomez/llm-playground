import torch
from datetime import datetime
import psutil
import os
import pandas as pd

def named_module_tensors(module, recurse=False):
    """Yield named parameters and buffers of a module."""
    for named_parameter in module.named_parameters(recurse=recurse):
        name, val = named_parameter
        if hasattr(val, "_data") or hasattr(val, "_scale"):
            if hasattr(val, "_data"):
                yield name + "._data", val._data
            if hasattr(val, "_scale"):
                yield name + "._scale", val._scale
        else:
            yield named_parameter

    for named_buffer in module.named_buffers(recurse=recurse):
        yield named_buffer

def dtype_byte_size(dtype) -> int:
    """Returns the size (in bytes) occupied by one parameter of type `dtype`."""
    import re
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8

def compare_state_dicts(model1, model2):
    """
    Compare the state dictionaries of two models to check if they are the same.
    """
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    for key in state_dict1.keys():
        if key not in state_dict2:
            print(f"Key {key} found in model1 but not in model2")
            return False
        
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]

        if tensor1 is None or tensor2 is None:
            if tensor1 != tensor2:
                print(f"Mismatch found at {key}: one of the items is None")
                return False
            continue
        
        if isinstance(tensor1, torch.dtype) and isinstance(tensor2, torch.dtype):
            if tensor1 != tensor2:
                print(f"Mismatch found at {key}: different data types ({tensor1} vs {tensor2})")
                return False
        elif isinstance(tensor1, tuple) and isinstance(tensor2, tuple):
            if len(tensor1) != len(tensor2):
                print(f"Mismatch found at {key}: different tuple lengths")
                return False
            for i in range(len(tensor1)):
                if tensor1[i] is None or tensor2[i] is None:
                    if tensor1[i] != tensor2[i]:
                        print(f"Mismatch found in tuple at {key}[{i}]: one of the items is None")
                        return False
                    continue
                if not torch.equal(tensor1[i], tensor2[i]):
                    print(f"Mismatch found in tuple at {key}[{i}]: tensor1[i] and tensor2[i] are not equal")
                    return False
        elif not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
            print(f"Mismatch found at {key}: one of the items is not a tensor (tensor1 type: {type(tensor1)}, tensor2 type: {type(tensor2)})")
            return False
        else:
            print(f"Comparing {key}: tensor1 dtype = {tensor1.dtype}, tensor2 dtype = {tensor2.dtype}")
            if tensor1.dtype != tensor2.dtype:
                print(f"Mismatch found at {key}: different data types ({tensor1.dtype} vs {tensor2.dtype})")
                return False
            if not torch.equal(tensor1, tensor2):
                print(f"Mismatch found at {key}: tensor values are different")
                return False

    for key in state_dict2.keys():
        if key not in state_dict1:
            print(f"Key {key} found in model2 but not in model1")
            return False

    return True

# Log to terminal and model log file
def log_text(model_name, text):
    model_log = './llm-playground/log/model_log.txt'
    text = str(model_name) + '_' + str(datetime.now()) + ': ' + str(text)
    open(model_log, 'a').write((text + '\n')) # Log file
    print(text) # Terminal

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # Resident Set Size (RSS) in bytes

def log_excel_rows(log_model_name: str, file_name: str, rows_df):
    if not os.path.exists(file_name):
        # If the file doesn't exist, create a new one
        with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
            rows_df.to_excel(writer, index=False, sheet_name='Data')
        log_text(log_model_name, 'Created new log file.')
    else:
        # If the file exists, try to append to it
        try:
            with pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                existing_df = pd.read_excel(file_name, sheet_name='Data')
                combined_df = pd.concat([existing_df, rows_df], ignore_index=True)
                combined_df.to_excel(writer, index=False, sheet_name='Data')
            log_text(log_model_name, 'Appended to existing log file.')
        except Exception as e:
            log_text(log_model_name, (f'Error while appending to the log file: {e}'))
            raise e