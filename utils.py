from logger import log
from config import *
from torch.nn import Module

def print_params(configuration):
    for (key, value) in configuration.items():
        log.info("%s = %s", key, value)

def get_header_title(string, new_line=False):
    str_result=""
    if new_line :
        str_result = int((header_length - len(string)) / 2) * "-" + string + int((header_length - len(string)) / 2) * "-" + "\n"
    else:
        str_result = int((header_length - len(string)) / 2) * "-" + string + int((header_length - len(string)) / 2) * "-"
    return str_result

def calculate_model_size(model: Module) -> int:
    total_size = 0

    for param in model.parameters():
        param_size = param.numel() * 4
        total_size += param_size

    for buffer in model.buffers():
        buffer_size = buffer.numel() * 4
        total_size += buffer_size

    return total_size


def get_model_summary_simple(model: Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = calculate_model_size(model) / (1024 * 1024)

    log.info(f"Model: {model.__class__.__name__}")
    log.info(f"Total parameters: {total_params:,}")
    log.info(f"Trainable parameters: {trainable_params:,}")
    log.info(f"Model size: {model_size_mb:.2f} MB")
