import math
import os
import random
import time
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_model_memory_bytes(total_params: int, train_dtype_bytes: int = 2, optimizer_multiplier: float = 2.0):
    weights = total_params * train_dtype_bytes
    grads = total_params * train_dtype_bytes
    optimizer = int(total_params * 4 * optimizer_multiplier)
    return weights + grads + optimizer


def bytes_to_readable(num_bytes: int):
    units = ["B", "KB", "MB", "GB", "TB"]
    val = float(num_bytes)
    idx = 0
    while val >= 1024.0 and idx < len(units) - 1:
        val /= 1024.0
        idx += 1
    return f"{val:.2f} {units[idx]}"


def gpu_memory_report():
    if not torch.cuda.is_available():
        return "CUDA not available"
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_alloc = torch.cuda.max_memory_allocated()
    return (
        f"allocated={bytes_to_readable(allocated)}, "
        f"reserved={bytes_to_readable(reserved)}, "
        f"max_allocated={bytes_to_readable(max_alloc)}"
    )


def get_lr_cosine(step, total_steps, warmup_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")
