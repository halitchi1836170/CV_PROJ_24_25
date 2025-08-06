from logger import log
from Globals import *
from torch.nn import Module
from matplotlib import pyplot as plt
import os
import numpy as np
from Globals import experiments_config
import torch

def plot_iterative_loss(epoch_losses, experiment_name, save_path):
    flattened_losses = [loss for epoch in epoch_losses for loss in epoch]

    plt.figure(figsize=(12, 6))
    plt.plot(flattened_losses, label=f"{experiment_name} - Iterative Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"{experiments_config['name']} - Mini batch Losses per Iteration")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log.info(f"{experiments_config['name']} - Mini batch losses plot salvato in: {save_path}")

def plot_loss_boxplot(epoch_losses, experiment_name, save_path):
    plt.figure(figsize=(12, 6))
    plt.boxplot(epoch_losses, showmeans=True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss per Iteration")
    plt.title(f"Loss Distribution per Epoch - {experiment_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log.info(f"Boxplot salvato in: {save_path}")

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



def save_overlay_image(image_tensor, cam_map, path, cmap='jet', alpha=0.75):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if torch.is_tensor(image_tensor):
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]  # da [1, C, H, W] a [C, H, W]
        if image_tensor.dim() == 3 and image_tensor.shape[0] in [1, 3]:  # CHW
            img_np = image_tensor.detach().cpu().numpy()  # CHW -> HWC
        else:
            img_np = image_tensor.cpu().numpy()
    else:
        img_np = image_tensor
        
    if cam_map.ndim == 3:
        cam_map = cam_map[0]
    
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    fig, ax = plt.subplots(3, 1, figsize=(5, 7))

    # Original image
    ax[0].imshow(img_np)
    ax[0].set_title(f"{experiments_config['name']}-Original Image")
    ax[0].axis('on')

    # Heatmap
    im = ax[1].imshow(cam_map, cmap=cmap)
    ax[1].set_title(f"{experiments_config['name']}-GradCAM Heatmap")
    ax[1].axis('on')
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    # Overlay
    if img_np.shape[-1] == 3:
        cam_color = plt.get_cmap(cmap)(cam_map)[..., :3]
        overlay = np.clip((1 - alpha) * img_np + alpha * cam_color, 0, 1)
        ax[2].imshow(overlay)
        ax[2].set_title(f"{experiments_config['name']}-Overlay: Image + CAM")
        ax[2].axis('on')
    else:
        ax[2].imshow(cam_map, cmap=cmap)
        ax[2].set_title(f"{experiments_config['name']}-Overlay not available")
        ax[2].axis('on')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()