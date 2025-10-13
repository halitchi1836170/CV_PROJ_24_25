from logger import log
from Globals import *
from torch.nn import Module
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
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

def save_grd_sat_overlay_image(grd_tensor, grd_cam_map, sat_tensor, sat_cam_map, path, cmap='jet', alpha=0.75):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if torch.is_tensor(grd_tensor):
        if grd_tensor.dim() == 4:
            grd_tensor = grd_tensor[0]  # da [1, C, H, W] a [C, H, W]
        if grd_tensor.dim() == 3 and grd_tensor.shape[0] in [1, 3]:  # CHW
            grd_img_np = grd_tensor.detach().cpu().numpy()  # CHW -> HWC
        else:
            grd_img_np = grd_tensor.cpu().numpy()
    else:
        grd_img_np = grd_tensor
    
    if torch.is_tensor(sat_tensor):
        if sat_tensor.dim() == 4:
            sat_tensor = sat_tensor[0]  # da [1, C, H, W] a [C, H, W]
        if sat_tensor.dim() == 3 and sat_tensor.shape[0] in [1, 3]:  # CHW
            sat_img_np = sat_tensor.detach().cpu().numpy()  # CHW -> HWC
        else:
            sat_img_np = sat_tensor.cpu().numpy()
    else:
        sat_img_np = sat_tensor

    if grd_cam_map.ndim == 3:
        grd_cam_map = grd_cam_map[0]
    grd_norm = Normalize(vmin=np.nanmin(grd_cam_map),vmax=np.nanmax(grd_cam_map))
    
    if sat_cam_map.ndim == 3:
        sat_cam_map = sat_cam_map[0]
    sat_norm = Normalize(vmin=np.nanmin(sat_cam_map),vmax=np.nanmax(sat_cam_map))
    
    grd_img_np = (grd_img_np - grd_img_np.min()) / (grd_img_np.max() - grd_img_np.min() + 1e-12)
    sat_img_np = (sat_img_np - sat_img_np.min()) / (sat_img_np.max() - sat_img_np.min() + 1e-12)
    
    fig, ax = plt.subplots(3, 2, figsize=(10, 7))
    fig.suptitle("Comparison between GRADCams calculated on Ground and Satellite NET branch's output")
    
    # Ground Original image
    ax[0,0].imshow(grd_img_np)
    ax[0,0].set_title(f"{experiments_config['name']}-GRD Original Image")
    ax[0,0].axis('on')
    
    # Ground Heatmap
    ax[1,0].imshow(grd_cam_map, cmap=cmap, norm=grd_norm, interpolation='nearest')
    ax[1,0].set_title(f"{experiments_config['name']}-GRD GradCAM Heatmap")
    ax[1,0].axis('on')
    fig.colorbar(ScalarMappable(norm=grd_norm, cmap=cmap), ax=ax[1,0], fraction=0.046, pad=0.04)

    # Ground Overlay
    ax[2,0].imshow(grd_img_np, interpolation='nearest')
    ax[2,0].imshow(grd_cam_map, cmap=cmap, norm=grd_norm, alpha=grd_norm(grd_cam_map)*alpha, interpolation='nearest')
    ax[2,0].set_title(f"{experiments_config['name']}-GRD Overlay: Image + CAM")
    ax[2,0].axis('on')
    
    # Satellite Original image
    ax[0,1].imshow(sat_img_np)
    ax[0,1].set_title(f"{experiments_config['name']}-SAT Original Image")
    ax[0,1].axis('on')
    
    # GroSatelliteund Heatmap
    ax[1,1].imshow(sat_cam_map, cmap=cmap, norm=sat_norm, interpolation='nearest')
    ax[1,1].set_title(f"{experiments_config['name']}-SAT GradCAM Heatmap")
    ax[1,1].axis('on')
    fig.colorbar(ScalarMappable(norm=sat_norm, cmap=cmap), ax=ax[1,1], fraction=0.046, pad=0.04)

    # Satellite Overlay
    ax[2,1].imshow(sat_img_np, interpolation='nearest')
    ax[2,1].imshow(sat_cam_map, cmap=cmap, norm=sat_norm, alpha=sat_norm(sat_cam_map)*alpha, interpolation='nearest')
    ax[2,1].set_title(f"{experiments_config['name']}-SAT Overlay: Image + CAM")
    ax[2,1].axis('on')
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_ovarlay_image_for_gif(image_tensor, cam_map, path, cmap='jet', alpha=0.75):
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
    norm = Normalize(vmin=np.nanmin(cam_map),vmax=np.nanmax(cam_map))
    
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-12)
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 7))
    fig.suptitle("Evolution of attention pixels during the training phase")
    
    ax.imshow(img_np, interpolation='nearest')
    ax.imshow(cam_map, cmap=cmap, norm=norm, alpha=norm(cam_map)*alpha, interpolation='nearest')
    ax.set_title(f"{experiments_config['name']}-Overlay: Image + CAM")
    ax.axis('on')
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
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
    norm = Normalize(vmin=np.nanmin(cam_map),vmax=np.nanmax(cam_map))
    
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-12)

    fig, ax = plt.subplots(3, 1, figsize=(5, 7))

    # Original image
    ax[0].imshow(img_np)
    ax[0].set_title(f"{experiments_config['name']}-Original Image")
    ax[0].axis('on')

    # Heatmap
    im = ax[1].imshow(cam_map, cmap=cmap, norm=norm, interpolation='nearest')
    ax[1].set_title(f"{experiments_config['name']}-GradCAM Heatmap")
    ax[1].axis('on')
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax[1], fraction=0.046, pad=0.04)

    # Overlay
    ax[2].imshow(img_np, interpolation='nearest')
    ax[2].imshow(cam_map, cmap=cmap, norm=norm, alpha=norm(cam_map)*alpha, interpolation='nearest')
    ax[2].set_title(f"{experiments_config['name']}-Overlay: Image + CAM")
    ax[2].axis('on')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()