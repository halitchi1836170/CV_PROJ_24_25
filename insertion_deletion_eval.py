import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict
from Globals import BLOCKING_COUNTER, GROUND_MEAN, GROUND_STD
from logger import log
from tqdm import tqdm
from Train import validate_original
from Network import HookManager, compute_gradcam_from_acts_grads, gradcam_from_activations, gradcam_config
from Globals import config, images_params
import matplotlib.pyplot as plt

def compute_gradcam(model, grd_img, polar_sat_img, segmap_img, device):
    """
    Compute GradCAM for ground images using existing gradcam_from_activations function.
    
    Returns:
        cam: torch.Tensor of shape [B, H, W] normalized GradCAM heatmaps
    """
    model.train()  # Need gradients
    
    # Forward pass with activations capture
    sat_matrix, grd_matrix, distance, pred_orien, grd_acts, sat_acts = model(
        grd_img, polar_sat_img, segmap_img, return_target_acts=True
    )
    
    # Use diagonal of distance as target (positive pairs - minimize distance)
    # Since we want to highlight important regions, we use the positive pair distance
    target_scalar = distance.diag().sum()
    
    # Use the existing gradcam_from_activations function
    # It computes: gradients -> weights -> weighted sum -> ReLU -> normalize
    cam = gradcam_from_activations(
        acts=grd_acts,
        target_scalar=target_scalar,
        upsample_to_hw=None,  # We'll resize later to match input size
        normalize=True
    )
    
    # cam is now [B, H, W] and already normalized
    model.eval()
    return cam

def apply_mask_to_image(images: torch.Tensor, mask: torch.Tensor, background_value=0.0) -> torch.Tensor:
    mask_expanded = mask.unsqueeze(-1)  # [B, H, W, 1]
    # Where mask=1: keep original pixel, where mask=0: use background_value
    return images * mask_expanded + background_value * (1 - mask_expanded)

def denormalize_image(img_normalized):
    img = img_normalized * GROUND_STD + GROUND_MEAN
    img = np.clip(img, 0, 1)
    return img

def save_example_image(grd_original, grd_masked, cam_resized, mask, pct, step, save_dir, test_type):
    # Take first image from batch
    img_orig = grd_original[0].cpu().numpy()  # [H, W, C]
    img_masked = grd_masked[0].cpu().numpy()  # [H, W, C]
    cam_viz = cam_resized[0].cpu().numpy()  # [H, W]
    mask_viz = mask[0].cpu().numpy()  # [H, W]
    
    img_orig = denormalize_image(img_orig)
    img_masked = denormalize_image(img_masked)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(img_orig)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # GradCAM heatmap
    im1 = axes[1].imshow(cam_viz, cmap='jet')
    axes[1].set_title('GradCAM')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Mask
    axes[2].imshow(mask_viz, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Mask ({pct*100:.1f}%)')
    axes[2].axis('off')
    
    # Masked result
    axes[3].imshow(img_masked)
    axes[3].set_title(f'{test_type.capitalize()}ed Image')
    axes[3].axis('off')
    
    plt.suptitle(f'{test_type.capitalize()} Test - Step {step+1} - {pct*100:.1f}% pixels', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    filename = f'{test_type}_step_{step:03d}_pct_{int(pct*100):03d}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info(f"  Saved example image: {filepath}")

def deletion_test(
    model,
    data,
    device,
    batch_size,
    grd_FOV,
    grd_noise,
    num_steps=100,
    save_path=None
):
    """
    Deletion test: progressively remove most important pixels according to GradCAM.
    
    Args:
        model: trained model
        data: InputData object
        device: torch device
        batch_size: batch size for evaluation
        grd_FOV: field of view for ground images
        grd_noise: noise level
        num_steps: number of deletion steps (default 100 for 1% increments)
        save_path: path to save results
    
    Returns:
        results: dict with 'accuracies', 'auc', 'percentages'
    """
    log.info("Starting DELETION test...")
    
    images_dir = None
    if save_path:
        images_dir = os.path.join(os.path.dirname(save_path), 'deletion_examples')
        os.makedirs(images_dir, exist_ok=True)
        log.info(f"Example images will be saved to: {images_dir}")
    
    percentages = np.linspace(0, 1, num_steps + 1)  # 0%, 1%, 2%, ..., 100%
    accuracies = []
    
    # Setup per le matrici globali
    n_test_samples = data.get_test_dataset_size()
    train_grd_FOV = config["train_grd_FOV"]
    max_angle = images_params["max_angle"]
    max_width = images_params["max_width"]
    width = int(train_grd_FOV / max_angle * max_width)
    
    # Dummy forward per ottenere le dimensioni
    grd_x = torch.zeros([2, int(max_width/4), width, 3]).to(device)
    polar_sat_x = torch.zeros([2, int(max_width/4), max_width, 3]).to(device)
    segmap_x = torch.zeros([2, int(max_width/4), max_width, 3]).to(device)
    
    with torch.no_grad():
        sat_matrix, grd_matrix, _, _ = model(grd_x, polar_sat_x, segmap_x)
    
    s_height, s_width, s_channel = list(sat_matrix.size())[1:]
    g_height, g_width, g_channel = list(grd_matrix.size())[1:]
    
    for step, pct in enumerate(percentages):
        log.info(f"Deletion step {step+1}/{len(percentages)}: removing {pct*100:.1f}% of pixels")
        
        data.__cur_id = 0
        val_i = 0
        counter = 0
        
        sat_global_matrix = np.zeros([n_test_samples, s_height, s_width, s_channel])
        grd_global_matrix = np.zeros([n_test_samples, g_height, g_width, g_channel])
        
        # Flag to save only first batch image
        saved_example_for_this_step = False
        
        while True:
            
            if(counter+1==BLOCKING_COUNTER):
                log.info("Breaking...")
                break
            
            batch_sat_polar, batch_sat, batch_grd, batch_segmap, batch_orien = data.next_batch_scan(
                batch_size, grd_noise=grd_noise, FOV=grd_FOV
            )
            
            if batch_sat is None:
                break
            
            grd = torch.from_numpy(batch_grd).float().to(device)
            sat_polar = torch.from_numpy(batch_sat_polar).float().to(device)
            seg = torch.from_numpy(batch_segmap).float().to(device)
            
            # Compute GradCAM
            #cam = compute_gradcam(model, grd.clone().requires_grad_(True), sat_polar, seg, device)
            
            hm_grd = HookManager(model, f"ground_branch.{gradcam_config['visualization_layer']}")

            torch.set_grad_enabled(True)
            model.eval()
            
            grd_for_cam = grd.clone().requires_grad_(True)
            
            sat_mat_cam, grd_mat_cam, distance_cam, _ = model(grd_for_cam, sat_polar, seg)
            
            target_scalar = distance_cam.diag().sum()
            model.zero_grad()
            target_scalar.backward()
            
            cam = compute_gradcam_from_acts_grads(hm_grd.activations, hm_grd.gradients, upsample_to=(grd.shape[1], grd.shape[2]))
        
            hm_grd.remove()
            torch.set_grad_enabled(False)
            
            # Resize CAM to match input image size
            cam_resized = F.interpolate(
                cam.unsqueeze(1),
                size=(grd.shape[1], grd.shape[2]),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [B, H, W]
            
            # Create deletion mask
            B, H, W = cam_resized.shape
            num_pixels = H * W
            num_to_remove = int(num_pixels * pct)
            
            mask = torch.ones_like(cam_resized)  # Start with all pixels
            
            if num_to_remove > 0:
                for b in range(B):
                    cam_flat = cam_resized[b].view(-1)
                    # Get indices of top pixels to remove
                    _, indices = torch.topk(cam_flat, num_to_remove, largest=True)
                    mask_flat = mask[b].view(-1)
                    mask_flat[indices] = 0
                    mask[b] = mask_flat.view(H, W)
            
            # Apply mask to ground images
            grd_masked = apply_mask_to_image(grd, mask)
            
            if not saved_example_for_this_step and images_dir:
                save_example_image(grd, grd_masked, cam_resized, mask, pct, step, 
                                    images_dir, 'deletion')
                saved_example_for_this_step = True
                    
            with torch.no_grad():
                # Forward pass with masked images
                sat_mat, grd_mat, distance, _ = model(grd_masked, sat_polar, seg)
            
            sat_global_matrix[val_i:val_i+sat_mat.shape[0], :] = sat_mat.cpu().detach().numpy()
            grd_global_matrix[val_i:val_i+grd_mat.shape[0], :] = grd_mat.cpu().detach().numpy()
            val_i += sat_mat.shape[0]
            counter += 1
        
        # Compute accuracy for this deletion percentage
        sat_descriptor = np.reshape(sat_global_matrix[:, :, :g_width, :], [-1, g_height*g_width*g_channel])
        norm = np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)
        sat_descriptor = sat_descriptor / np.maximum(norm, 1e-12)
        grd_descriptor = np.reshape(grd_global_matrix, [-1, g_height*g_width*g_channel])
        
        dist_array = 2.0 - 2.0 * np.matmul(grd_descriptor, np.transpose(sat_descriptor))
        
        # Calculate top-1 accuracy
        accuracy = validate_original(dist_array,1)
        accuracies.append(accuracy)
        
        log.info(f"  Accuracy at {pct*100:.1f}% deletion: {accuracy*100:.2f}%")
        
        if save_path:
            intermediate_results = {
                'percentages': percentages[:step+1],
                'accuracies': np.array(accuracies),
                'auc': np.trapz(accuracies, percentages[:step+1]) if len(accuracies) > 1 else 0.0
            }
            np.save(save_path, intermediate_results)
            log.info(f"  Intermediate results saved to {save_path}")
    
    # Compute AUC (Area Under Curve)
    auc = np.trapz(accuracies, percentages)
    log.info(f"Deletion AUC: {auc:.4f}")
    
    results = {
        'percentages': percentages,
        'accuracies': np.array(accuracies),
        'auc': auc
    }
    
    if save_path:
        np.save(save_path, results)
        log.info(f"Deletion results saved to {save_path}")
    
    return results


def insertion_test(
    model,
    data,
    device,
    batch_size,
    grd_FOV,
    grd_noise,
    num_steps=100,
    save_path=None
):
    """
    Insertion test: progressively add most important pixels to blank image according to GradCAM.
    
    Args:
        model: trained model
        data: InputData object
        device: torch device
        batch_size: batch size for evaluation
        grd_FOV: field of view for ground images
        grd_noise: noise level
        num_steps: number of insertion steps (default 100 for 1% increments)
        save_path: path to save results
    
    Returns:
        results: dict with 'accuracies', 'auc', 'percentages'
    """
    log.info("Starting INSERTION test...")
    model.eval()
    
    images_dir = None
    if save_path:
        images_dir = os.path.join(os.path.dirname(save_path), 'insertion_examples')
        os.makedirs(images_dir, exist_ok=True)
        log.info(f"Example images will be saved to: {images_dir}")
    
    percentages = np.linspace(0, 1, num_steps + 1)  # 0%, 1%, 2%, ..., 100%
    accuracies = []
    
    # Setup per le matrici globali
    n_test_samples = data.get_test_dataset_size()
    train_grd_FOV = config["train_grd_FOV"]
    max_angle = images_params["max_angle"]
    max_width = images_params["max_width"]
    width = int(train_grd_FOV / max_angle * max_width)
    
    # Dummy forward per ottenere le dimensioni
    grd_x = torch.zeros([2, int(max_width/4), width, 3]).to(device)
    polar_sat_x = torch.zeros([2, int(max_width/4), max_width, 3]).to(device)
    segmap_x = torch.zeros([2, int(max_width/4), max_width, 3]).to(device)
    
    with torch.no_grad():
        sat_matrix, grd_matrix, _, _ = model(grd_x, polar_sat_x, segmap_x)
    
    s_height, s_width, s_channel = list(sat_matrix.size())[1:]
    g_height, g_width, g_channel = list(grd_matrix.size())[1:]
    
    for step, pct in enumerate(percentages):
        log.info(f"Insertion step {step+1}/{len(percentages)}: adding {pct*100:.1f}% of pixels")
        
        data.__cur_id = 0
        val_i = 0
        counter = 0
        
        sat_global_matrix = np.zeros([n_test_samples, s_height, s_width, s_channel])
        grd_global_matrix = np.zeros([n_test_samples, g_height, g_width, g_channel])
        
        saved_example_for_this_step = False
        
        while True:
            
            if(counter+1==BLOCKING_COUNTER):
                log.info("Breaking...")
                break
            
            batch_sat_polar, batch_sat, batch_grd, batch_segmap, batch_orien = data.next_batch_scan(
                batch_size, grd_noise=grd_noise, FOV=grd_FOV
            )
            
            if batch_sat is None:
                break
            
            grd = torch.from_numpy(batch_grd).float().to(device)
            sat_polar = torch.from_numpy(batch_sat_polar).float().to(device)
            seg = torch.from_numpy(batch_segmap).float().to(device)
            
            # Compute GradCAM
            #cam = compute_gradcam(model, grd.clone().requires_grad_(True), sat_polar, seg, device)
            
            hm_grd = HookManager(model, f"ground_branch.{gradcam_config['visualization_layer']}")

            torch.set_grad_enabled(True)
            model.eval()
            
            grd_for_cam = grd.clone().requires_grad_(True)
            
            sat_mat_cam, grd_mat_cam, distance_cam, _ = model(grd_for_cam, sat_polar, seg)
            
            target_scalar = distance_cam.diag().sum()
            model.zero_grad()
            target_scalar.backward()
            
            cam = compute_gradcam_from_acts_grads(hm_grd.activations, hm_grd.gradients, upsample_to=(grd.shape[1], grd.shape[2]))
        
            hm_grd.remove()
            torch.set_grad_enabled(False)
            
            # Resize CAM to match input image size
            cam_resized = F.interpolate(
                cam.unsqueeze(1),
                size=(grd.shape[1], grd.shape[2]),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # [B, H, W]
            
            # Create insertion mask (start from blank)
            B, H, W = cam_resized.shape
            num_pixels = H * W
            num_to_add = int(num_pixels * pct)
            
            mask = torch.zeros_like(cam_resized)  # Start with no pixels
            
            if num_to_add > 0:
                for b in range(B):
                    cam_flat = cam_resized[b].view(-1)
                    # Get indices of top pixels to add
                    _, indices = torch.topk(cam_flat, num_to_add, largest=True)
                    mask_flat = mask[b].view(-1)
                    mask_flat[indices] = 1
                    mask[b] = mask_flat.view(H, W)
            
            # Apply mask to ground images (start from blank/white)
            # For blank image, we can use zeros or mean pixel value
            grd_masked = apply_mask_to_image(grd, mask, background_value=1.0)
            
            if not saved_example_for_this_step and images_dir:
                save_example_image(grd, grd_masked, cam_resized, mask, pct, step, 
                                    images_dir, 'insertion')
                saved_example_for_this_step = True
            
            with torch.no_grad():
                # Forward pass with masked images
                sat_mat, grd_mat, distance, _ = model(grd_masked, sat_polar, seg)
            
            sat_global_matrix[val_i:val_i+sat_mat.shape[0], :] = sat_mat.cpu().detach().numpy()
            grd_global_matrix[val_i:val_i+grd_mat.shape[0], :] = grd_mat.cpu().detach().numpy()
            val_i += sat_mat.shape[0]
            counter += 1
        
        # Compute accuracy for this insertion percentage
        sat_descriptor = np.reshape(sat_global_matrix[:, :, :g_width, :], [-1, g_height*g_width*g_channel])
        norm = np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)
        sat_descriptor = sat_descriptor / np.maximum(norm, 1e-12)
        grd_descriptor = np.reshape(grd_global_matrix, [-1, g_height*g_width*g_channel])
        
        dist_array = 2.0 - 2.0 * np.matmul(grd_descriptor, np.transpose(sat_descriptor))
        
        # Calculate top-1 accuracy
        accuracy = validate_original(dist_array,1)
        accuracies.append(accuracy)
        
        log.info(f"  Accuracy at {pct*100:.1f}% insertion: {accuracy*100:.2f}%")
        
        if save_path:
            intermediate_results = {
                'percentages': percentages[:step+1],
                'accuracies': np.array(accuracies),
                'auc': np.trapz(accuracies, percentages[:step+1]) if len(accuracies) > 1 else 0.0
            }
            np.save(save_path, intermediate_results)
            log.info(f"  Intermediate results saved to {save_path}")
    
    # Compute AUC (Area Under Curve)
    auc = np.trapz(accuracies, percentages)
    log.info(f"Insertion AUC: {auc:.4f}")
    
    results = {
        'percentages': percentages,
        'accuracies': np.array(accuracies),
        'auc': auc
    }
    
    if save_path:
        np.save(save_path, results)
        log.info(f"Insertion results saved to {save_path}")
    
    return results


def plot_insertion_deletion_curves(results_dict: Dict, save_path: str):
    """
    Plot insertion and deletion curves for all experiments.
    
    Args:
        results_dict: dict with structure {experiment_name: {'insertion': results, 'deletion': results}}
        save_path: path to save the plot
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Deletion plot
    for exp_name, results in results_dict.items():
        if 'deletion' in results:
            del_res = results['deletion']
            ax1.plot(del_res['percentages'] * 100, del_res['accuracies'] * 100, 
                    label=f"{exp_name} (AUC={del_res['auc']:.3f})", marker='o', markersize=3)
    
    ax1.set_xlabel('Percentage of Pixels Removed (%)')
    ax1.set_ylabel('Top-1 Accuracy (%)')
    ax1.set_title('Deletion Test')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Insertion plot
    for exp_name, results in results_dict.items():
        if 'insertion' in results:
            ins_res = results['insertion']
            ax2.plot(ins_res['percentages'] * 100, ins_res['accuracies'] * 100,
                    label=f"{exp_name} (AUC={ins_res['auc']:.3f})", marker='o', markersize=3)
    
    ax2.set_xlabel('Percentage of Pixels Added (%)')
    ax2.set_ylabel('Top-1 Accuracy (%)')
    ax2.set_title('Insertion Test')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Insertion/Deletion curves saved to {save_path}")