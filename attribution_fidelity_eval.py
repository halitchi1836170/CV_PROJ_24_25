import os
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List
from Globals import BLOCKING_COUNTER, config, images_params, GROUND_STD, GROUND_MEAN, gradcam_config
from logger import log
from Train import validate_original
from Network import gradcam_from_activations, HookManager, compute_gradcam_from_acts_grads
import matplotlib.pyplot as plt


def rescale_normalized_for_display(img_normalized):
    """
    Rescala immagini normalizzate per visualizzazione.
    Porta dal range della normalizzazione a [0, 1] usando min-max scaling.
    
    Args:
        img_normalized: torch.Tensor [H, W, C] normalizzato
    
    Returns:
        np.ndarray [H, W, C] in range [0, 1]
    """
    img = img_normalized.cpu().numpy()
    
    # Min-max scaling per ogni immagine
    img_min = img.min()
    img_max = img.max()
    
    if img_max - img_min > 1e-6:
        img_scaled = (img - img_min) / (img_max - img_min)
    else:
        img_scaled = np.zeros_like(img)
    
    return np.clip(img_scaled, 0, 1)


def save_normalized_comparison(grd_original_norm, grd_important_norm, grd_unimportant_norm,mask_important, mask_unimportant, sample_idx, save_dir):
    
    # Rescale per visualizzazione (min-max scaling)
    
    # img_orig = rescale_normalized_for_display(grd_original_norm[0])
    # img_important = rescale_normalized_for_display(grd_important_norm[0])
    # img_unimportant = rescale_normalized_for_display(grd_unimportant_norm[0])
    
    img_orig = np.clip(grd_original_norm[0])
    img_important = np.clip(grd_important_norm[0])
    img_unimportant = np.clip(grd_unimportant_norm[0])
    
    mask_imp = mask_important[0].cpu().numpy()
    mask_unimp = mask_unimportant[0].cpu().numpy()
    
    # Calcola differenze assolute
    diff_important = np.abs(img_important - img_orig)
    diff_unimportant = np.abs(img_unimportant - img_orig)
    
    # Statistiche
    noise_stats_imp = {
        'mean': diff_important[mask_imp > 0.5].mean() if mask_imp.sum() > 0 else 0,
        'max': diff_important[mask_imp > 0.5].max() if mask_imp.sum() > 0 else 0,
        'std': diff_important[mask_imp > 0.5].std() if mask_imp.sum() > 0 else 0
    }
    
    noise_stats_unimp = {
        'mean': diff_unimportant[mask_unimp > 0.5].mean() if mask_unimp.sum() > 0 else 0,
        'max': diff_unimportant[mask_unimp > 0.5].max() if mask_unimp.sum() > 0 else 0,
        'std': diff_unimportant[mask_unimp > 0.5].std() if mask_unimp.sum() > 0 else 0
    }
    
    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    
    # Row 1: Originale, Important perturbed, Unimportant perturbed
    axes[0, 0].imshow(img_important)
    axes[0, 0].set_title('Important Pixels Perturbed\n(Model Input)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_unimportant)
    axes[0, 1].set_title('Unimportant Pixels Perturbed\n(Model Input)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Row 2: Maschere
    axes[1, 0].imshow(mask_imp, cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Important Mask\n({mask_imp.sum():.0f} pixels)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask_unimp, cmap='Blues', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Unimportant Mask\n({mask_unimp.sum():.0f} pixels)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Row 3: Differenze assolute
    im1 = axes[2, 0].imshow(diff_important, cmap='hot', vmin=0, vmax=1)
    axes[2, 0].set_title(f'|Diff| Important\nMean: {noise_stats_imp["mean"]:.3f}, Max: {noise_stats_imp["max"]:.3f}',fontsize=10, fontweight='bold')
    axes[2, 0].axis('off')
    plt.colorbar(im1, ax=axes[2, 0], fraction=0.046)
    
    im2 = axes[2, 1].imshow(diff_unimportant, cmap='hot', vmin=0, vmax=1)
    axes[2, 1].set_title(f'|Diff| Unimportant\nMean: {noise_stats_unimp["mean"]:.3f}, Max: {noise_stats_unimp["max"]:.3f}',fontsize=10, fontweight='bold')
    axes[2, 1].axis('off')
    plt.colorbar(im2, ax=axes[2, 1], fraction=0.046)
    
    plt.suptitle(f'Normalized Inputs Comparison - Sample {sample_idx}\n(As seen by the model)',fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    filename = f'normalized_inputs_sample_{sample_idx:04d}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Log statistics
    log.info(f"Saved input batchs comparison in: {filepath}")
    #log.info(f"Sample {sample_idx} - Normalized input stats:")
    #log.info(f"Important perturbation - Mean diff: {noise_stats_imp['mean']:.4f}, Max: {noise_stats_imp['max']:.4f}, Std: {noise_stats_imp['std']:.4f}")
    #log.info(f"Unimportant perturbation - Mean diff: {noise_stats_unimp['mean']:.4f}, Max: {noise_stats_unimp['max']:.4f}, Std: {noise_stats_unimp['std']:.4f}")
    
    return noise_stats_imp, noise_stats_unimp

# def compute_gradcam(model, grd_img, polar_sat_img, segmap_img, device):
#     model.eval()
#     torch.set_grad_enabled(True)
    
#     grd_img_grad = grd_img.clone().requires_grad_(True)
#     polar_sat_img_grad = polar_sat_img.clone().requires_grad_(True)
    
#     grd_hm=HookManager(model, f'ground_branch.features.21')
#     sat_hm=HookManager(model, f'satellite_branch.features.21')
    
#     sat_matrix, grd_matrix, distance, pred_orien = model(grd_img_grad, polar_sat_img_grad, segmap_img)
#     target_scalar = -distance.diag().sum()
    
#     model.zero_grad()
#     target_scalar.backward(retain_graph=False)
    
#     grd_acts, grd_grads = grd_hm.activations, grd_hm.gradients
#     sat_acts, sat_grads = sat_hm.activations, sat_hm.gradients
    
#     grd_cam = compute_gradcam_from_acts_grads(grd_acts, grd_grads, upsample_to=(grd_img.shape[2], grd_img.shape[3]))
#     sat_cam = compute_gradcam_from_acts_grads(sat_acts, sat_grads, upsample_to=(grd_img.shape[2], grd_img.shape[3]))
    
#     grd_hm.remove()
#     return grd_cam, sat_cam

# def compute_gradcam(model, grd_img, polar_sat_img, segmap_img, device):
    
#     model.eval()

#     # Forward con gradienti attivi solo su ground image
#     grd_img_grad = grd_img.clone().requires_grad_(True)

#     # Forward pass
#     sat_matrix, grd_matrix, distance, pred_orien = model(grd_img_grad, polar_sat_img, segmap_img)

#     # Target scalar (esempio: diagonale di distance)
#     target_scalar = -distance.diag().sum()

#     # Backward
#     model.zero_grad()
#     #target_scalar.backward()

#     # Ora puoi accedere alle attivazioni salvate automaticamente:
#     grd_acts = model.ground_branch.target_acts   # già salvate nel forward
#     sat_acts = model.satellite_branch.target_acts

#     # Ma ti servono anche i gradienti rispetto a questi activations:
#     grd_grads = grd_acts.grad
#     sat_grads = sat_acts.grad

#     # (Se None, devi forzare retain_grad() nel forward hook originale, vedi sotto)
    
#     cam_size = (grd_img.shape[2], grd_img.shape[3])  # height, width

#     grd_dict = {'activations': grd_acts, 'gradients': grd_grads}
#     sat_dict = {'activations': sat_acts, 'gradients': sat_grads}

#     grd_cam = gradcam_from_activations(
#         acts=grd_dict['activations'],
#         target_scalar=target_scalar,
#         upsample_to_hw=cam_size,
#         normalize=True
#     )
#     sat_cam = gradcam_from_activations(
#         acts=sat_dict['activations'],
#         target_scalar=target_scalar,
#         upsample_to_hw=cam_size,
#         normalize=True
#     )

#     return grd_cam, sat_cam

def add_noise_to_pixels(images: torch.Tensor, mask: torch.Tensor, noise_mean: float = 0.0, noise_std: float = 0.8,return_visual: bool = False) -> torch.Tensor:
    B, H, W, C = images.shape
    GROUND_STD_T = torch.tensor(GROUND_STD, device=images.device, dtype=torch.float32)
    GROUND_MEAN_T = torch.tensor(GROUND_MEAN, device=images.device, dtype=torch.float32)
    
    denorm = images * GROUND_STD_T + GROUND_MEAN_T
    
    noise = torch.randn_like(denorm) * noise_std + noise_mean
    
    mask_expanded = mask.unsqueeze(-1)
    
    noisy_denorm = torch.where(mask_expanded > 0.5, denorm + noise, denorm)
    noisy_denorm = torch.clamp(noisy_denorm, 0, 1)
    
    if return_visual:
        return (noisy_denorm-GROUND_MEAN_T)/GROUND_STD_T, noisy_denorm  # per modello, per plot
    else:
        return (noisy_denorm-GROUND_MEAN_T)/GROUND_STD_T

def denormalize_image(img):
    """Denormalize image for visualization."""
    img = img * GROUND_STD + GROUND_MEAN
    img = np.clip(img, 0, 1)
    return img

def save_paired_gradcam_heatmap(type, exp_name, grd_original, grd_cam, sat_original, sat_cam, sample_idx, save_dir, alpha=0.75):
    
    img_orig = grd_original[0].cpu().numpy()
    img_orig = denormalize_image(img_orig)
    
    sat_orig = sat_original[0].cpu().numpy()
    sat_orig = denormalize_image(sat_orig)
    
    grd_cam_viz = grd_cam[0].cpu().numpy()
    sat_cam_viz = sat_cam[0].cpu().numpy()
    
    fig, axes = plt.subplots(3,2,figsize=(12,10))
    
    axes[0,0].imshow(img_orig)
    axes[0,0].set_title('GRD Original', fontsize=12)
    axes[0,0].axis('on')
    
    im1 = axes[1,0].imshow(grd_cam_viz, cmap='jet')
    axes[1,0].set_title('GRD GradCAM', fontsize=12)
    axes[1,0].axis('on')
    plt.colorbar(im1, ax=axes[1,0], fraction=0.046)
    
    norm = Normalize(vmin=np.nanmin(grd_cam_viz),vmax=np.nanmax(grd_cam_viz))
    axes[2,0].imshow(img_orig)
    axes[2,0].imshow(grd_cam_viz, cmap='jet', norm=norm, alpha=norm(grd_cam_viz)*alpha, interpolation='nearest')
    axes[2,0].set_title(f"{exp_name}-GRD Overlay")
    axes[2,0].axis('on')
    
    axes[0,1].imshow(sat_orig)
    axes[0,1].set_title('SAT Original', fontsize=12)
    axes[0,1].axis('on')
    
    im2 = axes[1,1].imshow(sat_cam_viz, cmap='jet')
    axes[1,1].set_title('SAT GradCAM', fontsize=12)
    axes[1,1].axis('on')
    plt.colorbar(im2, ax=axes[1,1], fraction=0.046)
    
    norm = Normalize(vmin=np.nanmin(sat_cam_viz),vmax=np.nanmax(sat_cam_viz))
    axes[2,1].imshow(sat_orig)
    axes[2,1].imshow(sat_cam_viz, cmap='jet', norm=norm, alpha=norm(sat_cam_viz)*alpha, interpolation='nearest')
    axes[2,1].set_title(f"{exp_name}-SAT Overlay")
    axes[2,1].axis('on')
    
    plt.suptitle(f'{type} - {exp_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    filename = f'{type}_overlay_sample_{sample_idx:04d}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    #log.info(f"Saved overlay example image: {filepath}")
    
    

def save_only_gradcam_heatmap(type, exp_name, grd_original, cam_resized, sample_idx, save_dir, alpha=0.75):
    
    img_orig = grd_original[0].cpu().numpy()
    img_orig = denormalize_image(img_orig)
    
    cam_viz = cam_resized[0].cpu().numpy()
    
    fig, axes = plt.subplots(3,1,figsize=(5,7))

    axes[0].imshow(img_orig)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('on')
    
    im1 = axes[1].imshow(cam_viz, cmap='jet')
    axes[1].set_title('GradCAM Heatmap', fontsize=12)
    axes[1].axis('on')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    norm = Normalize(vmin=np.nanmin(cam_viz),vmax=np.nanmax(cam_viz))
    
    axes[2].imshow(img_orig)
    axes[2].imshow(cam_viz, cmap='jet', norm=norm, alpha=norm(cam_viz)*alpha, interpolation='nearest')
    axes[2].set_title(f"{exp_name}-Overlay: Image + GradCAM")
    axes[2].axis('on')
    
    
    plt.suptitle(f'{type} - {exp_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    filename = f'{type}_overlay_sample_{sample_idx:04d}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    #log.info(f"Saved overlay example image: {filepath}")
    

def save_af_example_image(type, grd_original, grd_important, grd_unimportant, cam_resized, mask_important, mask_unimportant,sample_idx, save_dir):
    """
    Save visualization of original image, GradCAM, and perturbed versions.
    """
    # Take first image from batch
    img_orig = grd_original[0].cpu().numpy()
    img_important = grd_important[0].cpu().numpy()
    img_unimportant = grd_unimportant[0].cpu().numpy()

    cam_viz = cam_resized[0].cpu().numpy()
    mask_imp = mask_important[0].cpu().numpy()
    mask_unimp = mask_unimportant[0].cpu().numpy()
    
    img_orig = denormalize_image(img_orig)
    #img_important = denormalize_image(img_important)
    #img_unimportant = denormalize_image(img_unimportant)
    
    # Create figure with 6 subplots
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    
    # Row 1: Original, GradCAM, Important pixels mask
    axes[0, 0].imshow(img_orig)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('on')
    
    im1 = axes[0, 1].imshow(cam_viz, cmap='jet')
    axes[0, 1].set_title('GradCAM Heatmap', fontsize=12, fontweight='bold')
    axes[0, 1].axis('on')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    axes[1, 0].imshow(mask_imp, cmap='Reds', vmin=0, vmax=1)
    axes[1, 0].set_title('Important Pixels Mask (2%)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('on')
    
    axes[1, 1].imshow(img_important)
    axes[1, 1].set_title('Important Pixels Perturbed', fontsize=12, fontweight='bold')
    axes[1, 1].axis('on')
    
    axes[2, 0].imshow(mask_unimp, cmap='Blues', vmin=0, vmax=1)
    axes[2, 0].set_title('Unimportant Pixels Mask (2%)', fontsize=12, fontweight='bold')
    axes[2, 0].axis('on')
    
    axes[2, 1].imshow(img_unimportant)
    axes[2, 1].set_title('Unimportant Pixels Perturbed', fontsize=12, fontweight='bold')
    axes[2, 1].axis('on')
    
    plt.suptitle(f'Attribution Fidelity Test - Sample {sample_idx}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    filename = f'AF_{type}_sample_{sample_idx:04d}.png'
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    #log.info(f"Saved AF-{type} example image: {filepath}")


def compute_single_image_accuracy(model, grd_img, sat_polar, seg, sat_descriptor, device):
    """
    Compute accuracy for a single ground image against all satellite images.
    
    Args:
        model: trained model
        grd_img: [1, H, W, C] single ground image
        sat_polar: [1, H, W, C] corresponding satellite image
        seg: [1, H, W, C] segmentation map
        sat_descriptor: [N, D] all satellite descriptors
        device: torch device
    
    Returns:
        accuracy: 1.0 if top-1 match is correct, 0.0 otherwise
    """
    with torch.no_grad():
        _, grd_mat, _, _ = model(grd_img, sat_polar, seg)
        
        # Get descriptor dimensions
        g_height, g_width, g_channel = list(grd_mat.size())[1:]
        
        # Flatten ground descriptor
        grd_desc = grd_mat.cpu().numpy()
        grd_desc = np.reshape(grd_desc, [-1, g_height * g_width * g_channel])
        
        # Compute distances
        dist = 2.0 - 2.0 * np.matmul(grd_desc, np.transpose(sat_descriptor))
        
        # Check if top-1 is correct (index 0 should be smallest distance)
        top1_idx = np.argmin(dist[0])
        
        return 1.0 if top1_idx == 0 else 0.0


def attribution_fidelity_test(
    experiment_name,
    model,
    unperturbed_accuracy,
    data,
    device,
    batch_size,
    grd_FOV,
    grd_noise,
    perturbation_pct=0.02,
    noise_mean=0.0,
    noise_std=0.8,
    save_path=None,
    num_examples_to_save=1
):
    log.info("Starting ATTRIBUTION FIDELITY test...")
    log.info(f"Perturbation: {perturbation_pct*100:.1f}% pixels, noise N({noise_mean}, {noise_std}²)")
    
    images_dir = None
    if save_path:
        images_dir = os.path.join(os.path.dirname(save_path), 'af_examples')
        os.makedirs(images_dir, exist_ok=True)
        log.info(f"Example images will be saved to: {images_dir}")
    
    n_test_samples = data.get_test_dataset_size()
    
    train_grd_FOV = config["train_grd_FOV"]
    max_angle = images_params["max_angle"]
    max_width = images_params["max_width"]
    width = int(train_grd_FOV / max_angle * max_width)
    
    # Get dimensions from dummy forward pass
    grd_x = torch.zeros([2, int(max_width/4), width, 3]).to(device)
    polar_sat_x = torch.zeros([2, int(max_width/4), max_width, 3]).to(device)
    segmap_x = torch.zeros([2, int(max_width/4), max_width, 3]).to(device)
    
    sat_matrix, grd_matrix, _, _ = model(grd_x, polar_sat_x, segmap_x)
    s_height, s_width, s_channel = list(sat_matrix.size())[1:]
    g_height, g_width, g_channel = list(grd_matrix.size())[1:]
    
    # =========== PASS 1: IMPORTANT PIXELS PERTURBED ===========
    log.info("Pass 1: Computing descriptors with IMPORTANT pixels perturbed...")
    data.__cur_id = 0
    
    sat_global_matrix_imp = np.zeros([n_test_samples, s_height, s_width, s_channel])
    grd_global_matrix_imp = np.zeros([n_test_samples, g_height, g_width, g_channel])
    
    val_i = 0
    counter = 0
    examples_saved = 0
    
    hm_grd = HookManager(model, f"ground_branch.{gradcam_config['visualization_layer']}")
    hm_sat = HookManager(model, f"satellite_branch.{gradcam_config['visualization_layer']}")
    
    while True:
        if counter + 1 == BLOCKING_COUNTER:
            break
        
        batch_sat_polar, batch_sat, batch_grd, batch_segmap, batch_orien = data.next_batch_scan(batch_size, grd_noise=grd_noise, FOV=grd_FOV)
        
        if batch_sat is None:
            break
        
        grd = torch.from_numpy(batch_grd).float().to(device)
        sat_polar = torch.from_numpy(batch_sat_polar).float().to(device)
        seg = torch.from_numpy(batch_segmap).float().to(device)
        
        # Compute GradCAM
        #cam, sat_cam = compute_gradcam(model, grd.clone().requires_grad_(True), sat_polar, seg, device)
        
        torch.set_grad_enabled(True)
        model.eval()
        
        sat_matrix, grd_matrix, distance, pred_orien = model(grd.clone().requires_grad_(True), sat_polar, seg)
        target_scalar = -distance.diag().sum()
        
        model.zero_grad()
        target_scalar.backward()
        
        grd_cam = compute_gradcam_from_acts_grads(hm_grd.activations, hm_grd.gradients, upsample_to=(grd.shape[1], grd.shape[2]))
        sat_cam = compute_gradcam_from_acts_grads(hm_sat.activations, hm_sat.gradients, upsample_to=(grd.shape[1], grd.shape[2]))
        
        torch.set_grad_enabled(False)
        
        with torch.no_grad():
            # Resize CAM to match input image size
            cam_resized = F.interpolate(grd_cam.unsqueeze(1),size=(grd.shape[1], grd.shape[2]),mode='bilinear',align_corners=False).squeeze(1)  # [B, H, W]
            sat_cam_resized = F.interpolate(sat_cam.unsqueeze(1),size=(grd.shape[1], grd.shape[2]),mode='bilinear',align_corners=False).squeeze(1)  # [B, H, W]
            
            B, H, W = cam_resized.shape
            num_pixels = H * W
            num_to_perturb = int(num_pixels * perturbation_pct)
            
            # Create mask for important pixels
            mask_important = torch.zeros_like(cam_resized)
            sat_mask_important = torch.zeros_like(sat_cam_resized)
            
            if num_to_perturb > 0:
                for b in range(B):
                    cam_flat = cam_resized[b].view(-1)
                    # Get indices of top pixels (most important)
                    _, important_indices = torch.topk(cam_flat, num_to_perturb, largest=True)
                    mask_flat = mask_important[b].view(-1)
                    mask_flat[important_indices] = 1
                    mask_important[b] = mask_flat.view(H, W)
                    
                    sat_cam_flat = sat_cam_resized[b].view(-1)
                    _, sat_important_indices = torch.topk(sat_cam_flat, num_to_perturb, largest=True)
                    sat_mask_flat = sat_mask_important[b].view(-1)
                    sat_mask_flat[sat_important_indices] = 1
                    sat_mask_important[b] = sat_mask_flat.view(H, W)
            
            # Perturb important pixels
            grd_important_perturbed, grd_important_vis = add_noise_to_pixels(grd, mask_important, noise_mean, noise_std, return_visual=True)
            sat_important_perturbed, sat_important_vis = add_noise_to_pixels(sat_polar, sat_mask_important, noise_mean, noise_std, return_visual=True)
            
            # Save example images (only from first pass)
            if examples_saved < num_examples_to_save and images_dir:
                # We'll save with unimportant mask too, but compute it here
                mask_unimportant = torch.zeros_like(cam_resized)
                sat_mask_unimportant = torch.zeros_like(sat_cam_resized)
                
                if num_to_perturb > 0:
                    for b in range(B):
                        cam_flat = cam_resized[b].view(-1)
                        _, unimportant_indices = torch.topk(cam_flat, num_to_perturb, largest=False)
                        mask_flat = mask_unimportant[b].view(-1)
                        mask_flat[unimportant_indices] = 1
                        mask_unimportant[b] = mask_flat.view(H, W)
                        
                        sat_cam_flat = sat_cam_resized[b].view(-1)
                        _, sat_unimportant_indices = torch.topk(sat_cam_flat, num_to_perturb, largest=False)
                        sat_mask_flat = sat_mask_unimportant[b].view(-1)
                        sat_mask_flat[sat_unimportant_indices] = 1
                        sat_mask_unimportant[b] = sat_mask_flat.view(H, W)
                
                grd_unimportant_perturbed, grd_unimportant_vis = add_noise_to_pixels(grd, mask_unimportant, noise_mean, noise_std, return_visual=True)
                sat_unimportant_perturbed, sat_unimportant_vis = add_noise_to_pixels(sat_polar, sat_mask_unimportant, noise_mean, noise_std, return_visual=True)
                
                for b in range(min(B,num_examples_to_save-examples_saved)):
                    save_af_example_image(
                        "GRD",grd[b:b+1], grd_important_vis[b:b+1], grd_unimportant_vis[b:b+1],
                        cam_resized[b:b+1], mask_important[b:b+1], mask_unimportant[b:b+1],
                        examples_saved, images_dir
                    )
                    save_af_example_image(
                        "SAT",sat_polar[b:b+1], sat_important_vis[b:b+1], sat_unimportant_vis[b:b+1],
                        sat_cam_resized[b:b+1], sat_mask_important[b:b+1], sat_mask_unimportant[b:b+1],
                        examples_saved, images_dir
                    )
                    save_only_gradcam_heatmap("GRD",experiment_name,grd[b:b+1],cam_resized[b:b+1],examples_saved,images_dir)
                    save_only_gradcam_heatmap("SAT",experiment_name,sat_polar[b:b+1],sat_cam_resized[b:b+1],examples_saved,images_dir)
                    save_paired_gradcam_heatmap("GRD_SAT",experiment_name,grd[b:b+1],cam_resized[b:b+1],sat_polar[b:b+1],sat_cam_resized[b:b+1],examples_saved,images_dir)
                    # save_normalized_comparison(
                    #     grd[b:b+1],                      # Originale normalizzato
                    #     grd_important_perturbed[b:b+1],  # Important normalizzato
                    #     grd_unimportant_perturbed[b:b+1], # Unimportant normalizzato
                    #     mask_important[b:b+1],
                    #     mask_unimportant[b:b+1],
                    #     examples_saved,
                    #     images_dir
                    # )
                    examples_saved += 1
            #log.info(f"Inferencing the important pixel perturbed...")
            # Forward pass with perturbed images
            sat_mat, grd_mat, _, _ = model(grd, sat_important_perturbed, seg)
            #log.info("Inference terminated.")
        
        sat_global_matrix_imp[val_i:val_i + sat_mat.shape[0], :] = sat_mat.cpu().detach().numpy()
        grd_global_matrix_imp[val_i:val_i + grd_mat.shape[0], :] = grd_mat.cpu().detach().numpy()
        val_i += sat_mat.shape[0]
        counter += 1
    
    hm_grd.remove()
    hm_sat.remove()
    
    # Compute accuracy for important pixels perturbed
    sat_descriptor_imp = np.reshape(sat_global_matrix_imp[:, :, :g_width, :],[-1, g_height * g_width * g_channel])
    norm = np.linalg.norm(sat_descriptor_imp, axis=-1, keepdims=True)
    sat_descriptor_imp = sat_descriptor_imp / np.maximum(norm, 1e-12)
    grd_descriptor_imp = np.reshape(grd_global_matrix_imp, [-1, g_height * g_width * g_channel])
    
    dist_array_imp = 2.0 - 2.0 * np.matmul(grd_descriptor_imp, np.transpose(sat_descriptor_imp))
    acc_important = validate_original(dist_array_imp, 1)
    
    log.info(f"Accuracy with IMPORTANT pixels perturbed: {acc_important*100:.2f}%")
    
    # =========== PASS 2: UNIMPORTANT PIXELS PERTURBED ===========
    log.info("Pass 2: Computing descriptors with UNIMPORTANT pixels perturbed...")
    data.__cur_id = 0
    
    sat_global_matrix_unimp = np.zeros([n_test_samples, s_height, s_width, s_channel])
    grd_global_matrix_unimp = np.zeros([n_test_samples, g_height, g_width, g_channel])
    
    val_i = 0
    counter = 0
    
    hm_grd = HookManager(model, f"ground_branch.{gradcam_config['visualization_layer']}")
    hm_sat = HookManager(model, f"satellite_branch.{gradcam_config['visualization_layer']}")
    
    while True:
        if counter + 1 == BLOCKING_COUNTER:
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
        #cam, sat_cam = compute_gradcam(model, grd.clone().requires_grad_(True), sat_polar, seg, device)
        
        torch.set_grad_enabled(True)
        model.eval()
        
        sat_matrix, grd_matrix, distance, pred_orien = model(grd.clone().requires_grad_(True), sat_polar, seg)
        target_scalar = -distance.diag().sum()
        
        model.zero_grad()
        target_scalar.backward()
        
        grd_cam = compute_gradcam_from_acts_grads(hm_grd.activations, hm_grd.gradients, upsample_to=(grd.shape[1], grd.shape[2]))
        sat_cam = compute_gradcam_from_acts_grads(hm_sat.activations, hm_sat.gradients, upsample_to=(grd.shape[1], grd.shape[2]))
        
        torch.set_grad_enabled(False)
        
        with torch.no_grad():
            # Resize CAM to match input image size
            cam_resized = F.interpolate(grd_cam.unsqueeze(1),size=(grd.shape[1], grd.shape[2]),mode='bilinear',align_corners=False).squeeze(1)  # [B, H, W]
            sat_cam_resized = F.interpolate(sat_cam.unsqueeze(1),size=(grd.shape[1], grd.shape[2]),mode='bilinear',align_corners=False).squeeze(1)  # [B, H, W]
            
            B, H, W = cam_resized.shape
            num_pixels = H * W
            num_to_perturb = int(num_pixels * perturbation_pct)
            
            # Create mask for unimportant pixels
            mask_unimportant = torch.zeros_like(cam_resized)
            sat_mask_unimportant = torch.zeros_like(sat_cam_resized)
            
            if num_to_perturb > 0:
                for b in range(B):
                    cam_flat = cam_resized[b].view(-1)
                    # Get indices of bottom pixels (least important)
                    _, unimportant_indices = torch.topk(cam_flat, num_to_perturb, largest=False)
                    mask_flat = mask_unimportant[b].view(-1)
                    mask_flat[unimportant_indices] = 1
                    mask_unimportant[b] = mask_flat.view(H, W)
                    
                    sat_cam_flat = sat_cam_resized[b].view(-1)
                    # Get indices of bottom pixels (least important)
                    _, sat_unimportant_indices = torch.topk(sat_cam_flat, num_to_perturb, largest=False)
                    sat_mask_flat = sat_mask_unimportant[b].view(-1)
                    sat_mask_flat[sat_unimportant_indices] = 1
                    sat_mask_unimportant[b] = sat_mask_flat.view(H, W)
            
            # Perturb unimportant pixels
            grd_unimportant_perturbed = add_noise_to_pixels(grd, mask_unimportant, noise_mean, noise_std)
            sat_unimportant_perturbed = add_noise_to_pixels(sat_polar, sat_mask_unimportant, noise_mean, noise_std)
            
            #log.info(f"Inferencing the unimportant pixel perturbed...")
            # Forward pass with perturbed images
            sat_mat, grd_mat, _, _ = model(grd, sat_unimportant_perturbed, seg)
            #log.info("Inference terminated.")
        
        sat_global_matrix_unimp[val_i:val_i + sat_mat.shape[0], :] = sat_mat.cpu().detach().numpy()
        grd_global_matrix_unimp[val_i:val_i + grd_mat.shape[0], :] = grd_mat.cpu().detach().numpy()
        val_i += sat_mat.shape[0]
        counter += 1
    
    hm_grd.remove()
    hm_sat.remove()
    
    # Compute accuracy for unimportant pixels perturbed
    sat_descriptor_unimp = np.reshape(sat_global_matrix_unimp[:, :, :g_width, :],[-1, g_height * g_width * g_channel])
    norm = np.linalg.norm(sat_descriptor_unimp, axis=-1, keepdims=True)
    sat_descriptor_unimp = sat_descriptor_unimp / np.maximum(norm, 1e-12)
    grd_descriptor_unimp = np.reshape(grd_global_matrix_unimp, [-1, g_height * g_width * g_channel])
    
    dist_array_unimp = 2.0 - 2.0 * np.matmul(grd_descriptor_unimp, np.transpose(sat_descriptor_unimp))
    acc_unimportant = validate_original(dist_array_unimp, 1)
    
    log.info(f"Accuracy with UNIMPORTANT pixels perturbed: {acc_unimportant*100:.2f}%")
    
    # =========== COMPUTE AF METRIC ===========
    diff1 = abs(acc_important-unperturbed_accuracy)
    diff2 = abs(acc_unimportant-unperturbed_accuracy)
    denominator = diff1+diff2
    if denominator > 0:
        af_score = (diff1 - diff2) / denominator
    else:
        af_score = 0.0
    
    log.info(f"Attribution Fidelity Score: {af_score:.4f}")
    log.info(f"Accuracy (Important Perturbed): {acc_important:.4f}")
    log.info(f"Accuracy (Unimportant Perturbed): {acc_unimportant:.4f}")
    
    results = {
        'af_score': af_score,
        'acc_important': acc_important,
        'acc_unimportant': acc_unimportant,
        'num_samples': n_test_samples
    }
    
    if save_path:
        np.save(save_path, results)
        log.info(f"\nAttribution Fidelity results saved to {save_path}")
    
    return results


def plot_af_results(results_dict: Dict, save_path: str):
    """
    Plot Attribution Fidelity results for all experiments.
    
    Args:
        results_dict: dict with structure {experiment_name: af_results}
        save_path: path to save the plot
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prepare data
    exp_names = list(results_dict.keys())
    af_scores = [results_dict[name]['af_score'] for name in exp_names]
    acc_important = [results_dict[name]['acc_important'] for name in exp_names]
    acc_unimportant = [results_dict[name]['acc_unimportant'] for name in exp_names]
    
    # Plot 1: AF Scores
    ax1 = axes[0]
    x_pos = np.arange(len(exp_names))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(exp_names)]
    bars = ax1.bar(x_pos, af_scores, alpha=0.7, color=colors)
    ax1.set_ylabel('Attribution Fidelity Score', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax1.set_title('Attribution Fidelity Scores', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(exp_names, rotation=45, ha='right')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.grid(axis='y', alpha=0.3)
    #ax1.set_ylim(-1, 1)
    
    # Add value labels on bars
    for bar, val in zip(bars, af_scores):
        height = bar.get_height()
        y_pos = height + 0.02 if height >= 0 else height - 0.05
        ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.4f}', ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    # Plot 2: Accuracy comparison
    ax2 = axes[1]
    x = np.arange(len(exp_names))
    width = 0.35
    bars1 = ax2.bar(x - width/2, acc_important, width, label='Important Pixels Perturbed',
                    alpha=0.7, color='#d62728')
    bars2 = ax2.bar(x + width/2, acc_unimportant, width, label='Unimportant Pixels Perturbed',
                    alpha=0.7, color='#2ca02c')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy: Important vs Unimportant Perturbation', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"Attribution Fidelity plots saved to {save_path}")
