import torch
import matplotlib.pyplot as plt
import numpy as np
from Network import GroundToAerialMatchingModel
from gradcam import GradCAM
from Utils import *

# Dummy input
batch_grd = torch.randn([1, 128, 512, 3]).float()  # [B, H, W, C]
polar_sat = torch.randn(1, 128, 512, 3).float()
segmap = torch.randn(1, 128, 512, 3).float()

model = GroundToAerialMatchingModel()
model.train()

# Set target layer from ground branch
gradcam = GradCAM(model.ground_branch, target_layer_name="features.21")

# Forward
batch_grd_tensor = batch_grd.clone().detach().requires_grad_(True)
sat_tensor = polar_sat.clone().detach()
seg_tensor = segmap.clone().detach()

out = model(batch_grd_tensor, sat_tensor, seg_tensor)
sat_matrix, grd_matrix, distance, _ = out

# Loss fittizia
loss = torch.mean(distance)
loss.backward()

# GradCAM
# GradCAM
target_size = (batch_grd.shape[1], batch_grd.shape[2])  # (H, W)
cam_map = gradcam.generate(target_size=target_size)

# Plot
fig, ax = plt.subplots(3, 1, figsize=(5, 7))

# Original image
img_np = batch_grd.squeeze().detach().cpu().numpy()
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)  # normalize to [0,1]
ax[0].imshow(img_np)
ax[0].set_title("Original Ground Image")
ax[0].axis('on')

# GradCAM heatmap
im = ax[1].imshow(cam_map, cmap='jet')
ax[1].set_title("GradCAM Ground Image")
ax[1].axis('on')

# Overlay heatmap on image
if img_np.shape[-1] == 3:
    overlay = img_np.copy()
    cam_colored = plt.get_cmap('jet')(cam_map)[..., :3]  # remove alpha
    print(f"original image shape: {img_np.shape}")
    print(f"colored heatmap image shape: {cam_map.shape}")
    overlay = 0.5 * overlay + 0.5 * cam_colored  # blend
    overlay = np.clip(overlay, 0, 1)
    ax[2].imshow(overlay)
    ax[2].set_title("Overlay: Image + CAM")
    ax[2].axis('on')
else:
    ax[2].imshow(cam_map, cmap='jet')
    ax[2].set_title("Overlay not available")
    ax[2].axis('off')

# Colorbar for heatmap
cbar = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
cbar.set_label("Activation Intensity", rotation=270, labelpad=15)

plt.tight_layout()
plt.show()

save_overlay_image(batch_grd, cam_map, path=f"plots/plot_di_prova.png")
