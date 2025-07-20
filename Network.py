import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from Globals import *
import torch.nn.functional as F
from logger import log

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer = dict([*model.named_modules()])[target_layer_name]
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, target_size):
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        grad_cam = F.relu(grad_cam)
        grad_cam = F.interpolate(grad_cam, size=target_size, mode='bilinear', align_corners=False)
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)
        return grad_cam.squeeze().detach().cpu().numpy()


def compute_saliency_loss(gradcam, inputs, cam_size):
    cam = gradcam.generate(target_size=cam_size)
    cam_tensor = torch.from_numpy(cam).to(inputs.device)
    return -torch.mean(cam_tensor)


class ProcessFeatures(nn.Module):
    """
    Feature processing module for computing correlation and distance matrices
    """

    def __init__(self):
        super(ProcessFeatures, self).__init__()

    def VGG_13_conv_v2_cir(self, sat_features, grd_features):
        sat_matrix, distance, pred_orien = self.corr_crop_distance(sat_features, grd_features)
        return sat_features, grd_features, distance, pred_orien

    def corr_crop_distance(self, sat_vgg, grd_vgg):
        corr_out, corr_orien = self.corr(sat_vgg, grd_vgg)
        sat_cropped = self.crop_sat(sat_vgg, corr_orien, grd_vgg.size()[2])
        # shape = [batch_sat, batch_grd, h, grd_width, channel]

        norm = torch.sqrt(torch.sum(sat_cropped ** 2, dim=[2, 3, 4], keepdim=True) + 1e-8)
        sat_matrix = sat_cropped / norm

        grd_expanded = grd_vgg.unsqueeze(0)
        dot_product = torch.sum(sat_matrix * grd_expanded, dim=[2, 3, 4])
        distance = 2 - 2 * dot_product.t()

        return sat_matrix, distance, corr_orien

    def warp_pad_columns(self, x, n):
        """Padding circolare per le colonne"""
        padded = torch.cat([x, x[:, :, :, :n]], dim=3)
        return padded

    def corr(self, sat_matrix, grd_matrix):
        batch_sat, s_c, s_h, s_w = sat_matrix.shape
        batch_grd, g_c, g_h, g_w = grd_matrix.shape

        assert s_h == g_h, s_c == g_c  # devono avere la stessa altezza e lo stesso numero di canali

        n = g_w - 1

        def inner_warp_pad_columns(x, n):
            out = torch.concat([x, x[:, :, :n, :]], dim=2)
            return out

        x = inner_warp_pad_columns(sat_matrix, n)

        correlations = []
        for i in range(batch_sat):
            sat_single = x[i:i + 1]  # [1, channel, height, width_padded]

            # Correlazione con tutti i ground features
            corr_results = []
            for j in range(batch_grd):
                grd_single = grd_matrix[j:j + 1]  # [1, channel, height, width]
                # Flip del kernel per correlazione (invece di convoluzione)
                grd_flipped = torch.flip(grd_single, dims=[3])

                # Convoluzione 2D
                corr = F.conv2d(sat_single, grd_flipped, padding=0)
                corr_results.append(corr)

            # Concatena i risultati lungo la dimensione dei ground features
            sat_corr = torch.cat(corr_results, dim=1)  # [1, batch_grd, 1, s_w]
            correlations.append(sat_corr)

        # Concatena tutti i risultati satellite
        out = torch.concat(correlations, dim=0)  # [batch_sat, batch_grd, 1, s_w]
       #log.debug(f"out shape before squeeze: {out.shape}")
        # Rimuovi la dimensione dell'altezza (che dovrebbe essere 1)
        out = out.squeeze(3)  # [batch_sat, batch_grd, s_w]
       #log.debug(f"out shape after squeeze: {out.shape}")
        # Trova l'orientamento con correlazione massima
        orien = torch.argmax(out, dim=2)  # [batch_sat, batch_grd]
        #log.debug(f"orien: {orien}")
        #log.debug(f"orien.long: {orien.long()}")
        return out, orien.long()

    def torch_shape(self, x, rank):
       #log.debug(f"shape of x in input: {x.shape}")
        assert len(x.shape) == rank, f"Expected rank {rank}, but got {len(x.shape)}"
        return list(x.shape)

    def crop_sat(self, sat_matrix, orien, grd_width):
       #log.debug(f"Ground width: {grd_width}")
        batch_sat, batch_grd = self.torch_shape(orien,2)
       #log.debug(f"Batch sat size: {batch_sat}")
       #log.debug(f"Batch grd size: {batch_grd}")
        channel, h, w = sat_matrix.shape[1:]
       #log.debug(f"Starting shape of sat_matrix [B,C,H,W]: {sat_matrix.shape}")
        # Espandi sat_matrix per ogni ground feature
        # [batch_sat, channel, h, w] -> [batch_sat, 1, channel, h, w] -> [batch_sat, batch_grd, channel, h, w]
        sat_expanded = sat_matrix.unsqueeze(1).expand(-1, batch_grd, -1, -1, -1)
       #log.debug(f"Expanded shape of sat_matrix [B,BG,C,H,W]: {sat_expanded.shape}")

        # Riorganizza per avere width come prima dimensione spaziale
        # [batch_sat, batch_grd, channel, h, w] -> [batch_sat, batch_grd, channel, w, h]
        sat_transposed = sat_expanded.permute(0, 1, 2, 4, 3)
       #log.debug(f"Permuted shape of sat_matrix [B,BG,C,W,H]: {sat_transposed.shape}")
        # Crea gli indici per il cropping circolare
        batch_sat_idx = torch.arange(batch_sat, device=sat_matrix.device).view(-1, 1, 1)
        batch_grd_idx = torch.arange(batch_grd, device=sat_matrix.device).view(1, -1, 1)
        width_idx = torch.arange(w, device=sat_matrix.device).view(1, 1, -1)

        # Applica l'offset circolare
        orien_expanded = orien.unsqueeze(-1)  # [batch_sat, batch_grd, 1]
        shifted_idx = (width_idx + orien_expanded) % w

        # Gathering avanzato per il cropping
        sat_shifted = torch.zeros_like(sat_transposed)
        for i in range(batch_sat):
            for j in range(batch_grd):
                for k in range(w):
                    shifted_k = shifted_idx[i, j, k].item()
                    sat_shifted[i, j, :, k, :] = sat_transposed[i, j, :, shifted_k, :]

        # Prendi solo le prime grd_width colonne
        sat_cropped = sat_shifted[:, :, :, :grd_width, :]

        # Riorganizza per ottenere [batch_sat, batch_grd, channel, h, grd_width]
        sat_crop_matrix = sat_cropped.permute(0, 1, 2, 4, 3)
       #log.debug(f"sat_crop_matrix shape: {sat_crop_matrix.shape}")
        assert sat_crop_matrix.size()[3] == grd_width, f"Expected {grd_width}, but got {sat_crop_matrix.size()[3]}"

        return sat_crop_matrix

def compute_top1_accuracy(distance_matrix):
    # Distanze: shape (B, B) — ground vs satellite
    gt_indices = torch.arange(distance_matrix.shape[0]).to(distance_matrix.device)
    preds = torch.argmin(distance_matrix, dim=1)
    correct = (preds == gt_indices).sum().item()
    accuracy = correct / distance_matrix.shape[0]
    return accuracy

def compute_triplet_loss(distance, loss_weight=10.0):
    batch_size = distance.shape[0]

    # Estrai la diagonale (distanze positive)
    pos_dist = torch.diag(distance)

    pair_n = batch_size * (batch_size - 1.0)

    # satellite to ground
    triplet_dist_g2s = pos_dist.unsqueeze(1) - distance
    loss_g2s = torch.sum(torch.log(1 + torch.exp(triplet_dist_g2s * loss_weight).clamp(max=100))) / pair_n

    # ground to satellite
    triplet_dist_s2g = pos_dist.unsqueeze(0) - distance
    loss_s2g = torch.sum(torch.log(1 + torch.exp(triplet_dist_s2g * loss_weight).clamp(max=100))) / pair_n

    loss = (loss_g2s + loss_s2g) / 2.0
    return loss

class VGGGroundBranch(nn.Module):
    """
    VGG-based network for ground view (panoramic) images
    """

    def __init__(self):
        super(VGGGroundBranch, self).__init__()
        # Load pretrained VGG16 and extract features
        vgg = vgg16(weights=config["vgg_default_weights"])
        self.features = nn.ModuleList(list(vgg.features.children()))
        # Freeze early layers (equivalent to trainable=False for layers <= 9)
        for i, layer in enumerate(self.features[:config["no_layer_vgg_non_trainable"]+1]):
            for param in layer.parameters():
                param.requires_grad = False
        # Dropout layers for regularization
        self.dropout = nn.Dropout(config["dropout_ratio"])
        # Additional convolutional layers
        self.conv_extra1 = nn.Conv2d(images_params["max_width"], int(images_params["max_width"]/2), kernel_size=3, stride=(2, 1), padding="valid")
        self.conv_extra2 = nn.Conv2d(int(images_params["max_width"]/2), int(images_params["max_width"]/8), kernel_size=3, stride=(2, 1), padding="valid")
        self.conv_extra3 = nn.Conv2d(int(images_params["max_width"]/8), int(images_params["max_width"]/32), kernel_size=3, stride=(1, 1), padding="valid")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # Forward through VGG features
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Add dropout after specific block4 conv layers
            if i in [17, 19, 21]:  # block4_conv1, block4_conv2, block4_conv3
                x = self.dropout(x)
            # Stop before last 6 layers (equivalent to break at len-6)
            if i >= len(self.features) - config["no_layer_vgg_non_trainable"]:
                break
        # Additional convolutional layers
        x = self.warp_pad_columns(x,1)
        x = self.relu(self.conv_extra1(x))
        x = self.warp_pad_columns(x, 1)
        x = self.relu(self.conv_extra2(x))
        x = self.warp_pad_columns(x, 1)
        x = self.relu(self.conv_extra3(x))
        x = x.permute(0, 2, 3, 1)
        return x

    def warp_pad_columns(self,x, n=1):
        # Concatenazione laterale (wrap lungo larghezza)
        out = torch.cat([x[:, :, :, -n:], x, x[:, :, :, :n]], dim=3)
        # Add symmetric padding for height
        out = F.pad(out, (0, 0, n, n), mode='constant', value=0)
        return out

class VGGSatelliteBranch(nn.Module):
    """
    VGG-based network for satellite images with circular convolution
    """

    def __init__(self, name_suffix='_sat'):
        super(VGGSatelliteBranch, self).__init__()
        self.name_suffix = name_suffix
        # Load pretrained VGG16 and extract features
        vgg = vgg16(weights=config["vgg_default_weights"])
        self.features = nn.ModuleList(list(vgg.features.children()))
        # Freeze early layers (equivalent to trainable=False for layers <= 9)
        for i, layer in enumerate(self.features[:config["no_layer_vgg_non_trainable"]+1]):
            for param in layer.parameters():
                param.requires_grad = False
        # Dropout layers for regularization
        self.dropout = nn.Dropout(config["dropout_ratio"])
        # Additional convolutional layers with circular padding
        self.conv_extra1 = nn.Conv2d(images_params["max_width"], int(images_params["max_width"]/2), kernel_size=3, stride=(2, 1), padding=0)
        self.conv_extra2 = nn.Conv2d(int(images_params["max_width"]/2), int(images_params["max_width"]/8), kernel_size=3, stride=(2, 1), padding=0)
        self.conv_extra3 = nn.Conv2d(int(images_params["max_width"]/8), int(images_params["max_width"]/64), kernel_size=3, stride=(1, 1), padding=0)
        self.relu = nn.ReLU(inplace=True)

    def warp_pad_columns(self, x, n=1):
        """
        Circular padding for width dimension to handle panoramic images
        """
        # Concatenate columns circularly
        out = torch.cat([x[:, :, :, -n:], x, x[:, :, :, :n]], dim=3)
        # Add symmetric padding for height
        out = F.pad(out, (0, 0, n, n), mode='constant', value=0)
        return out

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # Forward through VGG features
        for i, layer in enumerate(self.features):
            x = layer(x)

            # Add dropout after specific block4 conv layers
            if i in [17, 19, 21]:  # block4_conv1, block4_conv2, block4_conv3
                #log.debug("Dropping connections...")
                x = self.dropout(x)

            #log.debug(f"At layer {i+1} of type {type(layer).__name__} the x shape is: {x.shape}")

            # Stop before last 6 layers
            if i >= len(self.features) - config["no_layer_vgg_non_trainable"]:
                break

        # Additional convolutional layers with circular padding
        #log.debug(f"x.shape before 1st warp and pad: {x.shape}")
        x = self.warp_pad_columns(x, 1)
        #log.debug(f"x.shape after 1st warp and pad: {x.shape}")
        x = self.relu(self.conv_extra1(x))
        #log.debug(f"x.shape before 2nd warp and pad: {x.shape}")
        x = self.warp_pad_columns(x, 1)
        #log.debug(f"x.shape after 2nd warp and pad: {x.shape}")
        x = self.relu(self.conv_extra2(x))
        #log.debug(f"x.shape before 3rd warp and pad: {x.shape}")
        x = self.warp_pad_columns(x, 1)
        #log.debug(f"x.shape after 3rd warp and pad: {x.shape}")
        x = self.relu(self.conv_extra3(x))
        #log.debug(f"Returning x shape (before permutation): {x.shape}")
        x=x.permute(0,2,3,1)
        #log.debug(f"Returning x shape (after permutation): {x.shape}")
        return x


class GroundToAerialMatchingModel(nn.Module):
    """
    Complete Ground-to-Aerial image matching model with three VGG branches
    """
    def __init__(self):
        super(GroundToAerialMatchingModel, self).__init__()
        # Three parallel VGG branches
        self.ground_branch = VGGGroundBranch()
        self.satellite_branch = VGGSatelliteBranch('_sat')
        self.segmap_branch = VGGSatelliteBranch('_segmap')
        # Feature processor
        self.processor = ProcessFeatures()

    def forward(self, ground_img, polar_sat_img, segmap_img, return_features=False):
        """
        Forward pass through the complete model

        Args:
            ground_img: Ground panoramic images [B, C, H, W]
            polar_sat_img: Polar satellite images [B, C, H, W]
            segmap_img: Segmentation mask images [B, C, H, W]
            return_features: If True, return intermediate features

        Returns:
            If return_features=True: (grd_features, sat_features, segmap_features)
            Else: (sat_matrix, grd_matrix, distance, pred_orien)
        """
        # Extract features from each branch
        grd_features = self.ground_branch(ground_img)
        sat_features = self.satellite_branch(polar_sat_img)
        segmap_features = self.segmap_branch(segmap_img)

        if return_features:
            return grd_features, sat_features, segmap_features

        # L2 normalize ground features
        #norm = torch.norm(grd_features, p=2, dim=[1, 2, 3], keepdim=True)
        #grd_features = grd_features / (norm + 1e-8)

        # Concatenate satellite and segmentation features
        sat_combined = torch.concat([sat_features, segmap_features], dim=3)

        # Process features to compute correlation and distance
        sat_matrix, grd_matrix, distance, pred_orien = self.processor.VGG_13_conv_v2_cir(
            sat_combined, grd_features
        )

        return sat_matrix, grd_matrix, distance, pred_orien

    def get_feature_dimensions(self, input_shapes):
        """
        Helper method to get output feature dimensions
        """
        ground_shape, polar_sat_shape, segmap_shape = input_shapes

        with torch.no_grad():
            dummy_ground = torch.zeros(1, *ground_shape[1:])
            dummy_polar_sat = torch.zeros(1, *polar_sat_shape[1:])
            dummy_segmap = torch.zeros(1, *segmap_shape[1:])

            grd_features, sat_features, segmap_features = self.forward(
                dummy_ground, dummy_polar_sat, dummy_segmap, return_features=True
            )

            return {
                'ground_features_shape': grd_features.shape,
                'sat_features_shape': sat_features.shape,
                'segmap_features_shape': segmap_features.shape
            }