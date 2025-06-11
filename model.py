import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from config import *
import torch.nn.functional as F
from features_manager import ProcessFeatures
from logger import log

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
                log.debug("Dropping connections...")
                x = self.dropout(x)

            log.debug(f"At layer {i+1} x shape: {x.shape}")

            # Stop before last 6 layers
            if i >= len(self.features) - config["no_layer_vgg_non_trainable"]:
                break

        # Additional convolutional layers with circular padding
        log.debug(f"x.shape before 1st warp and pad: {x.shape}")
        x = self.warp_pad_columns(x, 1)
        log.debug(f"x.shape after 1st warp and pad: {x.shape}")
        x = self.relu(self.conv_extra1(x))
        log.debug(f"x.shape before 2nd warp and pad: {x.shape}")
        x = self.warp_pad_columns(x, 1)
        log.debug(f"x.shape after 2nd warp and pad: {x.shape}")
        x = self.relu(self.conv_extra2(x))
        log.debug(f"x.shape before 3rd warp and pad: {x.shape}")
        x = self.warp_pad_columns(x, 1)
        log.debug(f"x.shape after 3rd warp and pad: {x.shape}")
        x = self.relu(self.conv_extra3(x))
        log.debug(f"Returning x shape (before permutation): {x.shape}")
        x=x.permute(0,2,3,1)
        log.debug(f"Returning x shape (after permutation): {x.shape}")
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
        norm = torch.norm(grd_features, p=2, dim=[1, 2, 3], keepdim=True)
        grd_features = grd_features / (norm + 1e-8)

        # Concatenate satellite and segmentation features
        sat_combined = torch.cat([sat_features, segmap_features], dim=1)

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