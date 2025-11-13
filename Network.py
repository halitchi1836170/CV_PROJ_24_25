import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from Globals import *
import torch.nn.functional as F
from logger import log
import math
from torch.nn.modules.utils import _pair
from typing import Optional, Tuple


##############################################################################################################################################
# ============================================ OLD GradCAM (temporarily disabled) ============================================================
# class GradCAM:
#     def __init__(self, model, target_layer_name):
#         self.model = model
#         self.target_layer = dict([*model.named_modules()])[target_layer_name]
#         self.activations = None
#         self.gradients = None

#         self.target_layer.register_forward_hook(self.forward_hook)
#         self.target_layer.register_full_backward_hook(self.backward_hook)

#     def forward_hook(self, module, input, output):
#         self.activations = output

#     def backward_hook(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0]

#     def generate(self, target_size):
#         w = self.gradients.mean(dim=(2,3), keepdim=True)             # [B,C,1,1]
#         cam = (w * self.activations).sum(dim=1, keepdim=True).relu() # [B,1,H,W]
#         cam = F.interpolate(cam, size=target_size, mode='bilinear', align_corners=False)
#         cam = (cam - cam.amin(dim=(2,3), keepdim=True)) / (cam.amax(dim=(2,3), keepdim=True) - cam.amin(dim=(2,3), keepdim=True) + 1e-12)
#         return cam
##############################################################################################################################################
##############################################################################################################################################

class HookManager:
    def __init__(self, model: torch.nn.Module, layer_name: str):
        modules = dict(model.named_modules())
        if layer_name not in modules:
            raise KeyError(f"Layer name '{layer_name}' not found in model.named_modules()")
        #log.info(f"HookerManager using {layer_name}.")
        self.module = modules[layer_name]
        self.activations = None
        self.gradients = None
        self._fwd_handle = self.module.register_forward_hook(self._forward_hook)
        self._grad_handle = None

    def _forward_hook(self, module, inp, out):
        self.activations = out
        self.gradients=None
        
        if out.requires_grad:
            def _save_grad(grad):
                self.gradients = grad
            if self._grad_handle is not None:
                try: self._grad_handle.remove()
                except Exception: pass
            self._grad_handle = out.register_hook(_save_grad)

    def remove(self):
        try:
            self._fwd_handle.remove()
        except Exception:
            pass
        if self._grad_handle is not None:
            try: self._grad_handle.remove()
            except Exception: pass

def compute_gradcam_from_acts_grads(
        acts: torch.Tensor, grads: torch.Tensor,
        upsample_to: Optional[Tuple[int,int]] = None,
        normalize: bool = True) -> torch.Tensor:
    """Compute Grad-CAM from activations and gradients."""
    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    if normalize:
        cam_min, cam_max = cam.amin(dim=(2,3), keepdim=True), cam.amax(dim=(2,3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-12)
    if upsample_to:
        cam = F.interpolate(cam, size=upsample_to, mode='bilinear', align_corners=False)
    return cam.squeeze(1)

def gradcam_from_activations(acts: torch.Tensor, target_scalar: torch.Tensor, upsample_to_hw=None, normalize=True):
    # Gradiente dL/dA_k
    grads = torch.autograd.grad(target_scalar, acts, retain_graph=True, create_graph=True)[0]   # [B,C,H,W]

    # Pesi alpha_k = media spaziale dei gradienti
    weights = grads.mean(dim=(2,3), keepdim=True)                                              # [B,C,1,1]

    # CAM = ReLU( sum_k alpha_k * A_k )
    cam = (weights * acts).sum(dim=1, keepdim=True)                                            # [B,1,H,W]
    cam = F.relu(cam)

    # Normalizzazione per-campione
    if normalize:
        cam = cam / (cam.amax(dim=(2,3), keepdim=True) + 1e-12)

    # Upsample opzionale
    if upsample_to_hw is not None:
        cam = F.interpolate(cam, size=upsample_to_hw, mode="bilinear", align_corners=False)    # [B,1,H0,W0]

    return cam.squeeze(1) 

def compute_saliency_loss(gradcam, inputs, cam_size):
    cam = gradcam.generate(target_size=cam_size)  # tensor con grad
    return -cam.mean()
    cam = gradcam.generate(target_size=cam_size)
    cam_tensor = torch.from_numpy(cam).to(inputs.device)
    return -torch.mean(cam_tensor)

def saliency_variability_loss(cams:torch.Tensor, eps=1e-12):
    B,H,W=cams.shape
    W=cams.std(dim=0,unbiased=False)
    W=(W - W.min()) / (W.max() - W.min() + eps)
    cams_flat = cams.view(B, -1)                          # [B,HW]
    W_flat = W.view(-1)                                   # [HW]
    overlap = (cams_flat * W_flat).sum(dim=1) / (cams_flat.sum(dim=1) + eps)   # [B]
    score = overlap.mean()
    return 1.0 - score

class ProcessFeatures(nn.Module):
    """
    Feature processing module for computing correlation and distance matrices
    """

    def __init__(self):
        super(ProcessFeatures, self).__init__()

    def VGG_13_conv_v2_cir(self, sat_features, grd_features):
        #log.debug("Starting VGG_13_conv_v2_cir method...")
        # Ensure input shapes are correct
        #log.debug(f"sat_features shape: {sat_features.shape}")
        #log.debug(f"grd_features shape: {grd_features.shape}")
        sat_matrix, distance, pred_orien = self.corr_crop_distance(sat_features, grd_features)
        return sat_features, grd_features, distance, pred_orien

    def corr_crop_distance(self, sat_vgg, grd_vgg):
        corr_out, corr_orien = self.corr(sat_vgg, grd_vgg)
        #log.debug("Output of corr method:")
        #log.debug(f"corr_orien values: {corr_orien[0,:5]}")
        sat_cropped = self.crop_sat(sat_vgg, corr_orien, grd_vgg.size()[2])
        #log.debug(f"sat_cropped values: {sat_cropped[0,0,0,0,:5]}")
        #log.debug(f"sat_cropped shape: {sat_cropped.shape} after cropping in corr_cropdistance method")
        # shape = [batch_sat, batch_grd, h, grd_width, channel]

        norm = torch.norm(sat_cropped, p=2, dim=[2,3,4], keepdim=True)
        sat_matrix = sat_cropped / norm.clamp_min(1e-12)

        grd_expanded = grd_vgg.unsqueeze(0) 
        dot_product = torch.sum(sat_matrix * grd_expanded, dim=[2, 3, 4])
        distance = 2.0 - 2.0 * dot_product.t()
        # shape = [batch_grd, batch_sat]
        #log.debug(f"distance shape: {distance.shape} after computing distance in corr_crop_distance method")

        return sat_matrix, distance, corr_orien

    def warp_pad_columns(self, x, n):
        """Padding circolare per le colonne"""
        padded = torch.cat([x, x[:, :, :, :n]], dim=3)
        return padded

    def corr(self, sat_matrix, grd_matrix):
        batch_sat, s_h, s_w, s_c = sat_matrix.shape
        batch_grd, g_h, g_w, g_c = grd_matrix.shape

        assert (s_h == g_h) and (s_c == g_c)  # devono avere la stessa altezza e lo stesso numero di canali

       #log.debug("Starting correlation computation...")
       #log.debug(f"sat_matrix shape: {sat_matrix.shape}")
       #log.debug(f"grd_matrix shape: {grd_matrix.shape}")

        n = g_w - 1

        #x = inner_warp_pad_columns(sat_matrix, n)
        #x = self.warp_pad_columns(sat_matrix, n)
        
        sat_padded = torch.concat([sat_matrix, sat_matrix[:,:, :n,:]], dim=2)
       #log.debug(f"Padded satellite matrix dimension: {sat_padded.shape}")
        
        x_nchw = sat_padded.permute(0,3,1,2).contiguous()
       #log.debug(f"x_nchw input dimension: {x_nchw.shape}")
        
        w_oihw = grd_matrix.permute(0,3,1,2).contiguous()
       #log.debug(f"w_oihw input dimension: {w_oihw.shape}")
        
        #filters = grd_matrix
        #filters = filters.to(sat_matrix)

        #out = torch.conv2d(sat_padded,filters,groups=1)
        #out = nn.Conv2d(sat_padded,filters,stride=(1,1,1,1),padding='valid')
        out = F.conv2d(x_nchw,w_oihw,bias=None,stride=1,padding=0)
       #log.debug(f"torch.nn.functional.conv2d returned object 'out' dimensions before squeeze and permutation: {out.shape}")
        
        #out = out.squeeze(3).permute(0,2,1)     # [B_sat, W, B_grd]  (come TF)
        out = out.squeeze(2).permute(0,2,1)
       #log.debug(f"torch.nn.functional.conv2d out dimensions after squeeze and permutation: {out.shape}")
        orien = out.argmax(dim=1)               # [B_sat, B_grd]
       #log.debug(f"torch.nn.functional.conv2d orien dimensions after argmax on axis 1 of out: {orien.shape}")
        return out, orien

        correlations = []
        for i in range(batch_sat):
            sat_single = x[i:i + 1]  # [1, channel, height, width_padded]

            # Correlazione con tutti i ground features
            corr_results = []
            for j in range(batch_grd):
                grd_single = grd_matrix[j:j + 1]  # [1, channel, height, width]
                ##log.debug(f"Processing satellite {i+1}/{batch_sat}, ground {j+1}/{batch_grd} with sat shapes: {sat_single.shape} and ground shapes: {grd_single.shape}")
                # Flip del kernel per correlazione (invece di convoluzione)
                grd_flipped = torch.flip(grd_single, dims=[3])

                # Convoluzione 2D
                corr = F.conv2d(sat_single, grd_flipped, padding="valid")
                corr_results.append(corr)

            # Concatena i risultati lungo la dimensione dei ground features
            sat_corr = torch.cat(corr_results, dim=1)  # [1, batch_grd, s_w, 1]
            correlations.append(sat_corr)

        # Concatena tutti i risultati satellite
        out = torch.concat(correlations, dim=0)  # [batch_sat, batch_grd, s_w, 1]
        ##log.debug(f"out shape before squeeze: {out.shape}")
        # Rimuovi la dimensione dell'altezza (che dovrebbe essere 1)
        out = out.squeeze(3)  # [batch_sat, batch_grd, s_w]
        ##log.debug(f"out shape after squeeze: {out.shape}")
        # Trova l'orientamento con correlazione massima
        orien = torch.argmax(out, dim=2)  # [batch_sat, batch_grd]
        ##log.debug(f"orien shape: {orien.shape} after argmax")
        return out, orien.to(torch.int32)

    def torch_shape(self, x, rank):
        ##log.debug(f"Shape of x in input: {x.shape}")
        assert len(x.shape) == rank, f"Expected rank {rank}, but got {len(x.shape)}"
        shape = list(x.size())[:rank]
        return tuple(shape)

    def crop_sat(self, sat_matrix, orien, grd_width):
        #log.debug(f"Ground width: {grd_width}")
        batch_sat, batch_grd = self.torch_shape(orien,2)
        #log.debug(f"Batch sat size: {batch_sat}")
        #log.debug(f"Batch grd size: {batch_grd}")
        B,h,w,channel = sat_matrix.shape
        assert B == batch_sat, "orient.shape[0] != sat_matrix.shape[0]"
        #log.debug(f"Starting shape of sat_matrix [B,H,W,C]: {sat_matrix.shape}")
        device = sat_matrix.device
        
        sat = sat_matrix.unsqueeze(1).expand(-1,batch_grd,-1,-1,-1)                             # [B_sat,B_grd,H,W,C]
        #log.debug(f"sat_matrix shape after unsqueeze and expand operations: {sat.shape} ")
        sat = sat.permute(0,1,3,2,4).contiguous()                                               # [B_sat,B_grd,W,H,C]
        #log.debug(f"sat_matrix shape after permute too: {sat.shape} ")
        
        orien = (orien.to(device=device, dtype=torch.long)% w).unsqueeze(-1)
        
        i = torch.arange(batch_sat,device=device)
        j = torch.arange(batch_grd,device=device)
        k = torch.arange(w,device=device)
        
        x, y, z = torch.meshgrid(i, j, k, indexing='ij')
        z_index = (z + orien) % w 
        
        x1 = x.reshape(-1)
        y1 = y.reshape(-1)
        z1 = z_index.reshape(-1)

        sat_gather = sat[x1,y1,z1]
        sat = sat_gather.view(batch_sat,batch_grd,w,h,channel)
        
        index1 = torch.arange(grd_width,device=device)
        
        sat_crop_matrix = sat.permute(2,0,1,3,4)[index1].permute(1,2,3,0,4).contiguous()
        
        assert sat_crop_matrix.size()[3] == grd_width
        return sat_crop_matrix 
                        
        k   = torch.arange(w, device=device)                                    # [W]
        #log.debug(f"k[None, None, :, None, None] dimensions: {k[None, None, :, None, None].shape}")
        orien_exp = orien.to(device=device,dtype=torch.long).view(B,batch_grd,1,1,1)
        #log.debug(f"orien_exp dimensions: {orien_exp.shape}")
        idx = (k[None, None, :, None, None] + orien_exp) % w       # [B_sat,B_grd,W,1,1]
        idx = idx.long()  
        
        sat_shifted = sat.gather(2, idx.expand(-1, -1, -1, h, channel))   
        
        sat_crop = sat_shifted[:, :, :grd_width]                    # crop width
        #log.debug(f"Returning permuted cropped sat_crop matrix of shape: {sat_crop.permute(0,1,3,2,4).contiguous().shape}")
        return sat_crop.permute(0,1,3,2,4).contiguous()                          # [B_sat,B_grd,H,grd_W,C]
        
        ##log.debug(f"Starting shape of sat_matrix [B,H,W,C]: {sat_matrix.shape}")
        # Espandi sat_matrix per ogni ground feature
        sat_expanded = sat_matrix.unsqueeze(1).expand(-1, batch_grd, -1, -1, -1)
        ##log.debug(f"Expanded shape of sat_matrix [B,BG,H,W,C]: {sat_expanded.shape}")
        sat_expanded = sat_expanded.permute(0, 1, 3, 2, 4) 
        ##log.debug(f"Permuted shape of sat_matrix [B,BG,W,H,C]: {sat_expanded.shape}")
        orien = orien.unsqueeze(-1) # [batch_sat, batch_grd, 1]    
        ##log.debug(f"orien shape after unsqueeze: {orien.shape}")
            
        i = torch.arange(batch_sat, device=sat_matrix.device)
        j = torch.arange(batch_grd, device=sat_matrix.device)
        k = torch.arange(w, device=sat_matrix.device)    
            
        x,y,z = torch.meshgrid(i, j, k, indexing='ij')
        z_index = ((z + orien) % w).long()  # Indici circolari
        
        sat_shifted = torch.zeros_like(sat_expanded)
        for bi in range(batch_sat):
            for bj in range(batch_grd):
                sat_shifted[bi, bj] = sat_expanded[bi, bj][z_index[bi, bj, :]]
        
        sat_crop = sat_shifted[:, :, :grd_width, :, :]
        ##log.debug(f"Shape of sat_crop before cropping and permutaion: {sat_crop.shape}")
        sat_crop_matrix = sat_crop.permute(0, 1, 3, 2, 4)  
        ##log.debug(f"sat_crop_matrix shape after cropping and permutaion: {sat_crop_matrix.shape}")
        
        assert sat_crop_matrix.size()[3] == grd_width, f"Expected {grd_width}, but got {sat_crop_matrix.size()[3]}"
        ##log.debug(f"Final shape: {sat_crop_matrix.shape}")
        return sat_crop_matrix
            

def compute_top1_accuracy(distance_matrix):
    # Distanze: shape (B, B) — ground vs satellite
    gt_indices = torch.arange(distance_matrix.shape[0]).to(distance_matrix.device)
    preds = torch.argmin(distance_matrix, dim=1)
    correct = (preds == gt_indices).sum().item()
    accuracy = correct / distance_matrix.shape[0]
    return accuracy

def compute_triplet_loss(distance, batch_size, loss_weight):
    # Estrai la diagonale (distanze positive)
    pos_dist = torch.diag(distance)

    pair_n = batch_size * (batch_size - 1.0)

    # satellite to ground
    triplet_dist_g2s = pos_dist.unsqueeze(1) - distance
    #triplet_dist_g2s = pos_dist - distance
    loss_g2s = torch.sum(torch.log(1 + torch.exp(triplet_dist_g2s * loss_weight))) / pair_n

    # ground to satellite
    triplet_dist_s2g = pos_dist.unsqueeze(0) - distance
    #triplet_dist_s2g = pos_dist.unsqueeze(1) - distance
    loss_s2g = torch.sum(torch.log(1 + torch.exp(triplet_dist_s2g * loss_weight))) / pair_n

    loss = (loss_g2s + loss_s2g) / 2.0
    return loss

class Conv2dSameTF(nn.Module):
    """
    Conv2d che replica il padding='SAME' di TensorFlow (NHWC) in PyTorch (NCHW),
    anche quando stride > 1. Esegue padding dinamico e poi conv con padding=0.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=0,                 # niente padding fisso
                              dilation=self.dilation,
                              groups=self.groups,
                              bias=bias)

    def forward(self, x):
        ih, iw = x.shape[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation

        # kernel "effettivo" con la dilatazione
        eff_kh = (kh - 1) * dh + 1
        eff_kw = (kw - 1) * dw + 1

        oh = math.ceil(ih / sh)
        ow = math.ceil(iw / sw)

        pad_h = max((oh - 1) * sh + eff_kh - ih, 0)
        pad_w = max((ow - 1) * sw + eff_kw - iw, 0)

        # TensorFlow distribuisce: top=floor, bottom=ceil (idem a sinistra/destra)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h or pad_w:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)

class VGGGroundBranch(nn.Module):
    """
    VGG-based network for ground view (panoramic) images
    """

    def __init__(self):
        #log.debug("Initializing VGGGroundBranch...")
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
        # self.conv_extra1 = nn.Conv2d(images_params["max_width"], int(images_params["max_width"]/2), kernel_size=3, stride=(2, 1), padding="same")
        # self.conv_extra2 = nn.Conv2d(int(images_params["max_width"]/2), int(images_params["max_width"]/8), kernel_size=3, stride=(2, 1), padding="same")
        # self.conv_extra3 = nn.Conv2d(int(images_params["max_width"]/8), int(images_params["max_width"]/32), kernel_size=3, stride=(1, 1), padding="same")
        
        self.conv_extra1 = Conv2dSameTF(images_params["max_width"], int(images_params["max_width"]/2), kernel_size=3, stride=(2, 1),bias=True)
        self.conv_extra2 = Conv2dSameTF(int(images_params["max_width"]/2), int(images_params["max_width"]/8), kernel_size=3, stride=(2, 1),bias=True)
        self.conv_extra3 = Conv2dSameTF(int(images_params["max_width"]/8), int(images_params["max_width"]/32), kernel_size=3, stride=(1, 1),bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        # self.target_layer_name = gradcam_config["target_layer"]
        # self.target_acts=None
        # target_layer = dict([*self.named_modules()])[self.target_layer_name]     
        # def _save_acts(module,inp,out):
        #     self.target_acts=out
        #     if out.requires_grad:
        #         out.retain_grad()
        # target_layer.register_forward_hook(_save_acts)

    def forward(self, x):
        #log.debug("Forward pass through VGGGroundBranch...")
        #log.debug(f"Input shape: {x.shape}")
        x = x.permute(0, 3, 1, 2)
        #log.debug(f"Permuted input shape: {x.shape}")
        # Forward through VGG features
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Add dropout after specific block4 conv layers
            if i in [17, 19, 21]:  # block4_conv1, block4_conv2, block4_conv3
                x = self.dropout(x)
            # Stop before last 6 layers (equivalent to break at len-6)
            if i >= len(self.features) - config["no_layer_vgg_non_trainable"]:
                break
        #log.debug(f"Shape after VGG features: {x.shape}")
        # Additional convolutional layers
        #x = self.warp_pad_columns(x,1)
        x = self.relu(self.conv_extra1(x))
        #x = self.warp_pad_columns(x, 1)
        x = self.relu(self.conv_extra2(x))
        #x = self.warp_pad_columns(x, 1)
        x = self.relu(self.conv_extra3(x))
        #log.debug(f"Shape after extra conv layers: {x.shape}")
        x = x.permute(0, 2, 3, 1)
        #log.debug(f"Final output shape: {x.shape}")
        return x

    # def warp_pad_columns(self,x, n=1):
    #     # Concatenazione laterale (wrap lungo larghezza)
    #     out = torch.cat([x[:, :, :, -n:], x, x[:, :, :, :n]], dim=3)
    #     # Add symmetric padding for height
    #     out = F.pad(out, (0, 0, n, n), mode='constant', value=0)
    #     return out

class VGGSatelliteBranch(nn.Module):
    """
    VGG-based network for satellite images with circular convolution
    """

    def __init__(self, name_suffix='_sat'):
        #log.debug(f"Initializing VGGSatelliteBranch with suffix {name_suffix}...")
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
        self.conv_extra1 = nn.Conv2d(images_params["max_width"], int(images_params["max_width"]/2), kernel_size=3, stride=(2, 1), padding="valid")
        self.conv_extra2 = nn.Conv2d(int(images_params["max_width"]/2), int(images_params["max_width"]/8), kernel_size=3, stride=(2, 1), padding="valid")
        self.conv_extra3 = nn.Conv2d(int(images_params["max_width"]/8), int(images_params["max_width"]/64), kernel_size=3, stride=(1, 1), padding="valid")
        self.relu = nn.ReLU(inplace=True)
        
        # self.target_layer_name = gradcam_config["target_layer"]
        # self.target_acts=None
        # target_layer = dict([*self.named_modules()])[self.target_layer_name]
        # def _save_acts(module,inp,out):
        #     self.target_acts=out
        #     if out.requires_grad:
        #         out.retain_grad()
        # target_layer.register_forward_hook(_save_acts)

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
        #log.debug(f"Forward pass through VGGSatelliteBranch with suffix {self.name_suffix}...")
        #log.debug(f"Input shape: {x.shape}")
        x = x.permute(0, 3, 1, 2)
        #log.debug(f"Permuted input shape: {x.shape}")
        # Forward through VGG features
        for i, layer in enumerate(self.features):
            x = layer(x)

            # Add dropout after specific block4 conv layers
            if i in [17, 19, 21]:  # block4_conv1, block4_conv2, block4_conv3
                ###log.debug("Dropping connections...")
                x = self.dropout(x)

           ##log.debug(f"At layer {i+1} of type {type(layer).__name__} the x shape is: {x.shape}")

            # Stop before last 6 layers
            if i >= len(self.features) - config["no_layer_vgg_non_trainable"]:
                break
        #log.debug(f"Shape after VGG features: {x.shape}")
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
        #log.debug("Initializing GroundToAerialMatchingModel...")
        super(GroundToAerialMatchingModel, self).__init__()
        # Three parallel VGG branches
        self.ground_branch = VGGGroundBranch()
        self.satellite_branch = VGGSatelliteBranch('_sat')
        self.segmap_branch = VGGSatelliteBranch('_segmap')
        # Feature processor
        self.processor = ProcessFeatures()

    def forward(self, ground_img, polar_sat_img, segmap_img):
        # Extract features from each branch
        #log.debug("Forward pass through GroundToAerialMatchingModel...")
        #log.debug(f"Ground image shape: {ground_img.shape}")
        grd_features = self.ground_branch(ground_img)
        #log.debug(f"Ground features values : {grd_features[0,0,0,:5]}")
        #log.debug(f"Satellite image shape: {polar_sat_img.shape}")
        sat_features = self.satellite_branch(polar_sat_img)
        #log.debug(f"Satellite features values: {sat_features[0,0,0,:5]}")
        #log.debug(f"Segmentation map shape: {segmap_img.shape}")
        segmap_features = self.segmap_branch(segmap_img)
        #log.debug(f"Segmentation map features values: {segmap_features[0,0,0,:5]}")

        # L2 normalize ground features
        norm = torch.norm(grd_features, p=2, dim=[1, 2, 3], keepdim=True)
        grd_features = grd_features / norm.clamp_min(1e-12)

        # Concatenate satellite and segmentation features
        sat_combined = torch.concat([sat_features, segmap_features], dim=3)

        # Process features to compute correlation and distance
        #log.debug("Processing features to compute correlation and distance...")
        #Valori in input a VGG13
        #log.debug(f"sat_combined values: {sat_combined[0,0,0,:5]}")
        #log.debug(f"grd_features values: {grd_features[0,0,0,:5]}")
        sat_matrix, grd_matrix, distance, pred_orien = self.processor.VGG_13_conv_v2_cir(
            sat_combined, grd_features
        )
        #log.debug("Output values after processing features:")
        #log.debug(f"sat_matrix values: {sat_matrix[0,0,0,:5]}")
        #log.debug(f"grd_matrix values: {grd_matrix[0,0,0,:5]}")
        
        #log.debug(f"Final pred sat_matrix shape: {sat_matrix.shape}")
        #log.debug(f"Final pred grd_matrix shape: {grd_matrix.shape}")
        #log.debug(f"Final pred distance shape: {distance.shape}")
        #log.debug(f"Final pred pred_orien shape: {pred_orien.shape}")
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