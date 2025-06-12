import torch.nn as nn
import torch.nn.functional as F
import torch
from logger import log

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
