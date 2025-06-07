import torch.nn as nn
import torch.nn.functional as F
import torch

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
        sat_cropped = self.crop_sat(sat_vgg, corr_orien, grd_vgg.get_shape().as_list()[2])
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
        x = self.warp_pad_columns(sat_matrix, n)

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
        out = torch.cat(correlations, dim=0)  # [batch_sat, batch_grd, 1, s_w]

        # Rimuovi la dimensione dell'altezza (che dovrebbe essere 1)
        out = out.squeeze(2)  # [batch_sat, batch_grd, s_w]

        # Trova l'orientamento con correlazione massima
        orien = torch.argmax(out, dim=2)  # [batch_sat, batch_grd]

        return out, orien.long()



    def crop_sat(self, sat_matrix, orien, grd_width):
        batch_sat, batch_grd = orien.shape
        channel, h, w = sat_matrix.shape[1:]

        # Espandi sat_matrix per ogni ground feature
        # [batch_sat, channel, h, w] -> [batch_sat, 1, channel, h, w] -> [batch_sat, batch_grd, channel, h, w]
        sat_expanded = sat_matrix.unsqueeze(1).expand(-1, batch_grd, -1, -1, -1)

        # Riorganizza per avere width come prima dimensione spaziale
        # [batch_sat, batch_grd, channel, h, w] -> [batch_sat, batch_grd, channel, w, h]
        sat_transposed = sat_expanded.permute(0, 1, 2, 4, 3)

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

        assert sat_crop_matrix.shape[4] == grd_width

        return sat_crop_matrix

    def torch_shape(self, x, rank):
        return list(x.shape)