import torch
import numpy as np
from Network import GroundToAerialMatchingModel
from Data import InputData
from Globals import config, images_params
from logger import log
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#Disabilitato causa incompatibilità con le GPU 
torch.backends.cudnn.enabled = False

def debug_model():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Instantiate data loader
    data_loader = InputData()
    batch_sat_polar, batch_sat, batch_grd, batch_segmap, batch_orien = data_loader.next_pair_batch(
        batch_size=5,
        grd_noise=config["train_grd_noise"],
        FOV=config["train_grd_FOV"]
    )
    
    assert batch_sat is not None, "No data loaded!"
    
    # Convert to tensors
    batch_grd = torch.from_numpy(batch_grd).float().to(device)
    batch_sat_polar = torch.from_numpy(batch_sat_polar).float().to(device)
    batch_segmap = torch.from_numpy(batch_segmap).float().to(device)
    
     # Instantiate model
    model = GroundToAerialMatchingModel().to(device)
    model.eval()
    
    # Forward with shape logging
    log.info("Passing through the model...")
    with torch.no_grad():
        grd_feat, sat_feat, seg_feat = model(batch_grd, batch_sat_polar, batch_segmap, return_features=True)

        log.info(f"Ground features shape: {grd_feat.shape}")
        log.info(f"Satellite features shape: {sat_feat.shape}")
        log.info(f"Segmentation features shape: {seg_feat.shape}")

        sat_combined = torch.cat([sat_feat, seg_feat], dim=3)
        log.info(f"Concatenated satellite + segmentation shape: {sat_combined.shape}")

        sat_matrix, grd_matrix, distance, pred_orien = model(batch_grd, batch_sat_polar, batch_segmap)
        log.info(f"sat_matrix shape: {sat_matrix.shape}")
        log.info(f"grd_matrix shape: {grd_matrix.shape}")
        log.info(f"distance matrix shape: {distance.shape}")
        log.info(f"predicted orientation shape: {pred_orien.shape}")
        
if __name__ == "__main__":
    debug_model()