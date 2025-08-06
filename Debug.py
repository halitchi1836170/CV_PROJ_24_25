import torch
import numpy as np
from Network import GroundToAerialMatchingModel
from Data import InputData
from Globals import config, folders_and_files
from logger import log
import logging
import os
from Train import validate_original

logs_folder = os.path.join(folders_and_files["log_folder"], "DEBUG")
os.makedirs(logs_folder, exist_ok=True)
streamFile = logging.FileHandler(filename=f"{logs_folder}/{folders_and_files['log_file']}", mode="w", encoding="utf-8")
streamFile.setLevel(logging.DEBUG)
log.addHandler(streamFile)

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
        sat_matrix, grd_matrix, distance, pred_orien = model(batch_grd, batch_sat_polar, batch_segmap)
        log.info(f"sat_matrix shape: {sat_matrix.shape}")
        log.info(f"grd_matrix shape: {grd_matrix.shape}")
        log.info(f"distance matrix shape: {distance.shape}")
        log.info(f"predicted orientation shape: {pred_orien.shape}")
    
    s_height, s_width, s_channel = list(sat_matrix.size())[1:]
    g_height, g_width, g_channel = list(grd_matrix.size())[1:]
    
    sat_batch_matrix = np.zeros([5, s_height, s_width, s_channel])
    grd_batch_matrix = np.zeros([5, g_height, g_width, g_channel])
    orientation_batch_gth = np.zeros([5])
    
    log.debug(f"sat_batch_matrix zeros shape: {sat_batch_matrix.shape}")
    log.debug(f"grd_batch_matrix zeros shape: {grd_batch_matrix.shape}")
    log.debug(f"orientation_batch_gth zeros shape: {orientation_batch_gth.shape}")
    
    #Popolo matrice sat_batch_matrix per valutazione ad ogni batch
    sat_batch_matrix[0:sat_matrix.shape[0],:]=sat_matrix.cpu().detach().numpy()
    grd_batch_matrix[0:grd_matrix.shape[0],:]=grd_matrix.cpu().detach().numpy()
    orientation_batch_gth[0:grd_matrix.shape[0]]=batch_orien
    
    log.debug(f"sat_batch_matrix shape after evaluation: {sat_batch_matrix.shape}")
    log.debug(f"grd_batch_matrix shape after evaluation: {grd_batch_matrix.shape}")
    log.debug(f"orientation_batch_gth shape after evaluation: {orientation_batch_gth.shape}")
    
    sat_batch_descriptor = np.reshape(sat_batch_matrix[:,:,:g_width,:],[-1,g_height*g_width*g_channel])
    norm = np.linalg.norm(sat_batch_descriptor, axis=-1, keepdims=True)
    sat_batch_descriptor = sat_batch_descriptor / np.maximum(norm,1e-12)
    log.debug(f"sat_batch_descriptor shape: {sat_batch_descriptor.shape}")
    
    grd_batch_descriptor = np.reshape(grd_batch_matrix,[-1,g_height*g_width*g_channel])
    log.debug(f"grd_batch_descriptor shape: {grd_batch_descriptor.shape}")
    
    data_batch_amount = grd_batch_descriptor.shape[0]
    log.debug(f"data_batch_amount: {data_batch_amount}")
    top1_percent_batch_value = int(data_batch_amount*0.01)+1
    log.debug(f"1% samples of the data batch: {top1_percent_batch_value}")
    
    dist_array = 2-2*np.matmul(grd_batch_descriptor,np.transpose(sat_batch_descriptor))
    log.debug(f"dist_array shape: {dist_array.shape}")
    print(dist_array)
    
    log.debug(f"distance matrix (returned by model shape: {distance.shape})")
    print(distance)
    
    
    val_batch_accuracy = validate_original(dist_array,1)*100 
    r5_batch = validate_original(dist_array,5)*100
    r10_batch = validate_original(dist_array,10)*100
    r1p_batch = validate_original(dist_array,top1_percent_batch_value)*100
        
if __name__ == "__main__":
    debug_model()