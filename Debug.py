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
    
    BATCH_SIZE=32
    np.random.seed(17)
    torch.manual_seed(17)
    torch.cuda.manual_seed(17)
    
    # Instantiate data loader
    data_loader = InputData()
    batch_sat_polar, batch_sat, batch_grd, batch_segmap, batch_orien = data_loader.next_pair_batch(
        batch_size=BATCH_SIZE,
        grd_noise=config["train_grd_noise"],
        FOV=config["train_grd_FOV"]
    )
    
    log.info("=== DATA ===")
    log.info(f"batch_sat_polar shape: {batch_sat_polar.shape}")
    log.info(f"batch_sat shape       : {batch_sat.shape}")
    log.info(f"batch_grd shape       : {batch_grd.shape}")
    log.info(f"batch_segmap shape    : {batch_segmap.shape}")
    log.info(f"batch_orien shape     : {batch_orien.shape}")
    
    # Stampa alcuni valori di esempio
    log.info(f"batch_sat_polar[0] : {batch_sat_polar[0, 0, 0, :5]}")
    log.info(f"batch_sat[0]       : {batch_sat[0, 0, 0, :5]}")
    log.info(f"batch_grd[0]       : {batch_grd[0, 0, 0, :5]}")
    log.info(f"batch_segmap[0]    : {batch_segmap[0, 0, 0, :5]}")
    log.info(f"batch_orien[0]     : {batch_orien[0]}")
    
    assert batch_sat is not None, "No data loaded!"
    
    # Convert to tensors
    batch_grd = torch.from_numpy(batch_grd).float().to(device)
    batch_sat_polar = torch.from_numpy(batch_sat_polar).float().to(device)
    batch_segmap = torch.from_numpy(batch_segmap).float().to(device)
    
     # Instantiate model
    model = GroundToAerialMatchingModel().to(device)
    model.eval()
    
    # Forward with shape logging
    log.info("=== FORWARD ===")
    with torch.no_grad():
        sat_matrix, grd_matrix, distance, pred_orien = model(batch_grd, batch_sat_polar, batch_segmap)
        log.info(f"sat_matrix shape: {sat_matrix.shape}")
        log.info(f"grd_matrix shape: {grd_matrix.shape}")
        log.info(f"distance matrix shape: {distance.shape}")
        log.info(f"predicted orientation shape: {pred_orien.shape}")
        #Alcuni valori di esempio
        log.info(f"sat_matrix[0, 0, 0, :5]: {sat_matrix[0, 0, 0, :5]}")
        log.info(f"grd_matrix[0, 0, 0, :5]: {grd_matrix[0, 0, 0, :5]}")
    
    s_height, s_width, s_channel = list(sat_matrix.size())[1:]
    g_height, g_width, g_channel = list(grd_matrix.size())[1:]
    
    sat_batch_matrix = np.zeros([BATCH_SIZE, s_height, s_width, s_channel])
    grd_batch_matrix = np.zeros([BATCH_SIZE, g_height, g_width, g_channel])
    orientation_batch_gth = np.zeros([BATCH_SIZE])
    
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
    
    dist_array = 2.0-2.0*np.matmul(grd_batch_descriptor,np.transpose(sat_batch_descriptor))
    log.debug(f"dist_array shape: {dist_array.shape}")
    log.debug(dist_array)
    
    log.debug(f"distance matrix (returned by model shape: {distance.shape})")
    log.debug(distance)
    
    
    val_batch_accuracy = validate_original(dist_array,1)*100 
    r5_batch = validate_original(dist_array,5)*100
    r10_batch = validate_original(dist_array,10)*100
    r1p_batch = validate_original(dist_array,top1_percent_batch_value)*100
    
    log.info(f"R@1: {val_batch_accuracy:.2f}%, R@5: {r5_batch:.2f}%, R@10: {r10_batch:.2f}%, R@1%: {r1p_batch:.2f}% with Samples 1%: {top1_percent_batch_value}") 
    
    val_accuracy = validate_original(distance.cpu().detach().numpy(),1)*100 
    r5 = validate_original(distance.cpu().detach().numpy(),5)*100
    r10 = validate_original(distance.cpu().detach().numpy(),10)*100
    r1p = validate_original(distance.cpu().detach().numpy(),top1_percent_batch_value)*100
    
    log.info(f"(DISTANCE) R@1: {val_accuracy:.2f}%, R@5: {r5:.2f}%, R@10: {r10:.2f}%, R@1%: {r1p:.2f}% with Samples 1%: {top1_percent_batch_value}")
    
    # --- Metriche di confronto ---
    D1 = distance.cpu().detach().numpy()
    D2 = dist_array
    
    # Differenze numeriche
    diff = D1 - D2

    # Alcune diagnostiche utili
    log.info("\n=== DIAGNOSTIC SNIPPETS ===")
    log.info(f"Diag(distance) head:, {np.round(np.diag(D1)[:8], 4)}")
    log.info(f"Diag(dist_array) head:, {np.round(np.diag(D2)[:8], 4)}")
    # Mostra le posizioni (i,j) di max differenza
    ij_max = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
    log.info(f"argmax|diff| at (i,j)={ij_max}, D1={D1[ij_max]:.6f}, D2={D2[ij_max]:.6f}")
        
if __name__ == "__main__":
    debug_model()