import os
import torch
import numpy as np
from torch import optim
from Globals import config, folders_and_files, gradcam_config, images_params, experiments_config
from logger import log
from Utils import get_header_title, print_params, get_model_summary_simple, save_overlay_image, plot_iterative_loss, plot_loss_boxplot
from Network import GroundToAerialMatchingModel, compute_triplet_loss, GradCAM, compute_saliency_loss, compute_top1_accuracy
from Data import InputData
from sky_removal import remove_sky_from_image
import logging
import random
import cv2
import torch.nn.functional as F
import shutil

def recall_at_k(dist_mat: torch.Tensor, k: int) -> float:
    _, idx = dist_mat.topk(k, largest=False)      # più vicino = distanza minima
    correct = (torch.arange(dist_mat.size(0))[:, None] == idx).any(dim=1).float()
    return correct.mean().item() 

def validate_original(dist_array,topK):
    accuracy = 0.0
    data_amount = 0.0
    
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[i, :] < gt_dist)
        if prediction < topK:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy

def top1_percent_recall(dist_array: np.ndarray) -> float:
    n = dist_array.shape[0]
    top1_percent = max(int(round(n * 0.01)), 1)  # almeno 1
    gt_dist = np.diag(dist_array)
    rank = np.sum(dist_array < gt_dist[:, None], axis=1)  # posizione (0‑based)
    return np.mean(rank < top1_percent)

def flatten_descriptor(feature: torch.Tensor) -> torch.Tensor:
    return feature.reshape(feature.size(0), -1)

def _distance_dot(grd_flat, sat_flat):
    # replicazione esatta del training: satellite L2, ground NO
    sat_flat = F.normalize(sat_flat, dim=1, eps=1e-8)
    # (facoltativo) nessuna normalizzazione ground
    sim = torch.matmul(grd_flat, sat_flat.t())
    dist = 2.0 - 2.0 * sim
    return dist

def _distance_cosine(grd_flat, sat_flat):
    grd_flat = F.normalize(grd_flat, dim=1, eps=1e-8)
    sat_flat = F.normalize(sat_flat, dim=1, eps=1e-8)
    sim = torch.matmul(grd_flat, sat_flat.t())          # cos θ
    dist = 2.0 - 2.0 * sim
    return dist

def _distance_scalar_scaled(grd_flat, sat_flat):
    # Ground non normalizzato, ma scalato da un unico fattore medio
    scale = grd_flat.norm(p=2, dim=1).mean().clamp(min=1e-8)
    grd_scaled = grd_flat / scale
    sat_flat = F.normalize(sat_flat, dim=1, eps=1e-8)
    sim = torch.matmul(grd_scaled, sat_flat.t())
    dist = 2.0 - 2.0 * sim
    return dist

def train_model(experiment_name, overrides):
    
    experiments_config["name"] = experiment_name
    experiments_config["use_attention"] = overrides["use_attention"]
    experiments_config["remove_sky"] = overrides["remove_sky"]
    
    logs_folder = os.path.join(folders_and_files["log_folder"], experiment_name)
    models_folder = os.path.join(folders_and_files["saved_models_folder"], experiment_name)
    plots_folder = os.path.join(folders_and_files["plots_folder"], experiment_name)
    
    shutil.rmtree(logs_folder,ignore_errors=True)
    os.makedirs(logs_folder, exist_ok=True)
    
    shutil.rmtree(models_folder,ignore_errors=True)
    os.makedirs(models_folder, exist_ok=True)
    
    shutil.rmtree(plots_folder,ignore_errors=True)
    os.makedirs(plots_folder, exist_ok=True)
    
    experiments_config["logs_folder"] = logs_folder
    experiments_config["saved_models_folder"] = models_folder
    experiments_config["plots_folder"] = plots_folder
    
    FILE_LOGFORMAT = "%(asctime)s - %(levelname)s - %(funcName)s | %(message)s"
    file_formatter  = logging.Formatter(FILE_LOGFORMAT)
    streamFile = logging.FileHandler(filename=f"{logs_folder}/{folders_and_files['log_file']}", mode="w", encoding="utf-8")
    streamFile.setLevel(logging.DEBUG)
    streamFile.setFormatter(file_formatter)
    log.addHandler(streamFile)
    
    log.info(get_header_title(f"SETTING UP - {experiment_name}"))
    
    log.info("Experiment configuration:")
    log.info(f"Logs folder: {logs_folder}")
    log.info(f"Models folder: {models_folder}")
    log.info(f"Plots folder: {plots_folder}")
    log.info(f"Flag Use attention: {experiments_config['use_attention']}")
    log.info(f"Flag Remove sky: {experiments_config['remove_sky']}")
    log.info(get_header_title("END",new_line=True))
    
    log.info(get_header_title(f"PARAMETERS OF : {config['name']}"))
    print_params(config)
    log.info(get_header_title("END",new_line=True))
    
    log.info(get_header_title("LOADING DATASET"))
    input_data = InputData()
    log.info(get_header_title("END",new_line=True))
    
    log.info(get_header_title("INSTANTIATION OF THE MODEL"))
    
    #INSTANTIATION OF CONSTANTS FROM CONFIGURATION FILE
    train_grd_FOV = config["train_grd_FOV"]
    max_angle = images_params["max_angle"]
    max_width = images_params["max_width"]
    learning_rate = config["learning_rate"]
    
    width = int(train_grd_FOV / max_angle * max_width)
    
    #DEFINITION OF THE MODEL
    log.info("Creation of the model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GroundToAerialMatchingModel().to(device)
    log.info("Model created, summary: ")
    get_model_summary_simple(model)
    log.info(get_header_title("END", new_line=True))
    
    log.info(get_header_title("DEFINITION OF THE INPUT DATA"))
    grd_x = torch.zeros([2, int(max_width/4), width,3]).to(device)                             #ORDINE CAMBIATO: B (batch size)-C (channels)-H-W
    sat_x = torch.zeros([2, int(max_width/2), max_width,3]).to(device)
    polar_sat_x = torch.zeros([2, int(max_width/4), max_width,3]).to(device)
    segmap_x = torch.zeros([2, int(max_width/4), max_width,3]).to(device)
    log.info(f"Ground (zero) input matrix dimension: {grd_x.shape}")
    log.info(f"Polar satellite (zero) input matrix dimension: {polar_sat_x.shape}")
    log.info(f"Segmentation (zero) input matrix dimension: {segmap_x.shape}")
    log.info(get_header_title("END", new_line=True))

    log.info(get_header_title("CORRELATION MATRICES"))
    log.info("Calculating correlation matrices...")
    sat_matrix, grd_matrix, distance, pred_orien = model(grd_x, polar_sat_x, segmap_x)
    s_height, s_width, s_channel = list(sat_matrix.size())[1:]
    g_height, g_width, g_channel = list(grd_matrix.size())[1:]
    sat_global_matrix = np.zeros([input_data.get_test_dataset_size(), s_height, s_width, s_channel])
    log.info(f"Global satellite matrix dimensions: {sat_global_matrix.shape}")
    grd_global_matrix = np.zeros([input_data.get_test_dataset_size(), g_height, g_width, g_channel])
    log.info(f"Global ground matrix dimensions: {grd_global_matrix.shape}")
    orientation_gth = np.zeros([input_data.get_test_dataset_size()])
    log.info(f"Orientation matrix dimensions: {orientation_gth.shape}")
    log.info(get_header_title("END", new_line=True))
    
    log.info(get_header_title(f"STARTING TRAINING - {experiment_name}"))
    
    # Device
    log.info(f"Using device: {device}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adatta il learning rate
    #log.info(f"Using optimizer: {optimizer}")
    
    # Training loop
    loss_history = []
    epoch_losses = []
    epoch_r1 = []
    epoch_r5 = []
    epoch_r10 = []
    epoch_top1_percent_recall = []
    
    number_of_epoch = config["epochs"]
    experiments_config["save_ground_wo_sky"] = True
    
    for epoch in range(number_of_epoch):
        
        input_data.__cur_id = 0
        #random.shuffle(input_data.id_idx_list)  # <== AGGIUNGI QUESTO
        
        log.info(f"Epoch {epoch + 1}/{config['epochs']}")
        
        experiments_config["epoch_for_save"] = epoch + 1
        
        model.train()
        total_loss = 0
        iter_losses = []
        iter_recalls_r1 = []
        iter_recalls_r5 = []
        iter_recalls_r10 = []
        iter_top1_percent_recall = []
        iter_count = 0
        val_i=0
        end = False
        
        sat_global_matrix = np.zeros([input_data.get_dataset_size(), s_height, s_width, s_channel])
        grd_global_matrix = np.zeros([input_data.get_dataset_size(), g_height, g_width, g_channel])
        orientation_gth = np.zeros([input_data.get_dataset_size()])
        
        while not end:
            
            optimizer.zero_grad()
            grd_feats, sat_feats = [],[]
            
            sat_batch_matrix = np.zeros([config["batch_size"], s_height, s_width, s_channel])
            grd_batch_matrix = np.zeros([config["batch_size"], g_height, g_width, g_channel])
            orientation_batch_gth = np.zeros([config["batch_size"]])
            
            if(iter_count==-1):
                log.debug(f"sat_batch_matrix zeros shape: {sat_batch_matrix.shape}")
                log.debug(f"grd_batch_matrix zeros shape: {grd_batch_matrix.shape}")
                log.debug(f"orientation_batch_gth zeros shape: {orientation_batch_gth.shape}")
            
            batch_sat_polar, batch_sat, batch_grd, batch_segmap, batch_orien = input_data.next_pair_batch(config["batch_size"], grd_noise=config["train_grd_noise"], FOV=train_grd_FOV)
                
            if batch_sat is None:
                log.info("Satellite batch is None, breaking...") 
                end = True
                break  
        
            # Converti in tensori PyTorch
            batch_grd = torch.from_numpy(batch_grd).float().to(device)
            batch_sat_polar = torch.from_numpy(batch_sat_polar).float().to(device)
            batch_segmap = torch.from_numpy(batch_segmap).float().to(device)

            if experiments_config["use_attention"]:
                gradcam = GradCAM(model.ground_branch, gradcam_config["target_layer"])

            # Compute correlation and distance matrix
            sat_matrix, grd_matrix, distance, orien = model(batch_grd, batch_sat_polar, batch_segmap)

            # Compute the loss
            loss_value = compute_triplet_loss(distance,loss_weight=config["loss_weight"])
            #loss_value = loss_value / 4  # Divide by accumulation steps
            
            if experiments_config["use_attention"]:
                loss_value.backward(retain_graph=True)
                cam_size = (batch_grd.shape[1], batch_grd.shape[2])
                saliency_loss = compute_saliency_loss(gradcam, batch_grd, cam_size)
                total_loss_batch = loss_value + gradcam_config["lambda_saliency"] * saliency_loss
                total_loss_batch.backward(retain_graph=True)
            else:
                total_loss_batch = loss_value
                total_loss_batch.backward()
    
            #batch_loss = total_loss_batch.item() * 4
            batch_loss = total_loss_batch.item()
            total_loss += batch_loss
            iter_losses.append(batch_loss)
                
            optimizer.step()
            
            #Popolo matrice sat_global_matrix per valutazione ad ogni epoca
            sat_global_matrix[val_i:val_i+sat_matrix.shape[0],:]=sat_matrix.cpu().detach().numpy()
            grd_global_matrix[val_i:val_i+grd_matrix.shape[0],:]=grd_matrix.cpu().detach().numpy()
            orientation_gth[val_i:val_i+grd_matrix.shape[0]]=batch_orien
            
            #Popolo matrice sat_batch_matrix per valutazione ad ogni batch
            sat_batch_matrix[0:sat_matrix.shape[0],:]=sat_matrix.cpu().detach().numpy()
            grd_batch_matrix[0:grd_matrix.shape[0],:]=grd_matrix.cpu().detach().numpy()
            orientation_batch_gth[0:grd_matrix.shape[0]]=batch_orien
            
            if(iter_count==-1):
                log.debug(f"sat_batch_matrix shape after evaluation: {sat_batch_matrix.shape}")
                log.debug(f"grd_batch_matrix shape after evaluation: {grd_batch_matrix.shape}")
                log.debug(f"orientation_batch_gth shape after evaluation: {orientation_batch_gth.shape}")
            
            sat_batch_descriptor = np.reshape(sat_batch_matrix[:,:,:g_width,:],[-1,g_height*g_width*g_channel])
            norm = np.linalg.norm(sat_batch_descriptor, axis=-1, keepdims=True)
            sat_batch_descriptor = sat_batch_descriptor / np.maximum(norm,1e-12)
            grd_batch_descriptor = np.reshape(grd_batch_matrix,[-1,g_height*g_width*g_channel])
            
            data_batch_amount = grd_batch_descriptor.shape[0]
            top1_percent_batch_value = int(data_batch_amount*0.01)+1
            
            dist_array = 2.0-2.0*np.matmul(grd_batch_descriptor,np.transpose(sat_batch_descriptor))
            val_batch_accuracy = validate_original(dist_array,1)*100 
            r5_batch = validate_original(dist_array,5)*100
            r10_batch = validate_original(dist_array,10)*100
            r1p_batch = validate_original(dist_array,top1_percent_batch_value)*100
            
            if(iter_count == -1):
                log.debug(f"sat_batch_descriptor shape: {sat_batch_descriptor.shape}")
                log.debug(f"grd_batch_descriptor shape: {grd_batch_descriptor.shape}")
                log.debug(f"data_batch_amount: {data_batch_amount}")
                log.debug(f"1% samples of the data batch: {top1_percent_batch_value}")
                #log.debug("printing dist_array...")
                #log.debug(dist_array)
                #log.debug("printing distance...")
                #log.debug(distance)
                
                val_batch_accuracy = validate_original(distance.cpu().detach().numpy(),1)*100 
                r5_batch = validate_original(distance.cpu().detach().numpy(),5)*100
                r10_batch = validate_original(distance.cpu().detach().numpy(),10)*100
                r1p_batch = validate_original(distance.cpu().detach().numpy(),top1_percent_batch_value)*100
                log.info(f"---> ITERATION: {iter_count},(DISTANCE) R@1: {val_batch_accuracy:.2f}%, R@5: {r5_batch:.2f}%, R@10: {r10_batch:.2f}%, R@1%: {r1p_batch:.2f}% with Samples 1%: {top1_percent_batch_value}") 
            
            val_i += sat_matrix.shape[0]
                
            iter_recalls_r1.append(val_batch_accuracy)
            iter_recalls_r5.append(r5_batch)
            iter_recalls_r10.append(r10_batch)
            iter_top1_percent_recall.append(r1p_batch)
            
            if iter_count % config["log_frequency"] == 0:
                #log.info(f"ITERATION: {iter_count}, mini-Batch {i+1} LOSS VALUE: {total_loss_batch.item() * 4:.6f}, TOTAL LOSS: {total_loss:.6f}")
                log.info(f"---> ITERATION: {iter_count}, R@1: {val_batch_accuracy:.2f}%, R@5: {r5_batch:.2f}%, R@10: {r10_batch:.2f}%, R@1%: {r1p_batch:.2f}% with Samples 1%: {top1_percent_batch_value}")
                log.info(f"---> ITERATION: {iter_count}, BATCH LOSS VALUE: {total_loss_batch.item():.6f}, (PARTIAL INCREASING) TOTAL EPOCH LOSS: {total_loss:.6f}") 
                if iter_count % config["save_cam_png_frequency"] == 0 and experiments_config["use_attention"]:
                    heatmap_np = gradcam.generate(cam_size)
                    save_overlay_image(batch_grd[0], heatmap_np[0], path=f"{plots_folder}/epoch{epoch+1}_iter{iter_count}_cam.png", alpha=0.75)
                    
            iter_count += 1

        sat_descriptor = np.reshape(sat_global_matrix[:,:,:g_width,:],[-1,g_height*g_width*g_channel])
        norm = np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)
        sat_descriptor = sat_descriptor / np.maximum(norm,1e-12)
        grd_descriptor = np.reshape(grd_global_matrix,[-1,g_height*g_width*g_channel])
        
        data_amount = grd_descriptor.shape[0]
        top1_percent_value = int(data_amount*0.01)+1
        dist_array = 2.0-2.0*np.matmul(grd_descriptor,np.transpose(sat_descriptor))
        
        if(epoch==-1):
            log.debug(f"End of the firts epoch...")
            log.debug(f"sat_descriptor shape: {sat_descriptor.shape}")
            log.debug(f"grd_descriptor shape: {grd_descriptor.shape}")
            log.debug(f"data_amount: {data_amount}")
            log.debug(f"1% samples of the data batch: {top1_percent_value}")
            #log.debug("printing dist_array...")
            #log.debug(dist_array)
            #log.debug("printing distance...")
            #log.debug(distance)
        
        val_accuracy = validate_original(dist_array,1)*100 
        r5_epoch = validate_original(dist_array,5)*100
        r10_epoch = validate_original(dist_array,10)*100
        r1p_epoch = validate_original(dist_array,top1_percent_value)*100
        
        log.info(f"FINAL LOG - EPOCH: {epoch+1}, ITERATION: {iter_count}, LAST BATCH LOSS VALUE: {total_loss_batch.item():.6f}, TOTAL EPOCH LOSS: {total_loss:.6f}")
        #log.info(f"FINAL LOG - EPOCH: {epoch+1}, SAMPLES 1%: {top1_percent_value}, R@1: {val_accuracy:.4f}%, R@5: {r5_epoch:.4f}%, R@10: {r10_epoch:.4f}%, R@1%: {r1p_epoch:.4f}%")
        #if experiments_config["use_attention"]:
        #    heatmap_np = gradcam.generate(cam_size)
        #    save_overlay_image(batch_grd[0], heatmap_np[0], path=f"{experiments_config['plots_folder']}/epoch_{epoch+1}_iter{iter_count}_cam.png")
        
        if experiments_config["remove_sky"] and not experiments_config["flag_save_ground_wo_sky"]:
            experiments_config["flag_save_ground_wo_sky"] = True # Reset flag for next epoch
        
        model_epoch_folder = os.path.join(models_folder, "epoch"+str(epoch+1))
        os.makedirs(model_epoch_folder, exist_ok=True)
        
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, os.path.join(model_epoch_folder, "model.pth"))
        
        loss_history.append(total_loss)
        epoch_losses.append(iter_losses)
        
        epoch_r1.append(iter_recalls_r1)
        epoch_r5.append(iter_recalls_r5)
        epoch_r10.append(iter_recalls_r10)
        epoch_top1_percent_recall.append(iter_top1_percent_recall)
        
        np.save(os.path.join(logs_folder, "loss_history.npy"), np.array(loss_history))
        np.save(os.path.join(logs_folder, "epoch_losses.npy"), np.array(epoch_losses, dtype=object))
        np.save(os.path.join(logs_folder, "epoch_r1.npy"), np.array(epoch_r1, dtype=object))
        np.save(os.path.join(logs_folder, "epoch_r5.npy"), np.array(epoch_r5, dtype=object))
        np.save(os.path.join(logs_folder, "epoch_r10.npy"), np.array(epoch_r10, dtype=object))
        np.save(os.path.join(logs_folder, "epoch_top1_percent_recall.npy"), np.array(epoch_top1_percent_recall, dtype=object))
        
    np.save(os.path.join(logs_folder, "loss_history.npy"), np.array(loss_history))
    np.save(os.path.join(logs_folder, "epoch_losses.npy"), np.array(epoch_losses, dtype=object))
    
    np.save(os.path.join(logs_folder, "epoch_r1.npy"), np.array(epoch_r1, dtype=object))
    np.save(os.path.join(logs_folder, "epoch_r5.npy"), np.array(epoch_r5, dtype=object))
    np.save(os.path.join(logs_folder, "epoch_r10.npy"), np.array(epoch_r10, dtype=object))
    np.save(os.path.join(logs_folder, "epoch_top1_percent_recall.npy"), np.array(epoch_top1_percent_recall, dtype=object))
    
    plot_iterative_loss(epoch_losses, experiment_name, os.path.join(plots_folder,f"{experiment_name}_training_iterative_loss.png"))
    #plot_loss_boxplot(epoch_losses, experiment_name, os.path.join(plots_folder, "loss_boxplot.png"))
    
    log.info(get_header_title(f"TRAINING COMPLETED - {experiment_name}"))    
        
        