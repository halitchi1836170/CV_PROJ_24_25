import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from torch import optim
from Globals import BLOCKING_COUNTER, config, folders_and_files, images_params, experiments_config,previous_models,gradcam_config
from logger import log
from Utils import get_header_title, print_params, get_model_summary_simple, save_overlay_image,save_grd_sat_overlay_image,save_ovarlay_image_for_gif
from Network import GroundToAerialMatchingModel, HookManager, compute_gradcam_from_acts_grads, compute_triplet_loss, saliency_variability_loss
from Data import InputData
import logging
import cv2
import torch.nn.functional as F
import shutil
import json


def plot_top5_matches(save_path, data: InputData, dist_array: np.ndarray, num_examples: int = 5):

    fig, axes = plt.subplots(num_examples, 8, figsize=(24, 3 * num_examples))
    for i in range(num_examples):
        top5_idx = np.argsort(dist_array[i])[:5]
        correct_idx = i
        # carica immagini ground e satellite
        grd_path = data.img_root + data.id_test_list[i][2].replace("input", "").replace("png", "jpg")
        sat_correct_path = data.img_root + data.id_test_list[i][1]

        grd_img = cv2.imread(grd_path)[..., ::-1]
        
        sat_correct = cv2.imread(sat_correct_path)[..., ::-1]
        sat_correct = cv2.resize(sat_correct, (int(sat_correct.shape[1] * 0.5), int(sat_correct.shape[0] * 0.5)))

        
        
        axes[i, 0].imshow(grd_img)
        axes[i, 0].set_title("Ground")
        
        axes[i, 1].imshow(sat_correct)
        axes[i, 1].set_title("Correct Sat")
        
        axes[i, 2].axis('off')  # spazio vuoto come separatore
        
        for j in range(5):
            sat_path = data.img_root + data.id_test_list[top5_idx[j]][1]
            
            sat_img = cv2.imread(sat_path)[..., ::-1]
            sat_img = cv2.resize(sat_img, (int(sat_img.shape[1] * 0.5), int(sat_img.shape[0] * 0.5)))
            
            axes[i, j+3].imshow(sat_img)
            title = f"Top-{j+1}"
            if top5_idx[j] == correct_idx:
                title += " OK"
                axes[i, j+3].spines['bottom'].set_color('green')
                axes[i, j+3].spines['bottom'].set_linewidth(4)
            elif j == 0:
                axes[i, j+3].spines['bottom'].set_color('red')
                axes[i, j+3].spines['bottom'].set_linewidth(4)

            axes[i, j+3].set_title(title)

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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

def lr_lambda(epoch):
    #log.info(f"ACTUAL EPOCH: {epoch}, CONFIG MAG EPOCHS: {config['epochs']}")
    if epoch < int((2/3)*config["epochs"]):
        #log.info("RETURNING 1.0 LR LAMBDA FACTOR")
        return 1.0
    else:
        #log.info(f"RETURNING {config['fin_tuning_learning_rate']/config['learning_rate']} LR LABDA FACTOR")
        return config["fin_tuning_learning_rate"]/config["learning_rate"]

def compute_cam_metrics(cam: torch.Tensor, thresholds=[0.2, 0.5]):
    cam_np=cam.numpy()
    if cam_np.ndim == 3:
        cam_np = cam_np.squeeze(0)

    metrics = {}

    # Statistiche base
    metrics["sum"] = float(cam_np.sum())
    metrics["mean"] = float(cam_np.mean())
    metrics["max"] = float(cam_np.max())
    metrics["std"] = float(cam_np.std())

    # Area sopra soglia
    for t in thresholds:
        metrics[f"area_thr{t}"] = float((cam_np > t).mean())

    # Centroide e varianza spaziale
    h, w = cam_np.shape
    y, x = np.mgrid[0:h, 0:w]
    cam_pos = np.maximum(cam_np, 0)  # CAM dovrebbe essere non-negativa (ReLU)
    cam_norm = cam_pos / (cam_pos.sum() + 1e-12)

    metrics["centroid_x"] = float((x * cam_norm).sum())
    metrics["centroid_y"] = float((y * cam_norm).sum())
    metrics["var_x"] = float(((x - metrics["centroid_x"])**2 * cam_norm).sum())
    metrics["var_y"] = float(((y - metrics["centroid_y"])**2 * cam_norm).sum())

    # Entropia spaziale (log base 2)
    flat = cam_norm.flatten()
    flat = flat[flat > 1e-12]  # evita log(0)
    metrics["entropy"] = float(-(flat * np.log2(flat)).sum())

    return metrics

def save_cam_metrics_json(cam, epoch, iter_count, image_name, config_name, cam_type, thresholds=[0.2, 0.5]):
    
    metrics = compute_cam_metrics(cam, thresholds=thresholds)
    metrics.update({
        "epoch": epoch,
        "iteration": iter_count,
        "type": cam_type,
        "image_name": image_name
    })

    # Path finale
    save_dir = f"./logs/{config_name}/cam_metrics/cam_metrics{image_name}/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch{epoch}_iter{iter_count}_{cam_type}_cam_metric_{image_name}.json")

    # Salvataggio
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics

def train_model(experiment_name, overrides):
    
    experiments_config["name"] = experiment_name
    experiments_config["use_attention"] = overrides["use_attention"]
    experiments_config["remove_sky"] = overrides["remove_sky"]
    
    logs_folder = os.path.join(folders_and_files["log_folder"], experiment_name)
    models_folder = os.path.join(folders_and_files["saved_models_folder"], experiment_name)
    plots_folder = os.path.join(folders_and_files["plots_folder"], experiment_name)
    
    shutil.rmtree(logs_folder,ignore_errors=True)
    os.makedirs(logs_folder, exist_ok=True)
    
    #shutil.rmtree(models_folder,ignore_errors=True)
    #os.makedirs(models_folder, exist_ok=True)
    
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
    
    hm_grd_loss = HookManager(model, f"ground_branch.{gradcam_config['saliency_loss_layer']}")
    
    hm_grd_vis = HookManager(model, f"ground_branch.{gradcam_config['visualization_layer']}")
    hm_sat_vis = HookManager(model, f"satellite_branch.{gradcam_config['visualization_layer']}")
    
    start_epoch=0
    if(previous_models[f"flag_use_last_checkpoint_{experiment_name}"]==True):
        log.info(f"Using last checkpoint model for configuration {experiment_name}")
        checkpoint = torch.load(previous_models[f"last_checkpoint_path_{experiment_name}"],map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        log.info("Model loaded:")
        log.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        log.info(f"Model loss vaulue: {checkpoint.get('loss', 'N/A')}")
        log.info(f"Model epoch: {checkpoint.get('epoch', 'N/A')}")
        start_epoch = checkpoint.get('epoch', 'N/A') 
    
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
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_lambda)
    
    # Training loop
    loss_history = []
    epoch_losses = []
    eval_epoch_losses = []
    
    epoch_r1 = []
    epoch_r5 = []
    epoch_r10 = []
    epoch_top1_percent_recall = []
    eval_epoch_r1 = []
    eval_epoch_r5 = []
    eval_epoch_r10 = []
    eval_epoch_top1_percent_recall = []
    
    number_of_epochs = config["epochs"]+start_epoch
    config["epochs"]=number_of_epochs
    
    # hyperparam consigliati
    ema_beta = 0.98           # smoothing EMA (più vicino a 1 => più lento)
    alpha_target = 0.2        # frazione desiderata (10–30% tipico)
    lambda_min, lambda_max = 0.01, 0.3
    ema_eps = 1e-8

    # stato EMA (float Python, non tensori)
    ema_trip = 0.0
    ema_sal  = 0.0
    ema_t    = 0              # contatore per bias-correction (facoltativa)

    # opzionale: warm-up su alpha (per non spingere subito troppo)
    warmup_epochs = 5
    
    ACCUMULATION_STEPS = config["accumulation_steps"]
    
    for epoch in range(start_epoch,number_of_epochs):
        
        experiments_config["flag_save_ground_wo_sky"] = True
        input_data.__cur_id = 0
        #random.shuffle(input_data.id_idx_list)  # <== AGGIUNGI QUESTO
        
        log.info(f"Epoch {epoch + 1}/{config['epochs']}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        experiments_config["epoch_for_save"] = epoch + 1
        
        model.train()
        total_loss = 0
        iter_losses = []
        iter_recalls_r1 = []
        iter_recalls_r5 = []
        iter_recalls_r10 = []
        iter_top1_percent_recall = []
        iter_count = 1
        val_i=0
        end = False
        
        sat_global_matrix = np.zeros([input_data.get_dataset_size(), s_height, s_width, s_channel])
        grd_global_matrix = np.zeros([input_data.get_dataset_size(), g_height, g_width, g_channel])
        orientation_gth = np.zeros([input_data.get_dataset_size()])
        
        while not end:
            
            if iter_count % config["log_frequency"] == 0:
                log.info(f"BEGINING ITERATION: {iter_count}...")
            
            if iter_count == BLOCKING_COUNTER:
                end=True
                break
            
            if iter_count % ACCUMULATION_STEPS == 0:
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

            # Compute correlation and distance matrix
            sat_matrix, grd_matrix, distance, orien = model(batch_grd, batch_sat_polar, batch_segmap)

            # Compute the loss
            loss_value = compute_triplet_loss(distance,batch_size=config["batch_size"],loss_weight=config["loss_weight"])
            #loss_value = loss_value / 4  # Divide by accumulation steps
            
            
            if experiments_config["use_attention"]:
                
                target_scalar = -torch.diag(distance).sum()
                target_scalar.backward(retain_graph=True)
                cam_grd_loss = compute_gradcam_from_acts_grads(hm_grd_loss.activations, hm_grd_loss.gradients, upsample_to=(batch_grd.shape[1], batch_grd.shape[2])).detach()
                
                cam_grd_vis = compute_gradcam_from_acts_grads(hm_grd_vis.activations, hm_grd_vis.gradients, upsample_to=(batch_grd.shape[1], batch_grd.shape[2])).detach()
                cam_sat_vis = compute_gradcam_from_acts_grads(hm_sat_vis.activations, hm_sat_vis.gradients, upsample_to=(batch_grd.shape[1], batch_grd.shape[2])).detach()
                
                cam_grd_log = cam_grd_vis.detach().cpu()
                cam_sat_log = cam_sat_vis.detach().cpu()
                
                model.zero_grad(set_to_none=True)
                
                sal = saliency_variability_loss(cam_grd_loss)
                
                ema_t += 1
                ema_trip = ema_beta * ema_trip + (1 - ema_beta) * float(loss_value.item())
                ema_sal  = ema_beta * ema_sal  + (1 - ema_beta) * float(sal.item())

                # (facoltativo) bias-correction stile Adam: migliora lo start-up
                corr = 1.0 - (ema_beta ** ema_t)
                ema_trip_corr = ema_trip / max(corr, 1e-12)
                ema_sal_corr  = ema_sal  / max(corr, 1e-12)

                # --- alpha con warm-up per epoche iniziali ---
                if epoch < warmup_epochs:
                    warmup_factor = (epoch + 1) / warmup_epochs
                else:
                    warmup_factor = 1.0
                alpha_eff = alpha_target * warmup_factor

                # --- calcolo lambda dinamico ---
                lambda_sal = alpha_eff * (ema_trip_corr / max(ema_sal_corr, ema_eps))
                lambda_sal = float(min(max(lambda_sal, lambda_min), lambda_max))
                
                total_loss_batch = loss_value + lambda_sal * sal
                # loss_value.backward(retain_graph=True)
                # saliency_loss = compute_saliency_loss(gradcam, batch_grd, cam_size)
                # total_loss_batch = loss_value + gradcam_config["lambda_saliency"] * saliency_loss
                # total_loss_batch.backward(retain_graph=True)
            else:
                total_loss_batch = loss_value
                

            (total_loss_batch / ACCUMULATION_STEPS).backward()
            
            #batch_loss = total_loss_batch.item() * 4
            batch_loss = total_loss_batch.item()
            total_loss += batch_loss
            iter_losses.append(batch_loss)
            
            if (iter_count +1)% ACCUMULATION_STEPS == 0:
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
            
            if(iter_count % config["log_frequency"] == 0):
                # log.debug(f"sat_batch_descriptor shape: {sat_batch_descriptor.shape}")
                # log.debug(f"grd_batch_descriptor shape: {grd_batch_descriptor.shape}")
                # log.debug(f"data_batch_amount: {data_batch_amount}")
                # log.debug(f"1% samples of the data batch: {top1_percent_batch_value}")
                # log.debug("printing dist_array...")
                # log.debug(dist_array)
                # log.debug("printing distance...")
                # log.debug(distance)
                
                val_batch_accuracy_distance = validate_original(distance.cpu().detach().numpy(),1)*100 
                r5_batch_distance = validate_original(distance.cpu().detach().numpy(),5)*100
                r10_batch_distance = validate_original(distance.cpu().detach().numpy(),10)*100
                r1p_batch_distance = validate_original(distance.cpu().detach().numpy(),top1_percent_batch_value)*100
                log.info(f"---> ITERATION: {iter_count},(DISTANCE) R@1: {val_batch_accuracy_distance:.2f}%, R@5: {r5_batch_distance:.2f}%, R@10: {r10_batch_distance:.2f}%, R@1%: {r1p_batch_distance:.2f}% with Samples 1%: {top1_percent_batch_value}") 
            
            val_i += sat_matrix.shape[0]
                
            iter_recalls_r1.append(val_batch_accuracy)
            iter_recalls_r5.append(r5_batch)
            iter_recalls_r10.append(r10_batch)
            iter_top1_percent_recall.append(r1p_batch)
            
            if iter_count % config["log_frequency"] == 0:
                #log.info(f"ITERATION: {iter_count}, mini-Batch {i+1} LOSS VALUE: {total_loss_batch.item() * 4:.6f}, TOTAL LOSS: {total_loss:.6f}")
                log.info(f"---> ITERATION: {iter_count}, R@1: {val_batch_accuracy:.2f}%, R@5: {r5_batch:.2f}%, R@10: {r10_batch:.2f}%, R@1%: {r1p_batch:.2f}% with Samples 1%: {top1_percent_batch_value}")
                log.info(f"---> ITERATION: {iter_count}, BATCH LOSS VALUE: {total_loss_batch.item():.6f}, TOTAL (MEAN) EPOCH LOSS: {total_loss/(iter_count+1):.6f}") 
            if (iter_count % config["save_cam_png_frequency"] == 0 and experiments_config["use_attention"]) or (experiments_config["use_attention"] and input_data.flag_ground_to_be_tracked):
                heatmap_np = cam_grd_log
                heatmap_np_satellite = cam_sat_log
                if input_data.flag_ground_to_be_tracked:
                    for index in range(len(input_data.index_for_overlay)):
                        indexToBeTracked = input_data.index_for_overlay[index]
                        nameAtGivenIndex = input_data.index_for_name[index]
                        log.info(f"---> ITERATION: {iter_count}, CALCULATING AND SAVING CAM METRICS")
                        grd_cam_metrics=save_cam_metrics_json(heatmap_np[indexToBeTracked],epoch+1,iter_count,nameAtGivenIndex,experiment_name,"GRD")
                        sat_cam_metrics=save_cam_metrics_json(heatmap_np_satellite[indexToBeTracked],epoch+1,iter_count,nameAtGivenIndex,experiment_name,"SAT")
                        log.info(f"---> ITERATION: {iter_count}, GRD IMAGE TBT DETECTED, SAVING OVERLAY OF BATCH INDEX: {indexToBeTracked} REFERING TO IMAGE: {nameAtGivenIndex}...")
                        save_overlay_image(batch_grd[indexToBeTracked], heatmap_np[indexToBeTracked].numpy(), path=f"{plots_folder}/epoch{epoch+1}/epoch{epoch+1}_iter{iter_count}_grd_cam{nameAtGivenIndex}.png", alpha=0.75)
                        save_overlay_image(batch_sat_polar[indexToBeTracked], heatmap_np_satellite[indexToBeTracked].numpy(), path=f"{plots_folder}/epoch{epoch+1}/epoch{epoch+1}_iter{iter_count}_sat_cam{nameAtGivenIndex}.png", alpha=0.75)
                        save_ovarlay_image_for_gif(batch_grd[indexToBeTracked], heatmap_np[indexToBeTracked].numpy(), path=f"{plots_folder}/{'GIF'+nameAtGivenIndex}/epoch{epoch+1}_iter{iter_count}_grd_cam{nameAtGivenIndex}.png", alpha=0.75)
                        save_ovarlay_image_for_gif(batch_sat_polar[indexToBeTracked],heatmap_np_satellite[indexToBeTracked].numpy(), path=f"{plots_folder}/{'GIF'+nameAtGivenIndex}/epoch{epoch+1}_iter{iter_count}_sat_cam{nameAtGivenIndex}.png", alpha=0.75)
                        save_grd_sat_overlay_image(batch_grd[indexToBeTracked], heatmap_np[indexToBeTracked].numpy(),batch_sat_polar[indexToBeTracked], heatmap_np_satellite[indexToBeTracked].numpy(),path=f"{plots_folder}/epoch{epoch+1}/epoch{epoch+1}_iter{iter_count}_comp_grd_sat_cams{nameAtGivenIndex}.png",alpha=0.75)
                    input_data.index_for_overlay.clear()
                    input_data.index_for_name.clear()
                    input_data.flag_ground_to_be_tracked = False   
                if iter_count % config["save_cam_png_frequency"] == 0:
                    log.info(f"---> ITERATION: {iter_count}, SAVING CAM OVERLAY...")
                    save_overlay_image(batch_grd[0], heatmap_np[0].numpy(), path=f"{plots_folder}/epoch{epoch+1}/epoch{epoch+1}_iter{iter_count}_grd_cam.png", alpha=0.75)
                    save_overlay_image(batch_sat_polar[0], heatmap_np_satellite[0].numpy(), path=f"{plots_folder}/epoch{epoch+1}/epoch{epoch+1}_iter{iter_count}_sat_cam.png", alpha=0.75)
                    save_grd_sat_overlay_image(batch_grd[0], heatmap_np[0].numpy(),batch_sat_polar[0], heatmap_np_satellite[0].numpy(),path=f"{plots_folder}/epoch{epoch+1}/epoch{epoch+1}_iter{iter_count}_comp_grd_sat_cams.png",alpha=0.75)
            
            if iter_count % config["log_frequency"] == 0:
                log.info(f"END ITERATION: {iter_count}.")
                    
            iter_count += 1
        
        hm_grd_loss.remove()
        hm_grd_vis.remove()
        hm_sat_vis.remove()
        
        scheduler.step()
        
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
    
        
        log.info(f"FINAL LOG - EPOCH: {epoch+1}, ITERATION: {iter_count}, LAST BATCH LOSS VALUE: {total_loss_batch.item():.6f}, FINAL MEAN EPOCH OSS: {total_loss/iter_count:.6f}")
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
            'loss': total_loss_batch,
        }, os.path.join(model_epoch_folder, "model.pth"))
        
        loss_history.append(total_loss/iter_count)
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
        
        log.info(get_header_title(f"TRAINING COMPLETED - {experiment_name}")) 
        
        log.info(get_header_title(f"EVALUATION - {experiment_name}"))
    
        local_logs_folder = os.path.join(folders_and_files["log_folder"],"EVALUATION",""f"{experiment_name}")
        eval_loss_path = os.path.join(local_logs_folder, f"{experiment_name}_eval_losses.npy")
        
        shutil.rmtree(local_logs_folder,ignore_errors=True)
        os.makedirs(local_logs_folder,exist_ok=True)
        
        sat_global_matrix = np.zeros([input_data.get_test_dataset_size(), s_height, s_width, s_channel])
        grd_global_matrix = np.zeros([input_data.get_test_dataset_size(), g_height, g_width, g_channel])
        orientation_gth = np.zeros([input_data.get_test_dataset_size()])
        
        val_i=0
        eval_iter_losses = []
        
        n_test_samples = input_data.get_test_dataset_size()
        
        model.eval()
        input_data.__cur_id = 0
        
        log.info(f"Started Evaluation...")
        
        while True:
            batch_sat_polar, batch_sat, batch_grd, batch_segmap, batch_orien = input_data.next_batch_scan(config["batch_size"], grd_noise=0, FOV=config["test_grd_FOV"])
            
            if batch_sat is None:
                break
            
            grd = torch.from_numpy(batch_grd).float().to(device)
            sat_polar = torch.from_numpy(batch_sat_polar).float().to(device)
            seg = torch.from_numpy(batch_segmap).float().to(device)
            
            with torch.no_grad():
                sat_mat, grd_mat, distance, _ = model(grd, sat_polar, seg)
                loss_value = compute_triplet_loss(distance,batch_size=config["batch_size"],loss_weight=config["loss_weight"])
                total_loss_batch = loss_value
                batch_loss = total_loss_batch.item()    
                
            sat_global_matrix[val_i:val_i+sat_mat.shape[0],:]=sat_mat.cpu().detach().numpy()
            grd_global_matrix[val_i:val_i+grd_mat.shape[0],:]=grd_mat.cpu().detach().numpy()
            orientation_gth[val_i:val_i+grd_mat.shape[0]]=batch_orien
            
            eval_iter_losses.append(batch_loss)
            val_i += sat_matrix.shape[0]
        
        log.info(f"Evaluation terminated, calculating metrics...")
           
        eval_epoch_losses.append(eval_iter_losses) 
        np.save(os.path.join(logs_folder, "eval_epoch_losses.npy"), np.array(eval_epoch_losses, dtype=object)) 
            
        sat_descriptor = np.reshape(sat_global_matrix[:,:,:g_width,:],[-1,g_height*g_width*g_channel])
        norm = np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)
        sat_descriptor = sat_descriptor / np.maximum(norm,1e-12)
        grd_descriptor = np.reshape(grd_global_matrix,[-1,g_height*g_width*g_channel])
        
        data_amount = grd_descriptor.shape[0]
        top1_percent_value = int(data_amount*0.01)+1
        dist_array = 2.0-2.0*np.matmul(grd_descriptor,np.transpose(sat_descriptor))
        
        eval_val_accuracy = validate_original(dist_array,1)*100.0 
        eval_r5_epoch = validate_original(dist_array,5)*100.0
        eval_r10_epoch = validate_original(dist_array,10)*100.0
        eval_r1p_epoch = validate_original(dist_array,top1_percent_value)*100.0
        
        log.info(f"FINAL LOG - EVALUATION, SAMPLES 1%: {top1_percent_value}, R@1: {eval_val_accuracy:.2f}%, R@5: {eval_r5_epoch:.2f}%, R@10: {eval_r10_epoch:.2f}%, R@1%: {eval_r1p_epoch:.2f}%")
        
        eval_epoch_r1.append(eval_val_accuracy)
        eval_epoch_r5.append(eval_r5_epoch)
        eval_epoch_r10.append(eval_r10_epoch)
        eval_epoch_top1_percent_recall.append(eval_r1p_epoch)
        
        if(epoch==number_of_epochs-1):
            # Plot top-5 matches
            num_examples_plot=5
            save_path = os.path.join(folders_and_files["plots_folder"],f"{experiment_name}", f"{experiment_name}_top_{num_examples_plot}_matches.png")
            log.info(f"Plotting top-{num_examples_plot} matches to {save_path}...")   
            plot_top5_matches(save_path,input_data,dist_array,num_examples_plot)
            
    np.save(os.path.join(logs_folder, "loss_history.npy"), np.array(loss_history))
    np.save(os.path.join(logs_folder, "epoch_losses.npy"), np.array(epoch_losses, dtype=object))
    np.save(os.path.join(logs_folder, "eval_epoch_losses.npy"), np.array(eval_epoch_losses, dtype=object))
    
    np.save(os.path.join(logs_folder, "epoch_r1.npy"), np.array(epoch_r1, dtype=object))
    np.save(os.path.join(logs_folder, "epoch_r5.npy"), np.array(epoch_r5, dtype=object))
    np.save(os.path.join(logs_folder, "epoch_r10.npy"), np.array(epoch_r10, dtype=object))
    np.save(os.path.join(logs_folder, "epoch_top1_percent_recall.npy"), np.array(epoch_top1_percent_recall, dtype=object))
    
    np.save(os.path.join(logs_folder, "eval_epoch_r1.npy"), np.array(eval_epoch_r1, dtype=object))
    np.save(os.path.join(logs_folder, "eval_epoch_r5.npy"), np.array(eval_epoch_r5, dtype=object))
    np.save(os.path.join(logs_folder, "eval_epoch_r10.npy"), np.array(eval_epoch_r10, dtype=object))
    np.save(os.path.join(logs_folder, "eval_epoch_top1_percent_recall.npy"), np.array(eval_epoch_top1_percent_recall, dtype=object))
    
       
    
    