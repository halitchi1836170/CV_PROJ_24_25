import argparse
import glob
import os
import sys
import logging
from datetime import datetime
from typing import List, Tuple
import numpy as np
import scipy.io as scio
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
from tqdm import tqdm
from Data import InputData
from Globals import folders_and_files, config, images_params
from Network import GroundToAerialMatchingModel
from Train import validate_original
import shutil

EXPERIMENTS = {
    "BASE": {"use_attention": False, "remove_sky": False},
    "ATTENTION": {"use_attention": True, "remove_sky": False},
    "SKYREMOVAL": {"use_attention": False, "remove_sky": True},
    "FULL": {"use_attention": True, "remove_sky": True},
}

from logger import log
from Network import compute_triplet_loss
import cv2
import matplotlib.pyplot as plt

from torch.nn import functional as F

logs_folder = f'{folders_and_files["log_folder"]}/EVALUATION'
shutil.rmtree(logs_folder,ignore_errors=True)
os.makedirs(logs_folder,exist_ok=True)

FILE_LOGFORMAT = "%(asctime)s - %(levelname)s - %(funcName)s | %(message)s"
file_formatter  = logging.Formatter(FILE_LOGFORMAT)
streamFile = logging.FileHandler(filename=f"{logs_folder}/{folders_and_files['log_file']}_final_evaluation", mode="w", encoding="utf-8")
streamFile.setLevel(logging.DEBUG)
streamFile.setFormatter(file_formatter)
log.addHandler(streamFile)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def plot_cmc_curve(save_path,cmc_dict: dict[str, np.ndarray]):
    plt.figure(figsize=(9, 6))
    for model_name, cmc_array in cmc_dict.items():
        ranks = np.arange(1, len(cmc_array) + 1)
        plt.plot(ranks, cmc_array * 100, label=model_name, marker='o')

    plt.xlabel("Rank (k)")
    plt.ylabel("Top-k Recall (%)")
    plt.title("CMC Curve - All Experiments")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_cmc(dist_array: np.ndarray, max_rank: int = 50) -> np.ndarray:
    n = dist_array.shape[0]
    ranks = np.sum(dist_array < np.diag(dist_array)[:, None], axis=1)
    cmc = np.zeros(max_rank)
    for r in range(max_rank):
        cmc[r] = np.mean(ranks <= r)
    return cmc

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


def l2_normalize(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=dim, eps=1e-12)

def recall_at_k(dist_mat: torch.Tensor, k: int) -> float:
    _, idx = dist_mat.topk(k, largest=False)      # più vicino = distanza minima
    correct = (torch.arange(dist_mat.size(0))[:, None] == idx).any(dim=1).float()
    return correct.mean().item() 

def validate(dist_array: np.ndarray, top_k: int = 1) -> float:
    
    correct = 0
    n = dist_array.shape[0]
    
    for i in range(n):
        gt_dist = dist_array[i, i]
        rank = np.sum(dist_array[i] < gt_dist)
        if rank < top_k:
            correct += 1
    
    return correct / n

def top1_percent_recall(dist_array: np.ndarray) -> float:
    n = dist_array.shape[0]
    top1_percent = max(int(round(n * 0.01)), 1)  # almeno 1
    gt_dist = np.diag(dist_array)
    rank = np.sum(dist_array < gt_dist[:, None], axis=1)  # posizione (0‑based)
    return np.mean(rank < top1_percent)

def find_checkpoints(root: str) -> List[str]:
    """Return latest `model.pth` in every experiment sub‑folder."""
    checkpoints = []
    for exp_dir in sorted(os.listdir(root)):
        log.info(f"Checking experiment folder: {exp_dir}...")
        exp_path = os.path.join(root, exp_dir)
        if not os.path.isdir(exp_path):
            log.info(f"Skipping non-directory: {exp_path}")
            continue
        # look for epoch*.pth files
        log.info(f"Searching for checkpoints in: {exp_path}")
        epochs = sorted(
            glob.glob(os.path.join(exp_path, "epoch*/model.pth")),
            key=lambda p: int(os.path.basename(os.path.dirname(p))[5:]),
        )
        log.info(f"Found {len(epochs)} checkpoints in {exp_path}")
        if epochs:
            checkpoints.append(epochs[-1])  # latest epoch
    return checkpoints

def evaluate_checkpoint(ckpt_path: str, batch_size: int = config["batch_size"], grd_FOV: int = 360, grd_noise: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading checkpoint: {ckpt_path}")
    model = GroundToAerialMatchingModel().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log.info("Model loaded:")
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    log.info(f"Model loss vaulue: {checkpoint.get('loss', 'N/A')}")
    log.info(f"Model epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # Dataset
    data = InputData()
    n_samples = data.get_test_dataset_size()
    log.info(f"Dataset size: {n_samples} test samples")
    
    pbar = tqdm(total=n_samples, ncols=80, desc="Inferencing", unit="img")
    
    eval_batch_losses = []
    name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))

    # Logs folder
    local_logs_folder = os.path.join(folders_and_files["log_folder"],"EVALUATION",""f"{name}")
    os.makedirs(local_logs_folder,exist_ok=True)
    
    eval_loss_path = os.path.join(local_logs_folder, f"{name}_eval_losses.npy")

    counter = 0
    
    train_grd_FOV = config["train_grd_FOV"]
    max_angle = images_params["max_angle"]
    max_width = images_params["max_width"]
    width = int(train_grd_FOV / max_angle * max_width)
    
    grd_x = torch.zeros([2, int(max_width/4), width,3]).to(device)                             #ORDINE CAMBIATO: B (batch size)-C (channels)-H-W
    sat_x = torch.zeros([2, int(max_width/2), max_width,3]).to(device)
    polar_sat_x = torch.zeros([2, int(max_width/4), max_width,3]).to(device)
    segmap_x = torch.zeros([2, int(max_width/4), max_width,3]).to(device)
    
    sat_matrix, grd_matrix, distance, pred_orien = model(grd_x, polar_sat_x, segmap_x)
    s_height, s_width, s_channel = list(sat_matrix.size())[1:]
    g_height, g_width, g_channel = list(grd_matrix.size())[1:]
    
    sat_global_matrix = np.zeros([data.get_test_dataset_size(), s_height, s_width, s_channel])
    grd_global_matrix = np.zeros([data.get_test_dataset_size(), g_height, g_width, g_channel])
    orientation_gth = np.zeros([data.get_test_dataset_size()])
    
    val_i=0
    
    epoch_r1 = []
    epoch_r5 = []
    epoch_r10 = []
    epoch_top1_percent_recall = []
    
    iter_recalls_r1 = []
    iter_recalls_r5 = []
    iter_recalls_r10 = []
    iter_top1_percent_recall = []
        
    while True:
        log.info(f"ITERATION {counter + 1}/{int(n_samples/batch_size)}...")
        batch_sat_polar, _, batch_grd, batch_segmap, batch_orien = data.next_batch_scan(batch_size, grd_noise=grd_noise, FOV=grd_FOV)
        
        if batch_sat_polar is None:
            break
        
        grd = torch.from_numpy(batch_grd).float().to(device)
        sat_polar = torch.from_numpy(batch_sat_polar).float().to(device)
        seg = torch.from_numpy(batch_segmap).float().to(device)
        
        with torch.no_grad():
            sat_mat, grd_mat, distance, _ = model(grd, sat_polar, seg)
            loss_value = compute_triplet_loss(distance,loss_weight=config["loss_weight"]).item()
            
            g_height, g_width, g_channel = list(grd_mat.size())[1:]
            grd_batch_matrix = np.zeros([batch_size, g_height, g_width, g_channel])
            grd_batch_matrix[0:grd_mat.shape[0],:]=grd_mat.cpu().detach().numpy()
            grd_batch_descriptor = np.reshape(grd_batch_matrix,[-1,g_height*g_width*g_channel])
            data_batch_amount = grd_batch_descriptor.shape[0]
            top1_percent_batch_value = int(data_batch_amount*0.01)+1
            
            val_batch_accuracy = validate_original(distance.cpu().detach().numpy(),1)*100 
            r5_batch = validate_original(distance.cpu().detach().numpy(),5)*100
            r10_batch = validate_original(distance.cpu().detach().numpy(),10)*100
            r1p_batch = validate_original(distance.cpu().detach().numpy(),top1_percent_batch_value)*100
            
            iter_recalls_r1.append(val_batch_accuracy)
            iter_recalls_r5.append(r5_batch)
            iter_recalls_r10.append(r10_batch)
            iter_top1_percent_recall.append(r1p_batch)
            
            log.info(f"EVALUATION (DISTANCE), R@1: {val_batch_accuracy:.2f}%, R@5: {r5_batch:.2f}%, R@10: {r10_batch:.2f}%, R@1%: {r1p_batch:.2f}% with Samples 1%: {top1_percent_batch_value}\n") 
        
        sat_global_matrix[val_i:val_i+sat_mat.shape[0],:]=sat_mat.cpu().detach().numpy()
        grd_global_matrix[val_i:val_i+grd_mat.shape[0],:]=grd_mat.cpu().detach().numpy()
        orientation_gth[val_i:val_i+grd_mat.shape[0]]=batch_orien
        
        eval_batch_losses.append(loss_value)
        
        #if(counter+1 <= 5):
        #    log.info(f"Batch {counter + 1} - Loss: {loss_value:.4f}")
        #    log.info("grd_mat shape: %s, sat_mat shape: %s", grd_mat.shape, sat_mat.shape)
        #    log.info("flattened grd_mat shape: %s, flattened sat_mat shape: %s", flatten_descriptor(grd_mat).cpu().shape, flatten_descriptor(sat_mat).cpu().shape)
            
        # grd_feats.append(flatten_descriptor(grd_mat).cpu())   # [B, D]
        # sat_feats.append(flatten_descriptor(sat_mat).cpu())   # [B, D]

        # pbar.update(grd_mat.shape[0])
        
        counter += 1
        val_i += sat_matrix.shape[0]
        
    pbar.close()
    
    epoch_r1.append(iter_recalls_r1)
    epoch_r5.append(iter_recalls_r5)
    epoch_r10.append(iter_recalls_r10)
    epoch_top1_percent_recall.append(iter_top1_percent_recall)
    
    np.save(os.path.join(local_logs_folder, "epoch_r1.npy"), np.array(epoch_r1, dtype=object))
    np.save(os.path.join(local_logs_folder, "epoch_r5.npy"), np.array(epoch_r5, dtype=object))
    np.save(os.path.join(local_logs_folder, "epoch_r10.npy"), np.array(epoch_r10, dtype=object))
    np.save(os.path.join(local_logs_folder, "epoch_top1_percent_recall.npy"), np.array(epoch_top1_percent_recall, dtype=object))
    
    sat_descriptor = np.reshape(sat_global_matrix[:,:,:g_width,:],[-1,g_height*g_width*g_channel])
    norm = np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)
    sat_descriptor = sat_descriptor / np.maximum(norm,1e-12)
    grd_descriptor = np.reshape(grd_global_matrix,[-1,g_height*g_width*g_channel])
    
    data_amount = grd_descriptor.shape[0]
    top1_percent_value = int(data_amount*0.01)+1
    dist_array = 2.0-2.0*np.matmul(grd_descriptor,np.transpose(sat_descriptor))
    
    val_accuracy = validate_original(dist_array,1)*100 
    r5_epoch = validate_original(dist_array,5)*100
    r10_epoch = validate_original(dist_array,10)*100
    r1p_epoch = validate_original(dist_array,top1_percent_value)*100
    
    log.info(f"FINAL LOG - EVALUATION, SAMPLES 1%: {top1_percent_value}, R@1: {val_accuracy:.2f}%, R@5: {r5_epoch:.2f}%, R@10: {r10_epoch:.2f}%, R@1%: {r1p_epoch:.2f}%")
    
    #grd_descriptor = torch.concat(grd_feats, dim=0)      # [N, D]
    #sat_descriptor = torch.concat(sat_feats, dim=0)      # [N, D]
    #assert grd_descriptor.shape == sat_descriptor.shape, "Ground-Sat Evaluation Shape mismatch"
    
    # dist_train_like   = _distance_dot(grd_descriptor, sat_descriptor) 
    # dist_cosine       = _distance_cosine(grd_descriptor, sat_descriptor)
    # dist_scalar_scale = _distance_scalar_scaled(grd_descriptor, sat_descriptor)
    
    # dist_train_like_np   = dist_train_like.cpu().numpy() 
    # dist_cosine_np       = dist_cosine.cpu().numpy() 
    # dist_scalar_scale_np = dist_scalar_scale.cpu().numpy() 
    
    # for name, dist in {
    #     "train-like": dist_train_like,
    #     "cosine"    : dist_cosine,
    #     "scaled"    : dist_scalar_scale,
    # }.items():

    #     r1  = recall_at_k(dist, 1) * 100
    #     r5  = recall_at_k(dist, 5) * 100
    #     r10 = recall_at_k(dist,10) * 100
    #     r1p = top1_percent_recall(dist.numpy()) * 100   # se vuoi riusare la tua
    #     log.info(f"[{name}]: Top-1 recall: {r1:5.2f}  Top-5 recall {r5:5.2f}  Top-10 recall {r10:5.2f}  Top-1% recall {r1p:5.2f}")
    
    
    #out_mat = os.path.join(os.path.dirname(ckpt_path), "descriptors.mat")
    # scio.savemat(
    #     out_mat,
    #     {
    #         "grd_descriptor": grd_descriptor,
    #         "sat_descriptor": sat_descriptor,
    #         "dist_array": dist_array,
    #         "top1": top1,
    #         "top1_percent": top1p,
    #     },
    # )
    # log.info(f"Saved descriptors & metrics in {out_mat}")
    
    checkpoint_experiment_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    
    cmc = compute_cmc(dist_array)
    
    # Plot top-5 matches
    num_examples_plot=5
    save_path = os.path.join(folders_and_files["plots_folder"],f"{checkpoint_experiment_name}", f"{checkpoint_experiment_name}_top_{num_examples_plot}_matches.png")
    log.info(f"Plotting top-{num_examples_plot} matches to {save_path}...")   
    plot_top5_matches(save_path,data,dist_array,num_examples_plot)
    
    log.info("Saving evaluation losses...\n")
    np.save(eval_loss_path, np.array(eval_batch_losses))
    
    return cmc
    
def main():
    parser = argparse.ArgumentParser(description="Evaluation for Ground-Aerial matching model")
    parser.add_argument("--model", type=str, help="Path to a single model checkpoint (.pth)")
    parser.add_argument("--batch_size", type=int, default=config["batch_size"], help="Batch size for inference")
    parser.add_argument("--grd_FOV", type=int, default=config["test_grd_FOV"] or 360)
    parser.add_argument("--grd_noise", type=int, default=0)
    args = parser.parse_args()    
    
    EXPERIMENT_NAMES = list(EXPERIMENTS.keys())
       
    if args.model:
        checkpoints = [args.model]
    else:
        root = folders_and_files["saved_models_folder"]
        log.info(f"Searching for checkpoints in: {root}")
        checkpoints = find_checkpoints(root)
        if not checkpoints:
            log.error("No checkpoints found — specify with --model, MODEL_PATH, or verify folder structure.")
            sys.exit(1)   
    
    log.info(f"Will evaluate {len(checkpoints)} following checkpoint(s):")
    counter = 1
    for c in checkpoints:
        log.info(f"checkpoint-{counter}: {c}")
        counter += 1
    
    all_cmc = {}
    
    for ckpt in checkpoints:
        cmc = evaluate_checkpoint(
            ckpt,
            batch_size=args.batch_size,
            grd_FOV=args.grd_FOV,
            grd_noise=args.grd_noise,
        )
        checkpoint_experiment_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
        all_cmc[checkpoint_experiment_name] = cmc
    
    plot_cmc_curve(os.path.join(folders_and_files["plots_folder"],"ALL_experiments_cmc_curve.png"),all_cmc)
    
if __name__ == "__main__":
    main()    