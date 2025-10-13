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
from Globals import folders_and_files, config, images_params, experiments_config,gradcam_config,BLOCKING_COUNTER
from Network import GroundToAerialMatchingModel,GradCAM,compute_saliency_loss,gradcam_from_activations,saliency_variability_loss
from Train import validate_original
import shutil
from Utils import get_header_title
from insertion_deletion_eval import insertion_test,deletion_test,plot_insertion_deletion_curves


EXPERIMENTS = {
    "BASE": {"use_attention": False, "remove_sky": False},
    #"ATTENTION": {"use_attention": True, "remove_sky": False},
    #"SKYREMOVAL": {"use_attention": False, "remove_sky": True},
    #"FULL": {"use_attention": True, "remove_sky": True},
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
    max_rank = int(np.clip(max_rank, 1, n))
    gt = np.diag(dist_array)
    ranks = (dist_array < gt[:, None]).sum(axis=1)  # zero-based rank
    cmc = np.array([np.mean(ranks <= r) for r in range(max_rank)], dtype=float)
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
        if(exp_dir in EXPERIMENTS.keys()):
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
        else:
            log.info(f"Experiment folder: {exp_dir} not in dictionary of possible experiments...")
    return checkpoints

def evaluate_checkpoint(ckpt_path, batch_size, grd_FOV, grd_noise, run_insertion_deletion=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Loading checkpoint: {ckpt_path}")
    model = GroundToAerialMatchingModel().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    log.info("Model loaded:")
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    log.info(f"Model loss vaulue: {checkpoint.get('loss', 'N/A')}")
    log.info(f"Model epoch: {checkpoint.get('epoch', 'N/A')}")
    
    name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    
    experiments_config["name"] = name
    experiments_config["use_attention"] = EXPERIMENTS[name]["use_attention"]
    experiments_config["remove_sky"] = EXPERIMENTS[name]["remove_sky"]
    
    # Logs folder
    local_logs_folder = os.path.join(folders_and_files["log_folder"],"EVALUATION",""f"{name}")
    eval_loss_path = os.path.join(local_logs_folder, f"{name}_eval_losses.npy")
    local_plots_folder = os.path.join(folders_and_files["log_folder"],"EVALUATION",""f"{name}")
    
    shutil.rmtree(local_logs_folder,ignore_errors=True)
    os.makedirs(local_logs_folder,exist_ok=True)
    
    shutil.rmtree(local_plots_folder,ignore_errors=True)
    os.makedirs(local_plots_folder,exist_ok=True)
    
    experiments_config["logs_folder"] = local_logs_folder
    experiments_config["plots_folder"] = local_plots_folder
    
    # Dataset
    log.info(get_header_title("LOADING DATASET"))
    data = InputData()
    n_test_samples = data.get_test_dataset_size()
    log.info(f"Dataset size: {n_test_samples} test samples")
    log.info(get_header_title("END",new_line=True))
    
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
    eval_batch_losses = []
    counter = 0
    data.__cur_id = 0
    
    model.eval()
    
    log.info(f"Started Evaluation...")
        
    while True:
        
        if(counter+1==BLOCKING_COUNTER):
            log.info("Breaking...")
            break
        
        log.info(f"Batch: {counter+1}/{int(n_test_samples/batch_size)}")
        batch_sat_polar, batch_sat, batch_grd, batch_segmap, batch_orien = data.next_batch_scan(batch_size, grd_noise=grd_noise, FOV=grd_FOV)
        
        if batch_sat is None:
            log.info("Breaking because there is no more data...")
            break
        
        grd = torch.from_numpy(batch_grd).float().to(device)
        sat_polar = torch.from_numpy(batch_sat_polar).float().to(device)
        seg = torch.from_numpy(batch_segmap).float().to(device)
        
        with torch.no_grad():
            log.info(f"Inferencing...")
            sat_mat, grd_mat, distance, _ = model(grd, sat_polar, seg)
            loss_value = compute_triplet_loss(distance,batch_size,loss_weight=config["loss_weight"])
            total_loss_batch = loss_value
            batch_loss = total_loss_batch.item()    
        
        log.info("Updating matrix for descriptors...")    
        sat_global_matrix[val_i:val_i+sat_mat.shape[0],:]=sat_mat.cpu().detach().numpy()
        grd_global_matrix[val_i:val_i+grd_mat.shape[0],:]=grd_mat.cpu().detach().numpy()
        orientation_gth[val_i:val_i+grd_mat.shape[0]]=batch_orien
        
        eval_batch_losses.append(batch_loss)
        val_i += sat_mat.shape[0]
        counter += 1
    
    log.info(f"Evaluation terminated, calculating metrics...")
        
    sat_descriptor = np.reshape(sat_global_matrix[:,:,:g_width,:],[-1,g_height*g_width*g_channel])
    norm = np.linalg.norm(sat_descriptor, axis=-1, keepdims=True)
    sat_descriptor = sat_descriptor / np.maximum(norm,1e-12)
    grd_descriptor = np.reshape(grd_global_matrix,[-1,g_height*g_width*g_channel])
    
    data_amount = grd_descriptor.shape[0]
    top1_percent_value = int(data_amount*0.01)+1
    dist_array = 2.0-2.0*np.matmul(grd_descriptor,np.transpose(sat_descriptor))
    
    val_accuracy = validate_original(dist_array,1)*100.0 
    r5_epoch = validate_original(dist_array,5)*100.0
    r10_epoch = validate_original(dist_array,10)*100.0
    r1p_epoch = validate_original(dist_array,top1_percent_value)*100.0
    
    log.info(f"FINAL LOG - EVALUATION, SAMPLES 1%: {top1_percent_value}, R@1: {val_accuracy:.2f}%, R@5: {r5_epoch:.2f}%, R@10: {r10_epoch:.2f}%, R@1%: {r1p_epoch:.2f}%")
    
    log.info("Saving evaluation losses...\n")
    np.save(eval_loss_path, np.array(eval_batch_losses))
    
    checkpoint_experiment_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    cmc_dist_array = compute_cmc(dist_array)
    
    # Plot top-5 matches
    num_examples_plot=5
    save_path = os.path.join(folders_and_files["plots_folder"],f"{checkpoint_experiment_name}", f"{checkpoint_experiment_name}_top_{num_examples_plot}_matches.png")
    log.info(f"Plotting top-{num_examples_plot} matches to {save_path}...")   
    plot_top5_matches(save_path,data,dist_array,num_examples_plot)
    
    # Run insertion and deletion tests if requested
    
    insertion_deletion_results = {}
    
    if run_insertion_deletion:
        log.info(get_header_title("INSERTION AND DELETION TESTS",new_line=True))
        
        log.info(get_header_title("DELETION TEST",new_line=True))
        deletion_save_path = os.path.join(local_logs_folder, f"{name}_deletion_results.npy")
        deletion_results= deletion_test(
            model=model,
            data=data,
            device=device,
            batch_size=batch_size,
            grd_FOV=grd_FOV,
            grd_noise=grd_noise,
            num_steps=100,  # 100 steps for 1% increments
            save_path=deletion_save_path
        )
        insertion_deletion_results['deletion'] = deletion_results
        
        log.info(get_header_title("INSERTION TEST",new_line=True))
        insertion_save_path = os.path.join(local_logs_folder, f"{name}_insertion_results.npy")
        insertion_results = insertion_test(
            model=model,
            data=data,
            device=device,
            batch_size=batch_size,
            grd_FOV=grd_FOV,
            grd_noise=grd_noise,
            num_steps=100,  # 100 steps for 1% increments
            save_path=insertion_save_path
        )
        insertion_deletion_results['insertion'] = insertion_results
        
        log.info(get_header_title("END INSERTION AND DELETION TESTS", new_line=True))
        
        log.info(get_header_title("SAVING INSERTIOON AND DELETION CURVES"))
        # Plot individual experiment curves
        individual_plot_path = os.path.join(local_plots_folder, f"{name}_insertion_deletion_curves.png")
        plot_insertion_deletion_curves(
            {name: insertion_deletion_results},
            individual_plot_path
        )
        
    return cmc_dist_array,insertion_deletion_results
    
def main():
    parser = argparse.ArgumentParser(description="Evaluation for Ground-Aerial matching model")
    parser.add_argument("--model", type=str, help="Path to a single model checkpoint (.pth)")
    parser.add_argument("--batch_size", type=int, default=config["batch_size"], help="Batch size for inference")
    parser.add_argument("--grd_FOV", type=int, default=config["test_grd_FOV"] or 360)
    parser.add_argument("--grd_noise", type=int, default=0)
    parser.add_argument("--skip_insertion_deletion", action="store_true", help="Skip insertion/deletion tests", default=False)
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
    
    all_cmc_dist_array = {}
    all_insertion_deletion_results={}
    
    for ckpt in checkpoints:
        cmc_dist_array,insertion_deletion_results = evaluate_checkpoint(
            ckpt,
            batch_size=args.batch_size,
            grd_FOV=args.grd_FOV,
            grd_noise=args.grd_noise,
            run_insertion_deletion=not args.skip_insertion_deletion
        )
        checkpoint_experiment_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
        all_cmc_dist_array[checkpoint_experiment_name] = cmc_dist_array

        if insertion_deletion_results:
            all_insertion_deletion_results[checkpoint_experiment_name]=insertion_deletion_results
        
    plot_cmc_curve(os.path.join(folders_and_files["plots_folder"],"ALL_experiments_cmc_curve_dist_array.png"),all_cmc_dist_array)
    
    if all_insertion_deletion_results:
        combined_plot_path = os.path.join(
            folders_and_files["plots_folder"],
            "ALL_experiments_insertion_deletion_curves.png"
        )
        plot_insertion_deletion_curves(all_insertion_deletion_results, combined_plot_path)
        
        # Summary log
        log.info(get_header_title("INSERTION/DELETION SUMMARY"))
        for exp_name, results in all_insertion_deletion_results.items():
            log.info(f"\n{exp_name}:")
            if 'deletion' in results:
                log.info(f"  Deletion AUC: {results['deletion']['auc']:.4f}")
            if 'insertion' in results:
                log.info(f"  Insertion AUC: {results['insertion']['auc']:.4f}")
        log.info(get_header_title("END", new_line=True))
    
if __name__ == "__main__":
    main()    