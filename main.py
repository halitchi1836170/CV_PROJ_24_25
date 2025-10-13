import argparse
from Train import train_model
from Evaluation import main as test_model
from Globals import EXPERIMENTS, folders_and_files
from logger import log
from Utils import get_header_title,print_params
import matplotlib.pyplot as plt
import os
import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#Disabilitato causa incompatibilità con le GPU 
torch.backends.cudnn.enabled = False

def create_comparison_plots():
    plt.figure(figsize=(10, 6))
    for name in EXPERIMENTS.keys():
        path = f"{folders_and_files['log_folder']}/{name}/loss_history.npy"
        if os.path.exists(path):
            log.info(f"Found, loading loss for plotting of experiment: {name}...")
            loss = np.load(path)
            plt.plot(loss, label=name)
        else:
            log.warning(f"File non trovato: {path}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Confronto andamento Loss tra esperimenti in fase di training")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folders_and_files['plots_folder']}/ALL_training_loss_comparison.png")
    log.info(f"Grafico salvato in {folders_and_files['plots_folder']}/ALL_training_loss_comparison.png")

def main():
    parser = argparse.ArgumentParser(description="Ground-Satellite Matching")
    parser.add_argument("--mode", type=str, choices=["TRAIN", "TEST"], default="TRAIN")
    parser.add_argument("--experiments", type=str, choices=["ALL", "FULL"], default="FULL")
    parser.add_argument("--model_path", type=str, help="Path to the model for TEST mode")
    args = parser.parse_args()

    if args.mode == "TRAIN":
        if args.experiments == "ALL":
            for name, config_overrides in EXPERIMENTS.items():
                log.info(get_header_title(f"Running experiment: {name}"))
                train_model(name, config_overrides)
                log.info(get_header_title(f"Experiment {name} completed", new_line=True))
        else:
            train_model("FULL", EXPERIMENTS["FULL"])

    elif args.mode == "TEST":
        if not args.model_path:
            raise ValueError("In TEST mode, --model_path must be specified.")
        test_model(args.model_path)


if __name__ == "__main__":
    main()