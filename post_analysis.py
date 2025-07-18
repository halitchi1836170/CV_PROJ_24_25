import os
import numpy as np
import matplotlib.pyplot as plt
from logger import log
from Globals import EXPERIMENTS, folders_and_files

EXPERIMENTS = EXPERIMENTS.keys()

def plot_loss_curves():
    plt.figure(figsize=(10, 6))
    for name in EXPERIMENTS:
        path = f"{folders_and_files["log_folder"]}/{name}/loss_history.npy"
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
    plt.savefig(f"{folders_and_files["log_folder"]}/loss_comparison.png")
    log.info(f"Grafico salvato in {folders_and_files["log_folder"]}/loss_comparison.png")
    #plt.show()

def main():
    plot_loss_curves()
    log.info(f"Grafico salvato in {folders_and_files["log_folder"]}/loss_comparison.png")

if __name__ == "__main__":
    main()