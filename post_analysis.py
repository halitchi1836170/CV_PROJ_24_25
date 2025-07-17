import os
import numpy as np
import matplotlib.pyplot as plt
from logger import log

EXPERIMENTS = ["BASE", "ATTENTION", "SKYREMOVAL", "FULL"]

def plot_loss_curves():
    plt.figure(figsize=(10, 6))
    for name in EXPERIMENTS:
        path = f"logs/{name}/loss_history.npy"
        if os.path.exists(path):
            loss = np.load(path)
            plt.plot(loss, label=name)
        else:
            log.warning(f"File non trovato: {path}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Confronto andamento Loss tra esperimenti")
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/loss_comparison.png")
    #plt.show()

def main():
    plot_loss_curves()
    log.info("Grafico salvato in logs/loss_comparison.png")

if __name__ == "__main__":
    main()