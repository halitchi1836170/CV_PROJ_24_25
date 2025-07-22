import os
import numpy as np
import matplotlib.pyplot as plt
from logger import log
from Globals import folders_and_files

EXPERIMENTS = {
    "BASE": {"use_attention": False, "remove_sky": False},
    "ATTENTION": {"use_attention": True, "remove_sky": False},
    "SKYREMOVAL": {"use_attention": False, "remove_sky": True},
    "FULL": {"use_attention": True, "remove_sky": True},
}

def plot_rolling_stats(window_size=50, title="Iterative Loss with Rolling Stats"):
    for name in EXPERIMENTS.keys():
        path = f"./container_files{folders_and_files['log_folder']}/{name}/epoch_losses.npy"
        if os.path.exists(path):
            save_path = f"./container_files{folders_and_files['plots_folder']}/{name}_epoch_losses_rolling_stats.png"
            log.info(f"Found, loading loss for plotting of experiment: {name}...")
            epoch_losses = np.load(path, allow_pickle=True)
            flattened_losses = [loss for epoch in epoch_losses for loss in epoch]
            
            losses = np.array(flattened_losses, dtype=float)
            N = len(losses)
            
            # Calcolo rolling mean e std
            rolling_mean = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            
            rolling_std = np.array([
                np.std(losses[i:i+window_size]) if i+window_size <= N else np.nan
                for i in range(N - window_size + 1)
            ])

            iterations = np.arange(len(rolling_mean))

            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(iterations, rolling_mean, label=f'{name} - Iterative Loss Rolling Mean', linewidth=1)
            plt.fill_between(iterations, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.3, label=f'{name} - ±1 Std Dev')
            plt.title(name+"-"+title+f" (WS={window_size})")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path)
            #plt.show()

def create_training_evaluation_comparison_plots_with_rolling_stats(window_size=50):
    for name in EXPERIMENTS.keys():
        fig, ax = plt.subplots(figsize=(15, 8))
        save_path = f"./container_files{folders_and_files['plots_folder']}/{name}_train_eval_batch_losses_rolling_stats.png"
        path_eval_losses = f"./container_files{folders_and_files['log_folder']}/{name}/{name}_eval_losses.npy"
        path_batch_losses = f"./container_files{folders_and_files['log_folder']}/{name}/epoch_losses.npy"
        
        if os.path.exists(path_batch_losses) and os.path.exists(path_eval_losses):
            log.info(f"Found, loading loss for plotting of experiment: {name}...")
            epoch_losses = np.load(path_batch_losses, allow_pickle=True)
            val_losses = np.load(path_eval_losses, allow_pickle=True)
            
            if isinstance(epoch_losses[0], (list, np.ndarray)):
                flattened_epoch_losses = [loss for epoch in epoch_losses for loss in epoch]
            else:
                flattened_epoch_losses = epoch_losses
            
            if isinstance(val_losses[0], (list, np.ndarray)):
                flattened_val_losses = [loss for epoch in val_losses for loss in epoch]
            else:
                flattened_val_losses = val_losses
                
            epoch_losses = np.array(flattened_epoch_losses, dtype=float)
            val_losses = np.array(flattened_val_losses, dtype=float)
            
            N_epoch = len(epoch_losses)
            N_val = len(val_losses)
            
            # Calcolo rolling mean e std
            rolling_mean_epoch = np.convolve(epoch_losses, np.ones(window_size)/window_size, mode='valid')
            rolling_mean_val = np.convolve(val_losses, np.ones(window_size)/window_size, mode='valid')
            
            rolling_std_epoch = np.array([
                np.std(epoch_losses[i:i+window_size]) if i+window_size <= N_epoch else np.nan
                for i in range(N_epoch - window_size + 1)
            ])
            rolling_std_val = np.array([
                np.std(val_losses[i:i+window_size]) if i+window_size <= N_val else np.nan
                for i in range(N_val - window_size + 1)
            ])
            iterations_epoch = np.arange(len(rolling_mean_epoch))
            iterations_val = np.arange(len(rolling_mean_val))
            
            scale = len(rolling_mean_val) / len(rolling_mean_epoch)
            it_val_scaled = iterations_val / scale  
            
            ax.plot(iterations_epoch, rolling_mean_epoch, label=f'{name} - Batch Losses Rolling Mean', linewidth=1)
            ax.fill_between(iterations_epoch, rolling_mean_epoch - rolling_std_epoch, rolling_mean_epoch + rolling_std_epoch, alpha=0.3, label=f'{name} - Batch ±1 Std Dev')
            
            ax.plot(it_val_scaled, rolling_mean_val, label=f'{name} - Validation Losses Rolling Mean', linewidth=1)
            ax.fill_between(it_val_scaled, rolling_mean_val - rolling_std_val, rolling_mean_val + rolling_std_val, alpha=0.3, label=f'{name} - Validation ±1 Std Dev')
            
            # ---------- second axis ----------
            def tr_to_val(x):   # forward  (bottom ➜ top)
                return x * scale
            def val_to_tr(x):   # inverse  (top ➜ bottom)
                return x / scale
            secax = ax.secondary_xaxis('top', functions=(tr_to_val, val_to_tr))
            secax.set_xlabel('Validation iteration')
            secax.grid(False)
            
            ax.set_xlabel('Training iteration')
            ax.set_ylabel('Triplet loss')
            ax.set_title(f'{name} - Train vs Validation loss (Windows Size = {window_size})')
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            
        else:
            log.warning(f"File non trovato: {path_batch_losses} o {path_eval_losses}")
            

def create_base_comparison_with_rolling_stats(window_size=50):
    plt.figure(figsize=(15, 8))
    save_path = f"./container_files{folders_and_files['plots_folder']}/BASE_old_new_COMPARISON_epoch_losses_rolling_stats.png"
    list_name = ["BASE", "BASE_old"]
    for name in list_name:
        path = f"./container_files{folders_and_files['log_folder']}/{name}/epoch_losses.npy"
        if os.path.exists(path):
            log.info(f"Found, loading loss for plotting of experiment: {name}...")
            epoch_losses = np.load(path, allow_pickle=True)
            
            if isinstance(epoch_losses[0], (list, np.ndarray)):
                flattened_losses = [loss for epoch in epoch_losses for loss in epoch]
            else:
                flattened_losses = epoch_losses
            
            losses = np.array(flattened_losses, dtype=float)
            N = len(losses)
            
            # Calcolo rolling mean e std
            rolling_mean = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            
            rolling_std = np.array([
                np.std(losses[i:i+window_size]) if i+window_size <= N else np.nan
                for i in range(N - window_size + 1)
            ])
            iterations = np.arange(len(rolling_mean))
            plt.plot(iterations, rolling_mean, label=f'{name} - Iterative Loss Rolling Mean', linewidth=1)
            plt.fill_between(iterations, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.3, label=f'{name} - ±1 Std Dev')
        else:
            log.warning(f"File non trovato: {path}")
    plt.title(f"Confronto andamento Loss con Rolling Stats tra esperimenti in fase di training (WS={window_size})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path) 
        
def create_comparisol_plots_with_rolling_stats(window_size=50):
    plt.figure(figsize=(15, 8))
    save_path = f"./container_files{folders_and_files['plots_folder']}/COMPARISON_epoch_losses_rolling_stats.png"
    for name in EXPERIMENTS.keys():
        path = f"./container_files{folders_and_files['log_folder']}/{name}/epoch_losses.npy"
        if os.path.exists(path):
            log.info(f"Found, loading loss for plotting of experiment: {name}...")
            epoch_losses = np.load(path, allow_pickle=True)
            
            if isinstance(epoch_losses[0], (list, np.ndarray)):
                flattened_losses = [loss for epoch in epoch_losses for loss in epoch]
            else:
                flattened_losses = epoch_losses
            
            losses = np.array(flattened_losses, dtype=float)
            N = len(losses)
            
            # Calcolo rolling mean e std
            rolling_mean = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            
            rolling_std = np.array([
                np.std(losses[i:i+window_size]) if i+window_size <= N else np.nan
                for i in range(N - window_size + 1)
            ])
            iterations = np.arange(len(rolling_mean))
            plt.plot(iterations, rolling_mean, label=f'{name} - Iterative Loss Rolling Mean', linewidth=1)
            plt.fill_between(iterations, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.3, label=f'{name} - ±1 Std Dev')
        else:
            log.warning(f"File non trovato: {path}")
    plt.title(f"Confronto andamento Loss con Rolling Stats tra esperimenti in fase di training (WS={window_size})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


def create_comparison_plots():
    plt.figure(figsize=(10, 6))
    for name in EXPERIMENTS.keys():
        path = f"./container_files{folders_and_files['log_folder']}/{name}/loss_history.npy"
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
    plt.savefig(f"./container_files{folders_and_files['plots_folder']}/loss_comparison.png")
    log.info(f"Grafico salvato in container_files{folders_and_files['plots_folder']}/loss_comparison.png")

            

def plot_iterative_loss():
    for name in EXPERIMENTS.keys():
        path = f"./container_files{folders_and_files['log_folder']}/{name}/epoch_losses.npy"
        if os.path.exists(path):
            save_path = f"./container_files{folders_and_files['plots_folder']}/{name}_epoch_losses.png"
            log.info(f"Found, loading loss for plotting of experiment: {name}...")
            epoch_losses = np.load(path, allow_pickle=True)
            flattened_losses = [loss for epoch in epoch_losses for loss in epoch]
            plt.figure(figsize=(12, 6))
            plt.plot(flattened_losses, label=f"{name} - Iterative Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title(f"{name} - Mini batch Losses per Iteration")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            log.info(f"{name} - Mini batch losses plot salvato in: {save_path}")
        else:
            log.warning(f"File non trovato: {path}")

    
    
    
def main():
    create_comparison_plots()
    plot_iterative_loss()
    plot_rolling_stats(window_size=50, title="Iterative Loss with Variance area")
    create_comparisol_plots_with_rolling_stats(window_size=50)
    create_training_evaluation_comparison_plots_with_rolling_stats(window_size=50)
    create_base_comparison_with_rolling_stats(window_size=50)
    
    
if __name__ == "__main__":
    main()