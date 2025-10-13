import os, re, glob
from PIL import Image, ImageDraw, ImageFont
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

def plot_evaluation_iterative_recalls_with_rolling_stats(window_size=50):
    for name in EXPERIMENTS.keys():
        path_r1 = f"./container_files{folders_and_files['log_folder']}/EVALUATION/{name}/epoch_r1.npy"
        path_r5 = f"./container_files{folders_and_files['log_folder']}/EVALUATION/{name}/epoch_r5.npy"
        path_r10 = f"./container_files{folders_and_files['log_folder']}/EVALUATION/{name}/epoch_r10.npy"
        path_r1p = f"./container_files{folders_and_files['log_folder']}/EVALUATION/{name}/epoch_top1_percent_recall.npy"
        
        if os.path.exists(path_r1) and os.path.exists(path_r5) and os.path.exists(path_r10) and os.path.exists(path_r1p):
            save_path = f"./container_files{folders_and_files['plots_folder']}/{name}/{name}_iter_recalls_rolling_stats.png"
            log.info(f"Found, loading recalls for plotting of experiment: {name}...")
            iter_r1 = np.load(path_r1, allow_pickle=True)
            iter_r5 = np.load(path_r5, allow_pickle=True)
            iter_r10 = np.load(path_r10, allow_pickle=True)
            iter_r1p = np.load(path_r1p, allow_pickle=True)
            
            flattened_r1 = [r for epoch in iter_r1 for r in epoch]
            flattened_r5 = [r for epoch in iter_r5 for r in epoch]
            flattened_r10 = [r for epoch in iter_r10 for r in epoch]
            flattened_r1p = [r for epoch in iter_r1p for r in epoch]

            N_r1 = len(flattened_r1)
            N_r5 = len(flattened_r5)    
            N_r10 = len(flattened_r10)
            N_r1p = len(flattened_r1p)
            
            # Calcolo rolling mean e std
            rolling_mean_r1 = np.convolve(flattened_r1, np.ones(window_size)/window_size, mode='valid')
            rolling_mean_r5 = np.convolve(flattened_r5, np.ones(window_size)/window_size, mode='valid')
            rolling_mean_r10 = np.convolve(flattened_r10, np.ones(window_size)/window_size, mode='valid')
            rolling_mean_r1p = np.convolve(flattened_r1p, np.ones(window_size)/window_size, mode='valid')
            
            rolling_std_r1 = np.array([
                np.std(flattened_r1[i:i+window_size]) if i+window_size <= N_r1 else np.nan
                for i in range(N_r1 - window_size + 1)
            ])
            rolling_std_r5 = np.array([
                np.std(flattened_r5[i:i+window_size]) if i+window_size <= N_r5 else np.nan
                for i in range(N_r5 - window_size + 1)
            ])
            rolling_std_r10 = np.array([
                np.std(flattened_r10[i:i+window_size]) if i+window_size <= N_r10 else np.nan
                for i in range(N_r10 - window_size + 1)
            ])
            rolling_std_r1p = np.array([
                np.std(flattened_r1p[i:i+window_size]) if i+window_size <= N_r1p else np.nan
                for i in range(N_r1p - window_size + 1)
            ])
            
            iterations_r1 = np.arange(len(rolling_mean_r1))
            iterations_r5 = np.arange(len(rolling_mean_r5))
            iterations_r10 = np.arange(len(rolling_mean_r10))
            iterations_r1p = np.arange(len(rolling_mean_r1p))
            
            plt.figure(figsize=(12, 6))
            plt.plot(iterations_r1, rolling_mean_r1, label=f"{name} - R@1 Rolling Mean", linewidth=2)
            plt.fill_between(iterations_r1, rolling_mean_r1 - rolling_std_r1, rolling_mean_r1 + rolling_std_r1, alpha=0.3, label=f"{name} - R@1 ±1 Std Dev")
            plt.plot(iterations_r5, rolling_mean_r5, label=f"{name} - R@5 Rolling Mean", linewidth=1)
            plt.fill_between(iterations_r5, rolling_mean_r5 - rolling_std_r5, rolling_mean_r5 + rolling_std_r5, alpha=0.3, label=f"{name} - R@5 ±1 Std Dev")
            plt.plot(iterations_r10, rolling_mean_r10, label=f"{name} - R@10 Rolling Mean", linewidth=1)
            plt.fill_between(iterations_r10, rolling_mean_r10 - rolling_std_r10, rolling_mean_r10 + rolling_std_r10, alpha=0.3, label=f"{name} - R@10 ±1 Std Dev")
            plt.plot(iterations_r1p, rolling_mean_r1p, label=f"{name} - Top-1% Recall Rolling Mean", linewidth=1)
            plt.fill_between(iterations_r1p, rolling_mean_r1p - rolling_std_r1p, rolling_mean_r1p + rolling_std_r1p, alpha=0.3, label=f"{name} - Top-1% Recall ±1 Std Dev")
            plt.xlabel("Iteration")
            plt.ylabel("Recall")
            plt.title(f"{name} - Batch Recalls with Rolling Stats (Window Size = {window_size})")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
            log.warning(f"File(s) non trovato per configurazione {name}")

def plot_evaluation_recall_k_comparison_distance_dist_array_rolling_stats(k_value, window_size=2):
    for name in EXPERIMENTS.keys():
        plt.figure(figsize=(15, 8))
        save_path = f"./container_files{folders_and_files['plots_folder']}/{name}/{name}_evaluation_comparison_recall_{k_value}_rolling_stats_distance_dist_array.png"
        path_rk = f"./container_files{folders_and_files['log_folder']}/EVALUATION/{name}/epoch_r{k_value}.npy"
        path_rk_distance = f"./container_files{folders_and_files['log_folder']}/EVALUATION/{name}/epoch_r{k_value}_distance.npy"
        if os.path.exists(path_rk) and os.path.exists(path_rk_distance):
            log.info(f"Found, loading recalls of distance and dist array comparison plot of experiment: {name}...")
            
            iter_rk = np.load(path_rk, allow_pickle=True)
            flattened_rk =  [r for epoch in iter_rk for r in epoch]
            N_rk = len(flattened_rk)
            rolling_mean_rk = np.convolve(flattened_rk, np.ones(window_size)/window_size, mode='valid')
            rolling_std_rk = np.array([
                np.std(flattened_rk[i:i+window_size]) if i+window_size <= N_rk else np.nan
                for i in range(N_rk - window_size + 1)
            ])
            iterations_rk = np.arange(len(rolling_mean_rk))
            
            iter_rk_distance = np.load(path_rk_distance, allow_pickle=True)
            flattened_rk_distance =  [r for epoch in iter_rk_distance for r in epoch]
            N_rk_distance = len(flattened_rk_distance)
            rolling_mean_rk_distance = np.convolve(flattened_rk_distance, np.ones(window_size)/window_size, mode='valid')
            rolling_std_rk_distance = np.array([
                np.std(flattened_rk_distance[i:i+window_size]) if i+window_size <= N_rk_distance else np.nan
                for i in range(N_rk_distance - window_size + 1)
            ])
            iterations_rk_distance = np.arange(len(rolling_mean_rk_distance))
            
            plt.plot(iterations_rk, rolling_mean_rk, label=f"{name} - R@{k_value} Rolling Mean", linewidth=2)
            plt.fill_between(iterations_rk, rolling_mean_rk - rolling_std_rk, rolling_mean_rk + rolling_std_rk, alpha=0.3, label=f"{name} - R@{k_value} ±1 Std Dev")
            
            plt.plot(iterations_rk_distance, rolling_mean_rk_distance, label=f"{name} - R@{k_value} Rolling Mean (distance)", linewidth=2)
            plt.fill_between(iterations_rk_distance, rolling_mean_rk_distance - rolling_std_rk_distance, rolling_mean_rk_distance + rolling_std_rk_distance, alpha=0.3, label=f"{name} - R@{k_value} ±1 Std Dev (distance)")
            
            plt.xlabel("Iteration")
            plt.ylabel("Recall")
            plt.title(f"{name} - Comparison Evaluation Batch Recalls with Rolling Stats when using distance or dist array (Window Size = {window_size})")
            plt.grid(True)
            plt.legend()
            plt.tight_layout(rect=[0, 0.07, 1, 1])
            
            # --- Calcolo medie globali e stampa sotto al grafico ---
            global_mean_rk = np.mean(flattened_rk)
            global_mean_rk_distance = np.mean(flattened_rk_distance)
            footer_text = f"Global Recall@{k_value} → dist array: {global_mean_rk:.2f}% | distance: {global_mean_rk_distance:.2f}%"
            plt.figtext(0.5, 0.02, footer_text, ha="center", va="bottom", fontsize=10, color="black")
            
            
            
            plt.savefig(save_path)
            plt.close() 
            
        else:
            log.warning(f"Not found distance nor dist array recall .npy file for {name}...")

def plot_evaluation_recall_k_comparison_with_rolling_stats_distance(k_value, window_size=2):
    plt.figure(figsize=(15, 8))
    save_path = f"./container_files{folders_and_files['plots_folder']}/ALL_evaluation_comparison_recall_{k_value}_rolling_stats_distance.png"
    text_lines = []
    for name in EXPERIMENTS.keys():
        path_rk = f"./container_files{folders_and_files['log_folder']}/EVALUATION/{name}/epoch_r{k_value}_distance.npy"
        if os.path.exists(path_rk):
            log.info(f"Found, loading recalls for plotting of experiment: {name}...")
            iter_rk = np.load(path_rk, allow_pickle=True)
            flattened_rk =  [r for epoch in iter_rk for r in epoch]
            N_rk = len(flattened_rk)
            rolling_mean_rk = np.convolve(flattened_rk, np.ones(window_size)/window_size, mode='valid')
            rolling_std_rk = np.array([
                np.std(flattened_rk[i:i+window_size]) if i+window_size <= N_rk else np.nan
                for i in range(N_rk - window_size + 1)
            ])
            iterations_rk = np.arange(len(rolling_mean_rk))
            plt.plot(iterations_rk, rolling_mean_rk, label=f"{name} - R@{k_value} Rolling Mean", linewidth=2)
            plt.fill_between(iterations_rk, rolling_mean_rk - rolling_std_rk, rolling_mean_rk + rolling_std_rk, alpha=0.3, label=f"{name} - R@{k_value} ±1 Std Dev")
            
            global_mean = np.mean(flattened_rk)
            text_lines.append(f"{name}: {global_mean:.2f}%")
        else:
            log.warning(f"Not found recall .npy file for {name}...")
    plt.xlabel("Iteration")
    plt.ylabel("Recall")
    plt.title(f"{name} - Comparison Evaluation Batch Recalls with Rolling Stats (Window Size = {window_size})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    footer_text = " | ".join(text_lines)
    plt.figtext(0.5, 0.01, f"Global Recall@{k_value} means → {footer_text}",ha="center", va="bottom", fontsize=10, color="black")
    plt.savefig(save_path)
    plt.close() 
        
def plot_evaluation_recall_k_comparison_with_rolling_stats(k_value, window_size=2):
    plt.figure(figsize=(15, 8))
    save_path = f"./container_files{folders_and_files['plots_folder']}/ALL_evaluation_comparison_recall_{k_value}_rolling_stats_dist_array.png"
    text_lines = []
    for name in EXPERIMENTS.keys():
        path_rk = f"./container_files{folders_and_files['log_folder']}/{name}/eval_epoch_r{k_value}.npy"
        if os.path.exists(path_rk):
            log.info(f"Found, loading recalls for plotting of experiment: {name}...")
            iter_rk = np.load(path_rk, allow_pickle=True)
            flattened_rk =  [r for epoch in iter_rk for r in epoch]
            N_rk = len(flattened_rk)
            rolling_mean_rk = np.convolve(flattened_rk, np.ones(window_size)/window_size, mode='valid')
            rolling_std_rk = np.array([
                np.std(flattened_rk[i:i+window_size]) if i+window_size <= N_rk else np.nan
                for i in range(N_rk - window_size + 1)
            ])
            iterations_rk = np.arange(len(rolling_mean_rk))
            plt.plot(iterations_rk, rolling_mean_rk, label=f"{name} - R@{k_value} Rolling Mean", linewidth=2)
            plt.fill_between(iterations_rk, rolling_mean_rk - rolling_std_rk, rolling_mean_rk + rolling_std_rk, alpha=0.3, label=f"{name} - R@{k_value} ±1 Std Dev")
            
            global_mean = np.mean(flattened_rk)
            text_lines.append(f"{name}: {global_mean:.2f}%")
        else:
            log.warning(f"Not found recall .npy file for {name}...")
    plt.xlabel("Iteration")
    plt.ylabel("Recall")
    plt.title(f"{name} - Comparison Evaluation Batch Recalls with Rolling Stats (Window Size = {window_size})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    footer_text = " | ".join(text_lines)
    plt.figtext(0.5, 0.01, f"Global Recall@{k_value} means → {footer_text}",ha="center", va="bottom", fontsize=10, color="black")
    plt.savefig(save_path)
    plt.close() 


def plot_evaluation_recall_k_comparison(k_value):
    plt.figure(figsize=(15, 8))
    save_path = f"./container_files{folders_and_files['plots_folder']}/ALL_evaluation_comparison_recall_{k_value}.png"
    text_lines = []
    for name in EXPERIMENTS.keys():
        path_rk = f"./container_files{folders_and_files['log_folder']}/{name}/eval_epoch_r{k_value}.npy"
        if os.path.exists(path_rk):
            log.info(f"Found, loading recalls for plotting of experiment: {name}...")
            values_rk = np.load(path_rk, allow_pickle=True)
            plt.plot(values_rk, label=f"{name} - R@{k_value}", linewidth=2)
            global_mean = np.mean(values_rk[-5:-1])
            last_value = values_rk[-1]
            text_lines.append(f"{name}: {global_mean:.2f}%, {last_value:.2f}%")
        else:
            log.warning(f"Not found recall .npy file for {name}...")
    plt.xlabel("Epoch")
    plt.ylabel("Recall [%]")
    plt.title(f"Comparison Evaluation Recalls")
    plt.grid(True)
    plt.legend()
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    footer_text = " | ".join(text_lines)
    plt.figtext(0.5, 0.01, f"Recall@{k_value} <Last 5 Mean>, <Last> → {footer_text}",ha="center", va="bottom", fontsize=10, color="black")
    plt.savefig(save_path)
    plt.close()


def plot_training_recall_k_comparison_with_rolling_stats(k_value, window_size=50):
    plt.figure(figsize=(15, 8))
    save_path = f"./container_files{folders_and_files['plots_folder']}/ALL_training_comparison_recall_{k_value}_rolling_stats.png"
    for name in EXPERIMENTS.keys():
        path_rk = f"./container_files{folders_and_files['log_folder']}/{name}/epoch_r{k_value}.npy"
        if os.path.exists(path_rk):
            log.info(f"Found, loading recalls for plotting of experiment: {name}...")
            iter_rk = np.load(path_rk, allow_pickle=True)
            flattened_rk =  [r for epoch in iter_rk for r in epoch]
            N_rk = len(flattened_rk)
            rolling_mean_rk = np.convolve(flattened_rk, np.ones(window_size)/window_size, mode='valid')
            rolling_std_rk = np.array([
                np.std(flattened_rk[i:i+window_size]) if i+window_size <= N_rk else np.nan
                for i in range(N_rk - window_size + 1)
            ])
            iterations_rk = np.arange(len(rolling_mean_rk))
            plt.plot(iterations_rk, rolling_mean_rk, label=f"{name} - R@{k_value} Rolling Mean", linewidth=2)
            plt.fill_between(iterations_rk, rolling_mean_rk - rolling_std_rk, rolling_mean_rk + rolling_std_rk, alpha=0.3, label=f"{name} - R@{k_value} ±1 Std Dev")
        else:
            log.info(f"Not found recall .npy file for {name}...")
    plt.xlabel("Iteration")
    plt.ylabel("Recall")
    plt.title(f"{name} - Comparison Training Batch Recalls with Rolling Stats (Window Size = {window_size})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()        
            
def plot_training_iterative_recalls_with_rolling_stats(window_size=50):
    for name in EXPERIMENTS.keys():
        path_r1 = f"./container_files{folders_and_files['log_folder']}/{name}/epoch_r1.npy"
        path_r5 = f"./container_files{folders_and_files['log_folder']}/{name}/epoch_r5.npy"
        path_r10 = f"./container_files{folders_and_files['log_folder']}/{name}/epoch_r10.npy"
        path_r1p = f"./container_files{folders_and_files['log_folder']}/{name}/epoch_top1_percent_recall.npy"
        
        if os.path.exists(path_r1) and os.path.exists(path_r5) and os.path.exists(path_r10) and os.path.exists(path_r1p):
            save_path = f"./container_files{folders_and_files['plots_folder']}/{name}/{name}_iter_recalls_rolling_stats.png"
            log.info(f"Found, loading recalls for plotting of experiment: {name}...")
            iter_r1 = np.load(path_r1, allow_pickle=True)
            iter_r5 = np.load(path_r5, allow_pickle=True)
            iter_r10 = np.load(path_r10, allow_pickle=True)
            iter_r1p = np.load(path_r1p, allow_pickle=True)
            
            flattened_r1 = [r for epoch in iter_r1 for r in epoch]
            flattened_r5 = [r for epoch in iter_r5 for r in epoch]
            flattened_r10 = [r for epoch in iter_r10 for r in epoch]
            flattened_r1p = [r for epoch in iter_r1p for r in epoch]

            N_r1 = len(flattened_r1)
            N_r5 = len(flattened_r5)    
            N_r10 = len(flattened_r10)
            N_r1p = len(flattened_r1p)
            
            # Calcolo rolling mean e std
            rolling_mean_r1 = np.convolve(flattened_r1, np.ones(window_size)/window_size, mode='valid')
            rolling_mean_r5 = np.convolve(flattened_r5, np.ones(window_size)/window_size, mode='valid')
            rolling_mean_r10 = np.convolve(flattened_r10, np.ones(window_size)/window_size, mode='valid')
            rolling_mean_r1p = np.convolve(flattened_r1p, np.ones(window_size)/window_size, mode='valid')
            
            rolling_std_r1 = np.array([
                np.std(flattened_r1[i:i+window_size]) if i+window_size <= N_r1 else np.nan
                for i in range(N_r1 - window_size + 1)
            ])
            rolling_std_r5 = np.array([
                np.std(flattened_r5[i:i+window_size]) if i+window_size <= N_r5 else np.nan
                for i in range(N_r5 - window_size + 1)
            ])
            rolling_std_r10 = np.array([
                np.std(flattened_r10[i:i+window_size]) if i+window_size <= N_r10 else np.nan
                for i in range(N_r10 - window_size + 1)
            ])
            rolling_std_r1p = np.array([
                np.std(flattened_r1p[i:i+window_size]) if i+window_size <= N_r1p else np.nan
                for i in range(N_r1p - window_size + 1)
            ])
            
            iterations_r1 = np.arange(len(rolling_mean_r1))
            iterations_r5 = np.arange(len(rolling_mean_r5))
            iterations_r10 = np.arange(len(rolling_mean_r10))
            iterations_r1p = np.arange(len(rolling_mean_r1p))
            
            plt.figure(figsize=(12, 6))
            plt.plot(iterations_r1, rolling_mean_r1, label=f"{name} - R@1 Rolling Mean", linewidth=2)
            plt.fill_between(iterations_r1, rolling_mean_r1 - rolling_std_r1, rolling_mean_r1 + rolling_std_r1, alpha=0.3, label=f"{name} - R@1 ±1 Std Dev")
            plt.plot(iterations_r5, rolling_mean_r5, label=f"{name} - R@5 Rolling Mean", linewidth=1)
            plt.fill_between(iterations_r5, rolling_mean_r5 - rolling_std_r5, rolling_mean_r5 + rolling_std_r5, alpha=0.3, label=f"{name} - R@5 ±1 Std Dev")
            plt.plot(iterations_r10, rolling_mean_r10, label=f"{name} - R@10 Rolling Mean", linewidth=1)
            plt.fill_between(iterations_r10, rolling_mean_r10 - rolling_std_r10, rolling_mean_r10 + rolling_std_r10, alpha=0.3, label=f"{name} - R@10 ±1 Std Dev")
            plt.plot(iterations_r1p, rolling_mean_r1p, label=f"{name} - Top-1% Recall Rolling Mean", linewidth=1)
            plt.fill_between(iterations_r1p, rolling_mean_r1p - rolling_std_r1p, rolling_mean_r1p + rolling_std_r1p, alpha=0.3, label=f"{name} - Top-1% Recall ±1 Std Dev")
            plt.xlabel("Iteration")
            plt.ylabel("Recall")
            plt.title(f"{name} - Training Batch Recalls with Rolling Stats (Window Size = {window_size})")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
            log.warning(f"File(s) non trovato per configurazione {name}")
            
            
def plot_evaluation_iterative_recalls_with_rolling_stats(window_size=50):
    for name in EXPERIMENTS.keys():
        path_r1 = f"./container_files{folders_and_files['log_folder']}/{name}/eval_epoch_r1.npy"
        path_r5 = f"./container_files{folders_and_files['log_folder']}/{name}/eval_epoch_r5.npy"
        path_r10 = f"./container_files{folders_and_files['log_folder']}/{name}/eval_epoch_r10.npy"
        path_r1p = f"./container_files{folders_and_files['log_folder']}/{name}/eval_epoch_top1_percent_recall.npy"
        
        if os.path.exists(path_r1) and os.path.exists(path_r5) and os.path.exists(path_r10) and os.path.exists(path_r1p):
            save_path = f"./container_files{folders_and_files['plots_folder']}/{name}/{name}_evaluation_iter_recalls_rolling_stats.png"
            log.info(f"Found, loading recalls for plotting of experiment: {name}...")
            iter_r1 = np.load(path_r1, allow_pickle=True)
            iter_r5 = np.load(path_r5, allow_pickle=True)
            iter_r10 = np.load(path_r10, allow_pickle=True)
            iter_r1p = np.load(path_r1p, allow_pickle=True)
            
            flattened_r1 = [r for epoch in iter_r1 for r in epoch]
            flattened_r5 = [r for epoch in iter_r5 for r in epoch]
            flattened_r10 = [r for epoch in iter_r10 for r in epoch]
            flattened_r1p = [r for epoch in iter_r1p for r in epoch]

            N_r1 = len(flattened_r1)
            N_r5 = len(flattened_r5)    
            N_r10 = len(flattened_r10)
            N_r1p = len(flattened_r1p)
            
            # Calcolo rolling mean e std
            rolling_mean_r1 = np.convolve(flattened_r1, np.ones(window_size)/window_size, mode='valid')
            rolling_mean_r5 = np.convolve(flattened_r5, np.ones(window_size)/window_size, mode='valid')
            rolling_mean_r10 = np.convolve(flattened_r10, np.ones(window_size)/window_size, mode='valid')
            rolling_mean_r1p = np.convolve(flattened_r1p, np.ones(window_size)/window_size, mode='valid')
            
            rolling_std_r1 = np.array([
                np.std(flattened_r1[i:i+window_size]) if i+window_size <= N_r1 else np.nan
                for i in range(N_r1 - window_size + 1)
            ])
            rolling_std_r5 = np.array([
                np.std(flattened_r5[i:i+window_size]) if i+window_size <= N_r5 else np.nan
                for i in range(N_r5 - window_size + 1)
            ])
            rolling_std_r10 = np.array([
                np.std(flattened_r10[i:i+window_size]) if i+window_size <= N_r10 else np.nan
                for i in range(N_r10 - window_size + 1)
            ])
            rolling_std_r1p = np.array([
                np.std(flattened_r1p[i:i+window_size]) if i+window_size <= N_r1p else np.nan
                for i in range(N_r1p - window_size + 1)
            ])
            
            iterations_r1 = np.arange(len(rolling_mean_r1))
            iterations_r5 = np.arange(len(rolling_mean_r5))
            iterations_r10 = np.arange(len(rolling_mean_r10))
            iterations_r1p = np.arange(len(rolling_mean_r1p))
            
            plt.figure(figsize=(12, 6))
            plt.plot(iterations_r1, rolling_mean_r1, label=f"{name} - R@1 Rolling Mean", linewidth=2)
            plt.fill_between(iterations_r1, rolling_mean_r1 - rolling_std_r1, rolling_mean_r1 + rolling_std_r1, alpha=0.3, label=f"{name} - R@1 ±1 Std Dev")
            plt.plot(iterations_r5, rolling_mean_r5, label=f"{name} - R@5 Rolling Mean", linewidth=1)
            plt.fill_between(iterations_r5, rolling_mean_r5 - rolling_std_r5, rolling_mean_r5 + rolling_std_r5, alpha=0.3, label=f"{name} - R@5 ±1 Std Dev")
            plt.plot(iterations_r10, rolling_mean_r10, label=f"{name} - R@10 Rolling Mean", linewidth=1)
            plt.fill_between(iterations_r10, rolling_mean_r10 - rolling_std_r10, rolling_mean_r10 + rolling_std_r10, alpha=0.3, label=f"{name} - R@10 ±1 Std Dev")
            plt.plot(iterations_r1p, rolling_mean_r1p, label=f"{name} - Top-1% Recall Rolling Mean", linewidth=1)
            plt.fill_between(iterations_r1p, rolling_mean_r1p - rolling_std_r1p, rolling_mean_r1p + rolling_std_r1p, alpha=0.3, label=f"{name} - Top-1% Recall ±1 Std Dev")
            plt.xlabel("Iteration")
            plt.ylabel("Recall")
            plt.title(f"{name} - Evaluation Batch Recalls with Rolling Stats (Window Size = {window_size})")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
            log.warning(f"File(s) non trovato per configurazione {name}")
                
                
def plot_training_rolling_stats(window_size=50, title="Iterative Loss with Rolling Stats"):
    for name in EXPERIMENTS.keys():
        path = f"./container_files{folders_and_files['log_folder']}/{name}/epoch_losses.npy"
        if os.path.exists(path):
            save_path = f"./container_files{folders_and_files['plots_folder']}/{name}/{name}_training_epoch_losses_rolling_stats.png"
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
        save_path = f"./container_files{folders_and_files['plots_folder']}/{name}/{name}_train_eval_batch_losses_rolling_stats.png"
        path_eval_losses = f"./container_files/logs/{name}/eval_epoch_losses.npy"
        path_batch_losses = f"./container_files/logs/{name}/epoch_losses.npy"
        
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
    list_name = ["BASE_old_3", "BASE_old_2" , "BASE_old_1", "BASE"]
    for name in list_name:
        path = f"./container_files/logs/{name}/epoch_losses.npy"
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
        
def create_training_comparison_plots_with_rolling_stats(window_size=50):
    plt.figure(figsize=(15, 8))
    save_path = f"./container_files{folders_and_files['plots_folder']}/ALL_comparison_epoch_losses_rolling_stats.png"
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


def create_training_comparison_plots():
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
    plt.savefig(f"./container_files{folders_and_files['plots_folder']}/ALL_training_loss_comparison.png")
    log.info(f"Grafico salvato in container_files{folders_and_files['plots_folder']}/loss_comparison.png")

def plot_training_iterative_recalls():
    for name in EXPERIMENTS.keys():
        path_r1 = f"./container_files{folders_and_files['log_folder']}/{name}/epoch_r1.npy"
        path_r5= f"./container_files{folders_and_files['log_folder']}/{name}/epoch_r5.npy"
        path_r10 = f"./container_files{folders_and_files['log_folder']}/{name}/epoch_r10.npy"
        path_r1p = f"./container_files{folders_and_files['log_folder']}/{name}/epoch_top1_percent_recall.npy"
        
        if os.path.exists(path_r1) and os.path.exists(path_r5) and os.path.exists(path_r10) and os.path.exists(path_r1p):
            save_path = f"./container_files{folders_and_files['plots_folder']}/{name}/{name}_training_iter_recalls.png"
            log.info(f"Found, loading recalls for plotting of experiment: {name}...")
            iter_r1 = np.load(path_r1, allow_pickle=True)
            iter_r5 = np.load(path_r5, allow_pickle=True)
            iter_r10 = np.load(path_r10, allow_pickle=True)
            iter_r1p = np.load(path_r1p, allow_pickle=True)
            flattened_r1 = [r for epoch in iter_r1 for r in epoch]
            flattened_r5 = [r for epoch in iter_r5 for r in epoch]
            flattened_r10 = [r for epoch in iter_r10 for r in epoch]
            flattened_r1p = [r for epoch in iter_r1p for r in epoch]
                             
            plt.figure(figsize=(12, 6))
            plt.plot(flattened_r1, label=f"{name} - R@1", markersize=2)
            plt.plot(flattened_r5, label=f"{name} - R@5", markersize=1)
            plt.plot(flattened_r10, label=f"{name} - R@10", markersize=1)
            plt.plot(flattened_r1p, label=f"{name} - Top-1% Recall", markersize=1)
            plt.xlabel("Iteration")
            plt.ylabel("Recall")
            plt.title(f"{name} - Batch Recalls")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            log.info(f"{name} - Batch recalls plot salvato in: {save_path}")
        else:
            log.warning(f"File(s) non trovato per configurazione {name}")            


# --- util: estrai ID immagine da nome cartella GIF_* ---
def extract_image_id_from_gif_dir(dir_name: str) -> str:
    """
    Ritorna l'ID immagine come stringa (es. '12' da 'GIF_12').
    Se non trova numeri, usa il nome cartella sanificato.
    """
    b = os.path.basename(dir_name)
    m = re.search(r"GIF[_-]?(\d+)", b, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    # fallback: togli caratteri non alfanumerici
    return re.sub(r"\W+", "_", b)

# --- indicizza i frame per (epoch, iter) ---
def frames_by_epoch_representative(folder, glob_pattern="epoch*_iter*_grd_cam_*.png", policy='min'):
    """
    Ritorna dict {epoch:int -> path} scegliendo un solo frame per ogni epoch.
    policy: 'min' (iter più piccolo) oppure 'max' (iter più grande).
    Se 'iter' manca nel filename, usa iter=0 per 'min' e iter=+inf per 'max'.
    """
    choose_min = (policy == 'min')
    rep = {}  # epoch -> (iter_val, path)
    for p in glob.glob(os.path.join(folder, glob_pattern)):
        b = os.path.basename(p)
        me = re.search(r"epoch(\d+)", b)
        if not me:
            continue
        ep = int(me.group(1))
        mi = re.search(r"iter(\d+)", b)
        if mi:
            it = int(mi.group(1))
        else:
            it = 0 if choose_min else float('inf')
        if ep not in rep:
            rep[ep] = (it, p)
        else:
            curr_it, _ = rep[ep]
            if (choose_min and it < curr_it) or ((not choose_min) and it > curr_it):
                rep[ep] = (it, p)
    # riduci a epoch->path
    return {ep: path for ep, (it, path) in rep.items()}

def _resize_to_height(im, target_h):
    if im.height == target_h:
        return im
    new_w = max(1, int(round(im.width * (target_h / im.height))))
    return im.resize((new_w, target_h), Image.LANCZOS)

def make_side_by_side_gif_for_folder(type,folder_A, folder_B, out_dir,
                                                label_A="ATTENTION", label_B="FULL",
                                                glob_pattern="epoch*_iter*_grd_cam_*.png",
                                                fps=2, sep_px=10, annotate=True,
                                                out_name_prefix="evolution_image",
                                                epoch_pick_policy='min'):
    """
    Crea GIF affiancata usando l'intersezione delle chiavi (epoch, iter) tra folder_A e folder_B.
    Salva in: out_dir/{out_name_prefix}_{image_id}_{label_A}_vs_{label_B}.gif
    """
    frames_A = frames_by_epoch_representative(folder_A, glob_pattern, policy=epoch_pick_policy)
    frames_B = frames_by_epoch_representative(folder_B, glob_pattern, policy=epoch_pick_policy)
    common_keys = sorted(set(frames_A.keys()) & set(frames_B.keys()))
    if not common_keys:
        return False, "no common (epoch, iter) between folders"

    # Altezza target: minima dei due primi frame comuni
    sampleA = Image.open(frames_A[common_keys[0]]).convert("RGB")
    sampleB = Image.open(frames_B[common_keys[0]]).convert("RGB")
    target_h = min(sampleA.height, sampleB.height)

    combined_frames = []
    for ep in common_keys:
        pa = frames_A[ep]
        pb = frames_B[ep]
        imA = Image.open(pa).convert("RGB")
        imB = Image.open(pb).convert("RGB")
        imA = _resize_to_height(imA, target_h)
        imB = _resize_to_height(imB, target_h)

        if annotate:
            imA = annotate_image(imA, f"{label_A} | epoch={ep}", pos="bottom")
            imB = annotate_image(imB, f"{label_B} | epoch={ep}", pos="bottom")

        W = imA.width + sep_px + imB.width
        canvas = Image.new("RGB", (W, target_h), (0, 0, 0))
        canvas.paste(imA, (0, 0))
        canvas.paste(imB, (imA.width + sep_px, 0))
        combined_frames.append(canvas)

    os.makedirs(out_dir, exist_ok=True)

    # metti l'ID immagine nel nome file
    image_id = extract_image_id_from_gif_dir(os.path.basename(out_dir))
    out_path = os.path.join(out_dir, f"{type}_{out_name_prefix}_{image_id}_{label_A}_vs_{label_B}.gif")

    duration_ms = int(1000 / max(1, fps))
    combined_frames[0].save(out_path, save_all=True, append_images=combined_frames[1:],
                            duration=duration_ms, loop=0, optimize=False)
    return True, out_path

def create_gradcam_pair_gifs(type,glob_pattern, expA="ATTENTION", expB="FULL",
                             root_plots_dir=os.path.join(".", "container_files", "plots"), fps=2):
    """
    Per ogni cartella GIF_* presente in entrambi gli esperimenti,
    crea la GIF affiancata e mette l'ID immagine nel nome file.
    Output: {root_plots_dir}/{expA}_vs_{expB}/{GIF_*}/evolution_image_{ID}_{expA}_vs_{expB}.gif
    """
    dirA = os.path.join(root_plots_dir, expA)
    dirB = os.path.join(root_plots_dir, expB)

    gif_dirs_A = {os.path.basename(p) for p in glob.glob(os.path.join(dirA, "GIF_*")) if os.path.isdir(p)}
    gif_dirs_B = {os.path.basename(p) for p in glob.glob(os.path.join(dirB, "GIF_*")) if os.path.isdir(p)}
    common = sorted(gif_dirs_A & gif_dirs_B)

    total = len(common)
    created = 0
    for sub in common:
        folder_A = os.path.join(dirA, sub)
        folder_B = os.path.join(dirB, sub)
        out_dir   = os.path.join(root_plots_dir, f"{expA}_vs_{expB}", sub)

        ok, msg = make_side_by_side_gif_for_folder(type,folder_A, folder_B, out_dir,
                                                   label_A=expA, label_B=expB,
                                                   glob_pattern=glob_pattern, fps=fps)
        if ok:
            print(f"GIF Created: {sub} -> {msg}")
            created += 1
        else:
            print(f"SKIP {sub}: {msg}")

    log.info(f"Done. Side-by-side GIFs: {created}/{total}")





def create_gradcam_gif(type,glob_pattern):
    total = 0
    for name in EXPERIMENTS.keys():
        cfg_dir = os.path.join(".\\container_files\\plots",name)
        log.info(f"Creating gif for configuration with dir: {cfg_dir}...")
        for gif_dir in sorted(glob.glob(os.path.join(cfg_dir,"GIF_*"))):
            log.info(f"Calling gif creation method for directory: {gif_dir}...")
            ok, msg = make_gif_for_folder(gif_dir,type,glob_pattern)
            if ok:
                print(f"GIF Created: {gif_dir} -> {msg}")
                total += 1
            else:
                print(f"SKIP GIF: {gif_dir}: {msg}")
    print(f"Done. GIF creation completed. Total of {total} GIFs.")

def natural_iter_key(p):
    """Ordina prima per 'epochNNN' se presente, altrimenti per mtime."""
    b = os.path.basename(p)
    m = re.search(r"epoch(\d+)", b)
    iter_num = int(m.group(1)) if m else None
    return (0, iter_num) if iter_num is not None else (1, os.path.getmtime(p))

def annotate_image(img, text, pos="bottom"):
    FONT = None                  # o un path a un .ttf se vuoi un font specifico
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    pad = int(0.02 * min(W, H))
    # Banner semitrasparente
    banner_h = int(0.08 * H)
    y0 = H - banner_h if pos == "bottom" else 0
    draw.rectangle([0, y0, W, y0 + banner_h], fill=(0, 0, 0, 120))
    # Testo
    try:
        font = ImageFont.truetype(FONT, int(banner_h*0.5)) if FONT else ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    tw, th = draw.textbbox((0,0), text, font=font)[2:]
    draw.text(((W - tw)//2, y0 + (banner_h - th)//2), text, fill=(255,255,255,255), font=font)
    return img.convert("RGB")

def make_gif_for_folder(folder,type,glob_pattern):
    FPS = 2                      # frame al secondo (GIF)
    DURATION_MS = int(1000 / FPS)
    ANNOTATE = True              # scrive "iter=..." in sovraimpressione
    TEXT_POS = "bottom"          # "top" oppure "bottom"
    frames = sorted(glob.glob(os.path.join(folder, glob_pattern)), key=natural_iter_key)
    if not frames:
        return False, "no frames"
    # prepara immagini con la stessa size
    imgs = []
    base = Image.open(frames[0]).convert("RGB")
    W, H = base.size
    for p in frames:
        im = Image.open(p).convert("RGB")
        if im.size != (W, H):
            im = im.resize((W, H), Image.LANCZOS)
        bname = os.path.basename(p)
        m = re.search(r"epoch(\d+)", bname)
        text = f"{bname}"
        if m: text = f"epoch={int(m.group(1))} | {bname}"
        if ANNOTATE:
            im = annotate_image(im, text, pos=TEXT_POS)
        imgs.append(im)
    out_path = os.path.join(folder, f"{type}_evolution.gif")
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=DURATION_MS, loop=0, optimize=False)
    return True, out_path

def plot_training_iterative_loss():
    for name in EXPERIMENTS.keys():
        path = f"./container_files{folders_and_files['log_folder']}/{name}/epoch_losses.npy"
        if os.path.exists(path):
            save_path = f"./container_files{folders_and_files['plots_folder']}/{name}/{name}_epoch_losses.png"
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
            log.warning(f"File non trovato: {path} per configurazione {name}")

def plot_evaluation_epoch_recalls():
    for name in EXPERIMENTS.keys():
        path_r1 = f"./container_files{folders_and_files['log_folder']}/{name}/eval_epoch_r1.npy"
        path_r5 = f"./container_files{folders_and_files['log_folder']}/{name}/eval_epoch_r5.npy"
        path_r10 = f"./container_files{folders_and_files['log_folder']}/{name}/eval_epoch_r10.npy"
        path_r1p = f"./container_files{folders_and_files['log_folder']}/{name}/eval_epoch_top1_percent_recall.npy"
        if os.path.exists(path_r1) and os.path.exists(path_r5) and os.path.exists(path_r10) and os.path.exists(path_r1p):
            save_path = f"./container_files{folders_and_files['plots_folder']}/{name}/{name}_evaluation_recall_over_epochs.png"
            log.info(f"Found, loading evaluation recalls for plotting of experiment: {name}...")
            
            r1_values = np.load(path_r1, allow_pickle=True)
            r5_values = np.load(path_r5, allow_pickle=True)
            r10_values = np.load(path_r10, allow_pickle=True)
            r1p_values = np.load(path_r1p, allow_pickle=True)
            
            plt.figure(figsize=(12, 6))
            plt.plot(r1_values, label=f"{name} - R@1")
            plt.plot(r5_values, label=f"{name} - R@5")
            plt.plot(r10_values, label=f"{name} - R@10")
            plt.plot(r1p_values, label=f"{name} - R@1p")
            
            plt.xlabel("Epoch")
            plt.ylabel("Recalls [%]")
            plt.title(f"{name} - Evaluation recalls over epochs")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            log.info(f"{name} - Evaluation recalls plot salvato in: {save_path}")
        else:
            log.warning(f"File(s) non trovato per configurazione {name}")    

def load_insertion_deletion_results(eval_folder: str) -> dict:
    results_dict = {}
    
    if not os.path.exists(eval_folder):
        log.error(f"Evaluation folder not found: {eval_folder}")
        return results_dict
    
    # Iterate through experiment folders
    for exp_name in os.listdir(eval_folder):
        exp_path = os.path.join(eval_folder, exp_name)
        
        # Skip if not a directory
        if not os.path.isdir(exp_path):
            continue
        
        log.info(f"Loading results for experiment: {exp_name}")
        results_dict[exp_name] = {}
        
        # Look for deletion results
        deletion_path = os.path.join(exp_path, f"{exp_name}_deletion_results.npy")
        if os.path.exists(deletion_path):
            try:
                deletion_data = np.load(deletion_path, allow_pickle=True).item()
                results_dict[exp_name]['deletion'] = deletion_data
                log.info(f"  Loaded deletion results (AUC: {deletion_data['auc']:.4f})")
            except Exception as e:
                log.warning(f"  Failed to load deletion results: {e}")
        
        # Look for insertion results
        insertion_path = os.path.join(exp_path, f"{exp_name}_insertion_results.npy")
        if os.path.exists(insertion_path):
            try:
                insertion_data = np.load(insertion_path, allow_pickle=True).item()
                results_dict[exp_name]['insertion'] = insertion_data
                log.info(f"  Loaded insertion results (AUC: {insertion_data['auc']:.4f})")
            except Exception as e:
                log.warning(f"  Failed to load insertion results: {e}")
        
        # Remove experiment if no data was loaded
        if not results_dict[exp_name]:
            del results_dict[exp_name]
            log.warning(f"  No insertion/deletion results found for {exp_name}")
    
    return results_dict    


def plot_insertion_deletion_from_saved(
    eval_folder: str = None,
    save_path: str = None,
    figsize: tuple = (15, 6),
    marker_size: int = 3,
    dpi: int = 300
):
    # Set default paths
    if eval_folder is None:
        eval_folder = os.path.join("./container_files/logs", "EVALUATION")
    
    if save_path is None:
        save_path = os.path.join(
            "./container_files/plots",
            "ALL_experiments_insertion_deletion_from_saved.png"
        )
    
    log.info(f"Loading insertion/deletion results from: {eval_folder}")
    
    # Load all results
    results_dict = load_insertion_deletion_results(eval_folder)
    
    if not results_dict:
        log.error("No insertion/deletion results found. Cannot create plot.")
        return
    
    log.info(f"Found results for {len(results_dict)} experiment(s)")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Define colors for consistency
    colors = plt.cm.tab10(range(len(results_dict)))
    
    # Deletion plot
    deletion_count = 0
    for idx, (exp_name, results) in enumerate(sorted(results_dict.items())):
        if 'deletion' in results:
            del_res = results['deletion']
            ax1.plot(
                del_res['percentages'] * 100,
                del_res['accuracies'] * 100,
                label=f"{exp_name} (AUC={del_res['auc']:.3f})",
                marker='o',
                markersize=marker_size,
                color=colors[idx],
                linewidth=2
            )
            deletion_count += 1
    
    if deletion_count > 0:
        ax1.set_xlabel('Percentage of Pixels Removed (%)', fontsize=12)
        ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        ax1.set_title('Deletion Test', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(-2, 102)
        ax1.set_ylim(0, 100)
    else:
        ax1.text(0.5, 0.5, 'No deletion results found',
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_xlabel('Percentage of Pixels Removed (%)', fontsize=12)
        ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        ax1.set_title('Deletion Test', fontsize=14, fontweight='bold')
    
    # Insertion plot
    insertion_count = 0
    for idx, (exp_name, results) in enumerate(sorted(results_dict.items())):
        if 'insertion' in results:
            ins_res = results['insertion']
            ax2.plot(
                ins_res['percentages'] * 100,
                ins_res['accuracies'] * 100,
                label=f"{exp_name} (AUC={ins_res['auc']:.3f})",
                marker='o',
                markersize=marker_size,
                color=colors[idx],
                linewidth=2
            )
            insertion_count += 1
    
    if insertion_count > 0:
        ax2.set_xlabel('Percentage of Pixels Added (%)', fontsize=12)
        ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        ax2.set_title('Insertion Test', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(-2, 102)
        ax2.set_ylim(0, 100)
    else:
        ax2.text(0.5, 0.5, 'No insertion results found',
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_xlabel('Percentage of Pixels Added (%)', fontsize=12)
        ax2.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        ax2.set_title('Insertion Test', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    log.info(f"Insertion/Deletion comparison plot saved to: {save_path}")
    
    # Print summary
    log.info("\n" + "="*60)
    log.info("INSERTION/DELETION SUMMARY (from saved results)")
    log.info("="*60)
    for exp_name, results in sorted(results_dict.items()):
        log.info(f"\n{exp_name}:")
        if 'deletion' in results:
            log.info(f"  Deletion AUC: {results['deletion']['auc']:.4f}")
        if 'insertion' in results:
            log.info(f"  Insertion AUC: {results['insertion']['auc']:.4f}")
    log.info("="*60 + "\n")


    
def main():
    #create_gradcam_gif(type="GRD",glob_pattern="epoch*_iter*_grd_cam_*.png")
    #create_gradcam_gif(type="SAT",glob_pattern="epoch*_iter*_sat_cam_*.png")
    #create_gradcam_pair_gifs(expA="ATTENTION", expB="FULL",root_plots_dir=os.path.join(".", "container_files", "plots"),glob_pattern="epoch*_iter*_grd_cam_*.png",fps=2,type="GRD")
    #create_gradcam_pair_gifs(expA="ATTENTION", expB="FULL",root_plots_dir=os.path.join(".", "container_files", "plots"),glob_pattern="epoch*_iter*_sat_cam_*.png",fps=2,type="SAT")
    #create_gradcam_pair_gifs(expA="ATTENTION", expB="FULL",root_plots_dir=os.path.join(".", "container_files", "plots"),glob_pattern="epoch*_iter*_comp_grd_sat_cams_*.png",fps=2,type="GRD_SAT")
    # create_training_comparison_plots()
    # plot_training_iterative_loss()
    # plot_training_rolling_stats(window_size=50, title="Iterative Loss with Variance area")
    # create_training_comparison_plots_with_rolling_stats(window_size=50)
    # create_training_evaluation_comparison_plots_with_rolling_stats(window_size=3)
    #create_base_comparison_with_rolling_stats(window_size=50)
    # plot_training_iterative_recalls()
    # plot_training_iterative_recalls_with_rolling_stats(window_size=50)
    # plot_evaluation_epoch_recalls()
    # plot_training_recall_k_comparison_with_rolling_stats(1,window_size=70)
    # plot_training_recall_k_comparison_with_rolling_stats(5,window_size=70)
    # plot_training_recall_k_comparison_with_rolling_stats(10,window_size=70)
    # plot_evaluation_recall_k_comparison(1)
    # plot_evaluation_recall_k_comparison(5)
    # plot_evaluation_recall_k_comparison(10)
    plot_insertion_deletion_from_saved()
    
if __name__ == "__main__":
    main()