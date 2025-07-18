import os
import shutil
from main import main as train_main
from Globals import experiments_config,EXPERIMENTS,folders_and_files  # Usa un dizionario modificabile
from logger import log
from Utils import get_header_title
import logging

def run_experiment(name, use_attention, remove_sky):
    
    log.info(get_header_title(f"STARTING EXPERIMENT: {name}"))
    experiments_config["use_attention"] = use_attention
    experiments_config["remove_sky"] = remove_sky
    experiments_config["name"] = name

    # Setup output directories
    os.makedirs(f"{folders_and_files['log_folder']}/{name}", exist_ok=True)
    os.makedirs(f"{folders_and_files['saved_models_folder']}/{name}", exist_ok=True)
    
    streamFile = logging.FileHandler(filename=f"{folders_and_files['log_folder']}/{name}/{folders_and_files['log_file']}", mode="w", encoding="utf-8")
    streamFile.setLevel(logging.DEBUG)
    log.addHandler(streamFile)
    
    experiments_config["logs_folder"] = f"{folders_and_files['log_folder']}/{name}"
    experiments_config["saved_models_folder"] = f"{folders_and_files['saved_models_folder']}/{name}"

    train_main()


def run_all():
    for name, params in EXPERIMENTS.items():
        run_experiment(name, **params)

if __name__ == "__main__":
    run_all()
