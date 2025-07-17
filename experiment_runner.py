import os
import shutil
from main import main as train_main
from config import config,experiments_config,EXPERIMENTS  # Usa un dizionario modificabile
from logger import log
from utils import get_header_title

def run_experiment(name, use_attention, remove_sky):
    log.info(get_header_title(f"STARTING EXPERIMENT: {name}"))
    experiments_config["use_attention"] = use_attention
    experiments_config["remove_sky"] = remove_sky
    experiments_config["name"] = name

    # Setup output directories
    os.makedirs(f"logs/{name}", exist_ok=True)
    os.makedirs(f"models/{name}", exist_ok=True)
    experiments_config["logs_folder"] = f"logs/{name}"
    experiments_config["saved_models_folder"] = f"models/{name}"

    train_main()


def run_all():
    for name, params in EXPERIMENTS.items():
        run_experiment(name, **params)

if __name__ == "__main__":
    run_all()
