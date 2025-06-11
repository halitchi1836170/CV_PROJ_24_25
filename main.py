import argparse
import torch.optim
from config import *
from logger import log
from data_loader import InputData
from utils import *
from model import GroundToAerialMatchingModel
import numpy as np

def main():
    log.info("Starting project...")

    parser = argparse.ArgumentParser(description="Train or evaluate a model on noisy labeled datasets.")
    parser.add_argument("--mode",type=str,help="Training or Testing mode",)
    parser.add_argument("--starting_epoch", type=int, help="Epoch reference to use when loading the pre-trained model")
    args = parser.parse_args()

    args_mode = "TRAIN" if args.mode is None else args.mode
    args_starting_epoch = 0 if args.starting_epoch is None else args.starting_epoch

    log.info("Args received...\n")

    log.info(get_header_title(f"PARAMETERS OF : {config['name']}"))
    print_params(config)
    log.info(get_header_title("END",new_line=True))

    log.info(get_header_title("LOADING DATASET"))
    input_data = InputData()
    log.info(get_header_title("END",new_line=True))

    if args_mode == "TRAIN":
        log.info(get_header_title("INSTANTIATION OF THE MODEL"))

        #INSTANTIATION OF CONSTANTS FROM CONFIGURATION FILE
        train_grd_FOV = config["train_grd_FOV"]
        max_angle = images_params["max_angle"]
        max_width = images_params["max_width"]
        learning_rate = config["learning_rate"]

        width = int(train_grd_FOV / max_angle * max_width)

        #DEFINITION OF THE MODEL
        log.info("Creation of the model...")
        model = GroundToAerialMatchingModel()
        log.info("Model created, summary: ")
        get_model_summary_simple(model)
        log.info(get_header_title("END", new_line=True))

        log.info(get_header_title("DEFINITION OF THE INPUT DATA"))
        grd_x = torch.zeros([2, int(max_width/4), width,3])                             #ORDINE CAMBIATO: B (batch size)-C (channels)-H-W
        sat_x = torch.zeros([2, int(max_width/2), max_width,3])
        polar_sat_x = torch.zeros([2, int(max_width/4), max_width,3])
        segmap_x = torch.zeros([2, int(max_width/4), max_width,3])
        log.info(f"Ground (zero) input matrix dimension: {grd_x.shape}")
        log.info(f"Polar satellite (zero) input matrix dimension: {polar_sat_x.shape}")
        log.info(f"Segmentation (zero) input matrix dimension: {segmap_x.shape}")
        log.info(get_header_title("END", new_line=True))

        log.info(get_header_title("FIRST FEATURES"))
        log.info("Calculating (forward pass) first features (zeros as input) ...")
        grd_features, sat_features, segmap_features = model(grd_x, polar_sat_x, segmap_x, return_features=True)
        log.info(f"Ground features (zero input) matrix dimension: {grd_features.shape}")
        log.info(f"Polar satellite features (zero input) matrix dimension: {sat_features.shape}")
        log.info(f"Segmentation features (zero input) matrix dimension: {segmap_features.shape}")
        sat_features = torch.concat([sat_features, segmap_features], dim=3)
        log.info(f"Concatenated Satellite and Segmentation features (zero input) matrix dimension: {sat_features.shape}")


        #DEFINITION OF THE OPTIMIZER
        log.info("Definition of optimizer...\n")
        #optimizer = torch.optim.Adam(lr=learning_rate)





    elif args_mode == "TEST":
        log.info(get_header_title("TEST MODE"))
        log.info("Loading test data...")





if __name__ == "__main__":
    main()