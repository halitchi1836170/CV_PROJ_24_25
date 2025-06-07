import argparse
import torch.optim
from config import *
from logger import log
from data_loader import InputData
from utils import *
from model import GroundToAerialMatchingModel

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
        log.info(get_header_title("TRAINING MODE"))

        #INSTANTIATION OF CONSTANTS FROM CONFIGURATION FILE
        train_grd_FOV = config["train_grd_FOV"]
        max_angle = images_params["max_angle"]
        max_width = images_params["max_width"]
        learning_rate = config["learning_rate"]


        width = int(train_grd_FOV / max_angle * max_width)

        #DEFINITION OF THE MODEL
        model = GroundToAerialMatchingModel()

        #DEFINITION OF THE OPTIMIZER
        log.info("Definition of optimizer...\n")
        #optimizer = torch.optim.Adam(lr=learning_rate)





    elif args_mode == "TEST":
        log.info(get_header_title("TEST MODE"))
        log.info("Loading test data...")





if __name__ == "__main__":
    main()