import argparse
from Train import train_model
from Evaluation import test_model
from Globals import EXPERIMENTS
from logger import log
from Utils import get_header_title
from post_analysis import plot_loss_curves as create_comparison_plots

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
                create_comparison_plots()
        else:
            train_model("FULL", EXPERIMENTS["FULL"])

    elif args.mode == "TEST":
        if not args.model_path:
            raise ValueError("In TEST mode, --model_path must be specified.")
        test_model(args.model_path)

if __name__ == "__main__":
    main()