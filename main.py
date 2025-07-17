import argparse
import os
import torch.optim
import torch.nn.functional as F
from torch import optim
from config import *
from logger import log
from data_loader import InputData
from utils import *
from model import GroundToAerialMatchingModel
import numpy as np
from model import compute_triplet_loss
from gradcam import *

def main():
    log.info("Starting project...")

    log.info(get_header_title(f"PARSING INPUT ARGUMENTS"))
    parser = argparse.ArgumentParser(description="Train or evaluate a model on noisy labeled datasets.")
    parser.add_argument("--mode",type=str,help="Training or Testing mode",)
    parser.add_argument("--starting_epoch", type=int, help="Epoch reference to use when loading the pre-trained model")
    args = parser.parse_args()
    log.info(get_header_title("END", new_line=True))

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
        log.info(get_header_title("END", new_line=True))

        log.info(get_header_title("CORRELATION MATRICES"))
        log.info("Calculating correlation matrices...")
        sat_matrix, grd_matrix, distance, pred_orien = model(grd_x, polar_sat_x, segmap_x)
        s_height, s_width, s_channel = list(sat_matrix.size())[1:]
        g_height, g_width, g_channel = list(grd_matrix.size())[1:]
        sat_global_matrix = np.zeros([input_data.get_test_dataset_size(), s_height, s_width, s_channel])
        log.info(f"Global satellite matrix dimensions: {sat_global_matrix.shape}")
        grd_global_matrix = np.zeros([input_data.get_test_dataset_size(), g_height, g_width, g_channel])
        log.info(f"Global ground matrix dimensions: {grd_global_matrix.shape}")
        orientation_gth = np.zeros([input_data.get_test_dataset_size()])
        log.info(f"Orientation matrix dimensions: {orientation_gth.shape}")
        log.info(get_header_title("END", new_line=True))

        log.info(get_header_title("STARTING TRAINING"))

        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Using device: {device}")

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])  # Adatta il learning rate
        log.info(f"Optimizer defined for device: {device}")

        # load a Model
        if args_starting_epoch != 0:
            model_path = f"{folders_and_files['saved_models_folder']}/{args_starting_epoch}/model.pth"
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                log.info("Model checkpoint uploaded")

        number_of_epoch = config["epochs"]
        # Iterate over the desired number of epochs
        for epoch in range(args_starting_epoch, args_starting_epoch + number_of_epoch):
            log.info(f"Epoch {epoch + 1}/{args_starting_epoch + number_of_epoch}")

            model.train()
            iter_count = 0
            end = False

            while True:
                total_loss = 0
                # Gradient accumulation (batch=8, 4 iterations => total batch=32)
                optimizer.zero_grad()

                for i in range(4):
                    # Ottieni batch di dati
                    batch_sat_polar, batch_sat, batch_grd, batch_segmap, batch_orien = input_data.next_pair_batch(8, grd_noise=config["train_grd_noise"], FOV=train_grd_FOV)

                    if batch_sat is None:
                        log.info("Satellite batch is None, breaking...")
                        end = True
                        break

                    # Converti in tensori PyTorch
                    batch_grd = torch.from_numpy(batch_grd).float().to(device)
                    batch_sat_polar = torch.from_numpy(batch_sat_polar).float().to(device)
                    batch_segmap = torch.from_numpy(batch_segmap).float().to(device)

                    # Forward pass
                    grd_features, sat_features, segmap_features = model(batch_grd, batch_sat_polar, batch_segmap,return_features=True)

                    # L2 normalization
                    norm = torch.norm(grd_features, p=2, dim=[1, 2, 3], keepdim=True)
                    grd_features = grd_features / (norm + 1e-8)

                    # Concatenation of satellite features and segmentation mask features
                    sat_features = torch.concat([sat_features, segmap_features], dim=3)

                    # Compute correlation and distance matrix
                    sat_matrix, grd_matrix, distance, orien = model(batch_grd, batch_sat_polar, batch_segmap)

                    # Compute the loss
                    loss_value = compute_triplet_loss(distance)
                    loss_value = loss_value / 4  # Divide by accumulation steps

                    if gradcam_config["use_gradcam"]:
                        gradcam = GradCAM(model.ground_branch, gradcam_config["target_layer"])
                        loss_value.backward(retain_graph=True)

                        cam_size = (batch_grd.shape[1], batch_grd.shape[2])
                        saliency_loss = compute_saliency_loss(gradcam, batch_grd, cam_size)

                        total_loss = loss_value + gradcam_config["lambda_saliency"] * saliency_loss
                        total_loss.backward()
                    else:
                        loss_value.backward()

                    total_loss += loss_value.item() * 4  # Scale back for logging

                if end:
                    break

                # Update weights dopo gradient accumulation
                optimizer.step()

                if iter_count % config["log_frequency"] == 0:
                    log.info(f"ITERATION: {iter_count}, LOSS VALUE: {loss_value.item() * 4:.6f}, TOTAL LOSS: {total_loss:.6f}")
                    if gradcam_config["save_plot_heatmaps"]:
                        heatmap_np = gradcam.generate(cam_size)
                        save_overlay_image(batch_grd, heatmap_np, path=f"plots/epoch{epoch}_iter{iter_count}_cam.png")

                iter_count += 1

            # Save the model
            model_path = f"{folders_and_files['saved_models_folder']}/{epoch}/model.pth"
            os.makedirs(model_path, exist_ok=True)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }
            torch.save(checkpoint, os.path.join(model_path, 'model.pth'))

        log.info(get_header_title("END", new_line=True))





    elif args_mode == "TEST":
        log.info(get_header_title("TEST MODE"))
        log.info("Loading test data...")





if __name__ == "__main__":
    main()