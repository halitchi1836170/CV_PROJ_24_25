from torchvision.models import VGG16_Weights
import logging

config = {
    "name": "MODEL CONFIGURATION",
    "epochs":1,
    "learning_rate":0.00002,
    "batch_size":32,
    "loss_weight":10.0,
    "train_grd_FOV": 360,
    "test_grd_FOV": 360,
    "dropout_ratio": 0.2,
    "no_layer_vgg_non_trainable": 9,
    "vgg_default_weights": VGG16_Weights.IMAGENET1K_V1,
    "train_grd_noise": 360,
    "log_frequency": 10,
    "save_cam_png_frequency":50,
    "seed":17
}

folders_and_files = {
    "log_file": "log.log",
    "saved_models_folder": "./models",
    "log_folder":"./logs",
    "plots_folder": "./plots",
}

data_loader_config = {
    "name": "DATA LOADER CONFIGURATION",
    "data_folder" : "./Data/",
    "train_list_file_name": "train-19zl.csv",
    "val_list_file_name": "val-19zl.csv",
}

images_params = {
    "name": "IMAGES CONFIGURATION",
    "max_width" : 512,
    "max_angle" : 360
}

header_length=100

logger_config = {
    "log_level":logging.DEBUG
}

gradcam_config = {
    "use_gradcam": True,
    "lambda_saliency": 0.5,
    "target_layer": "features.21",
    "apply_on": "ground",  # oppure "satellite"
    "save_plot_heatmaps": True,  # serve a salvare le heatmap ogni N iterazioni
}

sky_removal_config = {
    "remove_sky": True,
    "method": "deeplab",  # oppure "threshold"
}

EXPERIMENTS = {
    "BASE": {"use_attention": False, "remove_sky": False},
    #"ATTENTION": {"use_attention": True, "remove_sky": False},
    #"SKYREMOVAL": {"use_attention": False, "remove_sky": True},
    #"FULL": {"use_attention": True, "remove_sky": True},
}

experiments_config = {
    "remove_sky":"",
    "use_attention":"",
    "name":"",
    "logs_folder":"",
    "saved_models_folder":"",
    "plots_folder":"",
    "flag_save_ground_wo_sky": "",  # Flag to save ground image without sky,
    "epoch_for_save": ""
}