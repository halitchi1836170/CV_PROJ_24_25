from torchvision.models import VGG16_Weights
import logging
import numpy as np

config = {
    "name": "MODEL CONFIGURATION",
    "epochs":1,
    "learning_rate":0.0001,
    "fin_tuning_learning_rate":0.000001,
    "batch_size":8,
    "loss_weight":10.0,
    "train_grd_FOV": 360,
    "test_grd_FOV": 360,
    "dropout_ratio": 0.2,
    "no_layer_vgg_non_trainable": 9,
    "vgg_default_weights": VGG16_Weights.IMAGENET1K_V1,
    "train_grd_noise": 360,
    "log_frequency": 30,
    "save_cam_png_frequency":150,
    "seed":17,
    "accumulation_steps":4
}

previous_models={
    "flag_use_last_checkpoint_BASE":False,
    "last_checkpoint_path_BASE":"./models/BASE/epoch20/model.pth",
    "flag_use_last_checkpoint_ATTENTION":False,
    "last_checkpoint_path_ATTENTION":"./models/ATTENTION/epoch32/model.pth",
    "flag_use_last_checkpoint_SKYREMOVAL":False,
    "last_checkpoint_path_SKYREMOVAL":"",
    "flag_use_last_checkpoint_FULL":False,
    "last_checkpoint_path_FULL":""
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
    "list_grounds_tbt":["0000029","0001124","0003157", "0017660", "0031663","0032354","0006134","0010078","0032488","0011435"]
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
    #"ATTENTION": {"use_attention": True, "remove_sky": False},
    "BASE": {"use_attention": False, "remove_sky": False},
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
    "epoch_for_save": "",
    "index_for_name": {""}
}

BLOCKING_COUNTER=-3