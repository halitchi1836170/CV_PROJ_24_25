from torchvision.models import VGG16_Weights
import logging

config = {
    "name": "MODEL CONFIGURATION",
    "epochs":10,
    "learning_rate":0.0001,
    "batch_size":8,
    "loss_weight":10.0,
    "train_grd_FOV": 360,
    "test_grd_FOV": 0,
    "dropout_ratio": 0.2,
    "no_layer_vgg_non_trainable": 9,
    "vgg_default_weights": VGG16_Weights.IMAGENET1K_FEATURES,
    "train_grd_noise": 360,
    "log_frequency": 10
}

folders_and_files = {
    "log_file": "./log.log",
    "saved_models_folder": "./Models",
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
