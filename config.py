config = {
    "name": "MODEL CONFIGURATION",
    "epochs":10,
    "lr":0.0001,
    "batch_size":8,
    "loss_weight":10.0,
    "train_grd_FOV": 360,
    "test_grd_FOV": 0
}

folders_and_files = {
    "log_file": "./log.log"
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