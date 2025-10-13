# sky_removal.py

import os
import torch
import numpy as np
import cv2

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")
from detectron2.projects.deeplab import add_deeplab_config
from Mask2Former.mask2former import add_maskformer2_config
from logger import log
from Globals import experiments_config

# Setup Mask2Former predictor
_cfg = get_cfg()
add_deeplab_config(_cfg)
add_maskformer2_config(_cfg)
_cfg.merge_from_file("Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
_cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
_cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
_cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
_cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
#_cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_predictor = DefaultPredictor(_cfg)


def remove_sky_from_image(image: np.ndarray,index_for_save) -> np.ndarray:
    """
    Rimuove il cielo da un'immagine sostituendolo con pixel 0
    """
    #img_input = (image * 255).astype(np.uint8)
    img_input = image.copy().astype(np.uint8)  # Assicurati che l'immagine sia in formato uint8
    #image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    #image_bgr = cv2.cvtColor(image, cv2.COLOR_BAYER_BGGR2RGB)

    outputs = _predictor(img_input)
    
    #log.info("Output keys: %s", outputs.keys())
    
    panoptic_seg, segments_info = outputs["panoptic_seg"]
    stuff_classes = coco_metadata.stuff_classes
    sky_class_index = stuff_classes.index("sky-other-merged")
    sky_segments = [s for s in segments_info if s["category_id"] == sky_class_index and not s["isthing"]]
    
    sky_mask = torch.zeros_like(panoptic_seg, dtype=torch.bool)
    for seg in sky_segments:
        sky_mask |= (panoptic_seg == seg["id"])

    im_no_sky = img_input.copy()
    im_no_sky[sky_mask.cpu().numpy()] = 0  # imposta a nero
    
    if experiments_config["flag_save_ground_wo_sky"]:
        log.debug("Saving ground image without sky...")
        cv2.imwrite(f"{experiments_config['plots_folder']}/ground_wo_sky_epoch_{experiments_config['epoch_for_save']}{index_for_save}.jpg", im_no_sky)
        experiments_config["flag_save_ground_wo_sky"] = False
    
    return im_no_sky

def batch_remove_sky(images: np.ndarray) -> np.ndarray:
    """
    Applica remove_sky_from_image a un batch (array [B,H,W,C])
    """
    return np.stack([remove_sky_from_image(im) for im in images], axis=0)
