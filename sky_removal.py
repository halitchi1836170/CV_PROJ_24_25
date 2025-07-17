# sky_removal.py

import os
import torch
import numpy as np
import cv2

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from Mask2Former.mask2former import add_maskformer2_config

# Setup Mask2Former predictor
_cfg = get_cfg()
add_deeplab_config(_cfg)
add_maskformer2_config(_cfg)
_cfg.merge_from_file("Mask2Former/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
_cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
_cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
_cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
_cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False

_predictor = DefaultPredictor(_cfg)

# Semantic label index for sky in COCO
SKY_CLASS_INDEX = 0  # Check label mapping if needed

def remove_sky_from_image(image: np.ndarray) -> np.ndarray:
    """
    Rimuove il cielo da un'immagine sostituendolo con pixel 0
    """
    outputs = _predictor(image)
    semantic = outputs["sem_seg"].argmax(dim=0).cpu().numpy()
    mask = semantic == SKY_CLASS_INDEX
    image[mask] = 0
    return image

def batch_remove_sky(images: np.ndarray) -> np.ndarray:
    """
    Applica remove_sky_from_image a un batch (array [B,H,W,C])
    """
    return np.stack([remove_sky_from_image(im) for im in images], axis=0)
