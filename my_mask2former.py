import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="my_mask2former")

import numpy as np
import cv2
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

from Mask2Former.mask2former import add_maskformer2_config

im = cv2.imread("./panorama.jpg")
cv2.imshow("image", im)