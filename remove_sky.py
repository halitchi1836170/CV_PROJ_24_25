import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights, deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights


# --- HSV Threshold Method ---
def remove_sky_hsv(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_sky = np.array([90, 0, 120])     # H, S, V
    upper_sky = np.array([140, 60, 255])
    mask = cv2.inRange(hsv, lower_sky, upper_sky)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image_np, image_np, mask=mask_inv)
    return result

# --- DeepLabV3 Method ---
def remove_sky_deeplab(image_np):
    if deeplabv3_resnet50 is None: raise ImportError("torchvision not compiled with DeepLabV3")

    #model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)

    model.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_np.shape[0], image_np.shape[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image_np).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()

    SKY_CLASS = 7  # COCO class for 'sky'
    sky_mask = (mask == SKY_CLASS).astype(np.uint8) * 255
    sky_mask_inv = cv2.bitwise_not(sky_mask)
    result = cv2.bitwise_and(image_np, image_np, mask=sky_mask_inv)
    return result

# --- Dispatcher ---
def remove_sky(image_np, method="deeplab"):
    if method == "hsv":
        return remove_sky_hsv(image_np)
    elif method == "deeplab":
        return remove_sky_deeplab(image_np)
    else:
        raise ValueError("Unsupported method: choose 'hsv' or 'deeplab'")