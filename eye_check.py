# eye_check.py
from PIL import Image
import numpy as np

def is_eye_image(pil_img, sat_thresh=25, brightness_thresh=230):
    """
    Quick heuristic check if an image resembles a retinal (fundus) photo.
    Checks color saturation and brightness in the central area.
    """
    img = pil_img.convert("RGB")
    w, h = img.size
    left, top = int(0.2 * w), int(0.2 * h)
    right, bottom = int(0.8 * w), int(0.8 * h)
    crop = img.crop((left, top, right, bottom))
    hsv = np.array(crop.convert("HSV"))
    sat_mean = hsv[..., 1].mean()
    val_mean = hsv[..., 2].mean()

    # Fundus images are generally moderately saturated and not too bright
    return sat_mean > sat_thresh and val_mean < brightness_thresh
