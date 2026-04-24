import os
import torch
from PIL import Image
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_image_tensor(t, path):
    """
    t: [3,H,W] in [0,1]
    """
    t = torch.clamp(t, 0, 1).detach().cpu().numpy()
    t = (t.transpose(1,2,0) * 255.0).astype(np.uint8)
    Image.fromarray(t).save(path)

def load_image(path):
    img = Image.open(path).convert("RGB")
    return img