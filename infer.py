import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image

from models import DehazeNet
from utils import ensure_dir, save_image_tensor, recover_image, guided_filter

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="dehazed_dehazenet.png")
    parser.add_argument("--img_size", type=int, default=0)
    parser.add_argument("--t0", type=float, default=0.1)
    args = parser.parse_args()

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model initialization
    model = DehazeNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Image loading and preprocessing
    img = Image.open(args.input).convert("RGB")
    if args.img_size > 0:
        w, h = img.size
        scale = args.img_size / min(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
        img = T.CenterCrop(args.img_size)(img)

    x = T.ToTensor()(img).unsqueeze(0).to(device)
    
    # Inference
    print("Running model inference...")
    t_raw, beta_from_model = model(x)
    
    t_clipped = torch.clamp(t_raw, 0.0, 1.0)
    t = guided_filter(x, t_clipped, r=40, eps=1e-3)
    
    # Define beta - Note: we use a scalar to avoid the memory explosion 
    # that happens with full-sized tensors in the physical model.
    beta = torch.tensor([0.0]).to(device) 
    
    # Image recovery (Dehazing)
    print("Recovering image...")
    pred, _ = recover_image(x, t, beta=beta, A=None, t0=args.t0)

    # Save logic
    out_dir = os.path.dirname(args.output)
    if out_dir:
        ensure_dir(out_dir)
        
    save_image_tensor(pred.squeeze(0), args.output)
    
    # Final confirmation prints
    print("-" * 40)
    print("PROCESS SUCCESSFUL!")
    print(f"Input:  {args.input}")
    print(f"Output: {os.path.abspath(args.output)}")
    print("-" * 40)

if __name__ == "__main__":
    main()