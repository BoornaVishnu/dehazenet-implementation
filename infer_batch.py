import os
import sys
import argparse
import random
from glob import glob

import torch
import torchvision.transforms as T
from PIL     import Image
from pathlib import Path

from models import DehazeNet
from utils  import ensure_dir, save_image_tensor, load_image, recover_image
from utils  import psnr, ssim
from utils  import guided_filter


def _list_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(folder, e)))
    return sorted(set(files))


def _make_pair_paths(hazy_path, clear_dir):
    """
    Map hazy file name XXXX_YY_ZZ.jpg -> GT XXXX.jpg in clear_dir.
    """

    base = os.path.basename(hazy_path)
    stem = Path(hazy_path).stem          # handles multiple dots safely
    prefix = stem.split("_", 1)[0]       # split only once

    cand_exts = [
        os.path.splitext(base)[1].lower(),
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    ]

    for ext in cand_exts:
        if not ext:
            continue
        gt = os.path.join(clear_dir, prefix + ext)
        if os.path.exists(gt):
            return gt
    return None


def _build_transform(img_size: int):
    if img_size and img_size > 0:
        def _resize_crop(pil_img):
            w, h = pil_img.size
            scale = img_size / min(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            pil_img = T.CenterCrop(img_size)(pil_img)
            return pil_img
        return _resize_crop
    return lambda x: x


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("DehazeNet batch inference + metrics")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("-i", required=False)
    parser.add_argument("--input_hazy", required=False)
    parser.add_argument("--input_clear", required=False)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("-n", "--num_images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=0)
    parser.add_argument("--t0", type=float, default=0.1)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    if args.seed:
        random.seed(args.seed)

    if not (args.i or not (args.input_hazy and args.input_clear)):
        if args.input_hazy:
            print(f'Both --input_hazy and --input_clear arguments are requried but only --input_hazy was provided')
        elif args.input_clear:
            print(f'Both --input_hazy and --input_clear arguments are requried but only --input_clear was provided')
        else:
            print(f'Either -i or (--input_hazy, --input_clear) arguments are requried')
        sys.exit()

    input_hazy_path  = os.path.join(args.i, "hazy") if args.i else args.input_hazy
    input_clear_path = os.path.join(args.i, "clear") if args.i else args.input_clear

    if not os.path.exists(input_hazy_path):
        print(f'Input image path {input_hazy_path} does not exist')
        sys.exit()

    if not os.path.exists(input_clear_path):
        print(f'Groundtruth image path {input_clear_path} does not exist')
        sys.exit()

    hazy_files = _list_images(input_hazy_path)
    if len(hazy_files) == 0:
        raise RuntimeError("No hazy images found")

    n = min(args.num_images, len(hazy_files))
    hazy_files = random.sample(hazy_files, n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DehazeNet().to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    preprocess = _build_transform(args.img_size)

    psnr_vals, ssim_vals = [], []

    for idx, hazy_path in enumerate(hazy_files, 1):
        gt_path = _make_pair_paths(hazy_path, input_clear_path)
        if gt_path is None:
            print(f"[WARN] GT not found for {os.path.basename(hazy_path)}")
            continue

        hazy = preprocess(load_image(hazy_path))
        gt = preprocess(load_image(gt_path))

        x = T.ToTensor()(hazy).unsqueeze(0).to(device)
        gt = T.ToTensor()(gt).unsqueeze(0).to(device)

        t_raw, beta = model(x)
        t_clipped = torch.clamp(t_raw, 0.0, 1.0)
        t = guided_filter(x, t_clipped, r=40, eps=1e-3)
        pred, _ = recover_image(x, t, beta, A=None, t0=args.t0)

        base = os.path.splitext(os.path.basename(hazy_path))[0]
        save_image_tensor(pred.squeeze(0).cpu(),
                          os.path.join(args.output_dir, f"{base}_dehaze.jpg"))

        tx3 = t.squeeze(0).cpu().repeat(3, 1, 1)
        save_image_tensor(tx3,
                          os.path.join(args.output_dir, f"{base}_txmap.jpg"))

        save_image_tensor(gt.squeeze(0).cpu(),
                          os.path.join(args.output_dir, f"{base}_gt.jpg"))

        save_image_tensor(x.squeeze(0).cpu(),
                          os.path.join(args.output_dir, f"{base}_hazy.jpg"))

        p = float(psnr(pred, gt).item())
        s = float(ssim(pred, gt).item())
        psnr_vals.append(p)
        ssim_vals.append(s)

        print(f"[{idx:03d}/{n:03d}] {base} | PSNR {p:.2f} SSIM {s:.4f}")

    if psnr_vals:
        print("\n=== Summary ===")
        print(f"Processed: {len(psnr_vals)}")
        print(f"Avg PSNR: {sum(psnr_vals) / len(psnr_vals):.2f} dB")
        print(f"Avg SSIM: {sum(ssim_vals) / len(ssim_vals):.4f}")


if __name__ == "__main__":
    main()


