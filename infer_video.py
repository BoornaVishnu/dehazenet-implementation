import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from models import DehazeNet
from utils import ensure_dir, recover_image, guided_filter


def _build_transform(img_size: int):
    """Resize (keep aspect) + center-crop to img_size if img_size>0, else identity."""
    if img_size and img_size > 0:
        def _resize_crop(pil_img: Image.Image) -> Image.Image:
            w, h = pil_img.size
            scale = img_size / min(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
            pil_img = T.CenterCrop(img_size)(pil_img)
            return pil_img
        return _resize_crop
    return lambda x: x


def _open_video_io():
    """Prefer OpenCV; fall back to imageio if OpenCV isn't available."""
    try:
        import cv2
        return "cv2", cv2
    except Exception:
        try:
            import imageio.v3 as iio
            return "imageio", iio
        except Exception as e:
            raise ImportError("Need either opencv-python (cv2) or imageio installed for MP4 IO.") from e


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        "DehazeNet video inference: MP4 -> side-by-side MP4 (input | dehazed | txmap)"
    )
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--input_mp4", required=True, help="Input MP4 path")

    parser.add_argument("--output_dir", required=True, help="Directory to write output video")
    parser.add_argument(
        "--output_mp4",
        default="",
        help="Optional explicit output MP4 path (default: <output_dir>/<stem>_sbs.mp4)",
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=0,
        help="If >0, resize+center-crop frames to img_size (all panes use this size)",
    )
    parser.add_argument(
        "--t0",
        type=float,
        default=0.1,
        help="Lower bound for transmission during reconstruction",
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="If >0, process only first max_frames (after applying --every)",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Process every k-th frame (k>=1). Default 1 = all frames",
    )

    parser.add_argument(
        "--tx_colormap",
        default="gray",
        choices=["gray", "jet", "turbo", "magma", "viridis"],
        help="How to visualize transmission map pane",
    )

    args = parser.parse_args()

    in_path = Path(args.input_mp4)
    if not in_path.exists():
        raise FileNotFoundError(f"Input MP4 not found: {in_path}")

    ensure_dir(args.output_dir)

    stem = in_path.stem
    out_mp4 = Path(args.output_mp4) if args.output_mp4 else Path(args.output_dir) / f"{stem}_sbs.mp4"

    backend_name, backend = _open_video_io()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DehazeNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    preprocess = _build_transform(args.img_size)

    if backend_name == "cv2":
        cv2 = backend

        # colormap mapping
        cmap = None
        if args.tx_colormap != "gray":
            cmap_map = {
                "jet": cv2.COLORMAP_JET,
                "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
                "magma": getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET),
                "viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
            }
            cmap = cmap_map[args.tx_colormap]

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {in_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # pane size
        if args.img_size and args.img_size > 0:
            pane_w = pane_h = int(args.img_size)
        else:
            pane_w, pane_h = w, h

        # output size = 3 panes horizontally
        out_w, out_h = pane_w * 3, pane_h

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (out_w, out_h), True)

        frame_idx = 0
        written = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1

            if args.every > 1 and ((frame_idx - 1) % args.every != 0):
                continue
            if args.max_frames and written >= args.max_frames:
                break

            # BGR -> RGB PIL, apply same preprocessing used for model
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            pil = preprocess(pil)

            # Model input tensor
            x = T.ToTensor()(pil).unsqueeze(0).to(device)  # [1,3,H,W]

            # Predict transmission and recover
            t_raw, beta = model(x)
            t_clipped = torch.clamp(t_raw, 0.0, 1.0)  # [1,1,H,W]
            t = guided_filter(x, t_clipped, r=40, eps=1e-3)
            pred, _ = recover_image(x, t, beta=beta, A=None, t0=args.t0)

            # Pane 1: input (after preprocessing, for perfect alignment)
            inp_u8 = (x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).round().clip(0, 255).astype("uint8")
            inp_bgr = cv2.cvtColor(inp_u8, cv2.COLOR_RGB2BGR)

            # Pane 2: dehazed
            pred_u8 = (pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).round().clip(0, 255).astype("uint8")
            pred_bgr = cv2.cvtColor(pred_u8, cv2.COLOR_RGB2BGR)

            # Pane 3: txmap visualization
            tx = t.squeeze(0).squeeze(0).detach().cpu().numpy()  # [H,W]
            tx_u8 = (tx * 255.0).round().clip(0, 255).astype("uint8")
            if cmap is None:
                tx_bgr = cv2.cvtColor(tx_u8, cv2.COLOR_GRAY2BGR)
            else:
                tx_bgr = cv2.applyColorMap(tx_u8, cmap)

            # Stitch horizontally: input | dehazed | txmap
            sbs = cv2.hconcat([inp_bgr, pred_bgr, tx_bgr])
            writer.write(sbs)

            written += 1
            if written % 50 == 0:
                print(f"Processed {written} frames...")

        cap.release()
        writer.release()

    else:
        # imageio fallback: reads frames iterator; stores output frames then writes (memory heavy for long videos)
        iio = backend
        frames_iter = iio.imiter(str(in_path))

        out_frames = []
        written = 0

        # colormap in imageio path (simple grayscale only)
        if args.tx_colormap != "gray":
            print("[WARN] imageio fallback supports only gray txmap visualization; using gray.")

        import numpy as np

        for i, frame in enumerate(frames_iter):
            if args.every > 1 and (i % args.every != 0):
                continue
            if args.max_frames and written >= args.max_frames:
                break

            pil = Image.fromarray(frame)
            pil = preprocess(pil)
            x = T.ToTensor()(pil).unsqueeze(0).to(device)

            t_raw, beta = model(x)
            t_clipped = torch.clamp(t_raw, 0.0, 1.0)  # [1,1,H,W]
            t = guided_filter(x, t_clipped, r=40, eps=1e-3)
            pred, _ = recover_image(x, t, A=None, t0=args.t0)

            inp_u8 = (x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).round().clip(0, 255).astype("uint8")
            pred_u8 = (pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0).round().clip(0, 255).astype("uint8")
            tx = t.squeeze(0).squeeze(0).detach().cpu().numpy()
            tx_u8 = (tx * 255.0).round().clip(0, 255).astype("uint8")
            tx_u8_rgb = np.stack([tx_u8, tx_u8, tx_u8], axis=-1)

            sbs = np.concatenate([inp_u8, pred_u8, tx_u8_rgb], axis=1)
            out_frames.append(sbs)

            written += 1
            if written % 50 == 0:
                print(f"Processed {written} frames...")

        iio.imwrite(str(out_mp4), out_frames, fps=30)

    print("Saved:")
    print(" -", out_mp4)


if __name__ == "__main__":
    main()