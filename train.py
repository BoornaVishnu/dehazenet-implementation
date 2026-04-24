import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import DehazeNet
from data import PairedDehazeDataset
from utils import psnr, ssim, DehazeLoss, ensure_dir, recover_image, guided_filter

def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, optim, loss_fn, device, scaler=None, t0=0.1):
    model.train()
    total_loss = total_psnr = total_ssim = 0.0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        hazy = batch["hazy"].to(device)
        clear = batch["clear"].to(device)

        optim.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                t_raw, beta = model(hazy)
                with torch.no_grad():
                    t = guided_filter(hazy, t_raw, r=40, eps=1e-3)
                pred, _ = recover_image(hazy, t, beta=beta, A=None, t0=t0)
                loss = loss_fn(pred, clear, t)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            t_raw, beta = model(hazy)
            with torch.no_grad():
                t = guided_filter(hazy, t_raw, r=40, eps=1e-3)
            pred, _ = recover_image(hazy, t, beta=beta, A=None, t0=t0)
            loss = loss_fn(pred, clear, t)
            loss.backward()
            optim.step()

        with torch.no_grad():
            total_loss += loss.item()
            total_psnr += psnr(pred, clear).item()
            total_ssim += ssim(pred, clear).item()

        pbar.set_postfix(loss=loss.item())

    n = len(loader)
    return total_loss / n, total_psnr / n, total_ssim / n

@torch.no_grad()
def validate(model, loader, loss_fn, device, t0=0.1):
    model.eval()
    total_loss = total_psnr = total_ssim = 0.0

    for batch in tqdm(loader, desc="val", leave=False):
        hazy = batch["hazy"].to(device)
        clear = batch["clear"].to(device)

        t_raw, beta = model(hazy)
        t_clipped = torch.clamp(t_raw, 0.0, 1.0)
        t = guided_filter(hazy, t_clipped, r=40, eps=1e-3)
        pred, _ = recover_image(hazy, t, beta = beta, A=None, t0=t0)

        loss = loss_fn(pred, clear, t_raw)
        total_loss += loss.item()
        total_psnr += psnr(pred, clear).item()
        total_ssim += ssim(pred, clear).item()

    n = len(loader)
    return total_loss / n, total_psnr / n, total_ssim / n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="runs/dehazenet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_perceptual", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--t0", type=float, default=0.1, help="Lower bound on transmission in recovery")
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    ensure_dir(ckpt_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PairedDehazeDataset(args.data_root, split="train",
                                   val_ratio=args.val_ratio, seed=args.seed,
                                   size=args.img_size, random_crop=True)
    val_ds = PairedDehazeDataset(args.data_root, split="val",
                                 val_ratio=args.val_ratio, seed=args.seed,
                                 size=args.img_size, random_crop=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = DehazeNet().to(device)
    loss_fn = DehazeLoss(use_perceptual=(not args.no_perceptual)).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    best_val_psnr = -1.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_psnr, tr_ssim = train_one_epoch(model, train_loader, optim, loss_fn, device, scaler, t0=args.t0)
        va_loss, va_psnr, va_ssim = validate(model, val_loader, loss_fn, device, t0=args.t0)
        scheduler.step()

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/val", va_loss, epoch)
        writer.add_scalar("psnr/train", tr_psnr, epoch)
        writer.add_scalar("psnr/val", va_psnr, epoch)
        writer.add_scalar("ssim/train", tr_ssim, epoch)
        writer.add_scalar("ssim/val", va_ssim, epoch)
        writer.add_scalar("lr", optim.param_groups[0]["lr"], epoch)

        print(f"[Epoch {epoch:03d}] "
              f"train loss {tr_loss:.4f} psnr {tr_psnr:.2f} ssim {tr_ssim:.4f} | "
              f"val loss {va_loss:.4f} psnr {va_psnr:.2f} ssim {va_ssim:.4f}")

        # Save last
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optim": optim.state_dict(), "scheduler": scheduler.state_dict(),
                    "best_val_psnr": best_val_psnr}, os.path.join(ckpt_dir, "last.pt"))

        # Save best
        if va_psnr > best_val_psnr:
            best_val_psnr = va_psnr
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optim": optim.state_dict(), "scheduler": scheduler.state_dict(),
                        "best_val_psnr": best_val_psnr}, os.path.join(ckpt_dir, "best.pt"))

    writer.close()
    print("Training complete. Best val PSNR:", best_val_psnr)

if __name__ == "__main__":
    main()