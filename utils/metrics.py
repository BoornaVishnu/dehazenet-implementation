import torch
import torch.nn.functional as F
import math

def psnr(pred, target, eps=1e-8):
    """
    pred/target: [B,3,H,W] in [0,1]
    """
    mse = torch.mean((pred - target) ** 2, dim=(1,2,3))
    psnr_val = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr_val.mean()

def _gaussian_window(window_size=11, sigma=1.5, device="cpu", channel=3):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    w = g[:, None] * g[None, :]
    w = w.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    w = w.repeat(channel, 1, 1, 1)   # [C,1,H,W]
    return w

def ssim(pred, target, window_size=11, sigma=1.5, data_range=1.0, k1=0.01, k2=0.03):
    """
    Simplified SSIM (per-image average, multi-channel).
    pred/target: [B,C,H,W] in [0,1]
    """
    device = pred.device
    C = pred.size(1)
    window = _gaussian_window(window_size, sigma, device=device, channel=C)

    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=C) - mu12

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()