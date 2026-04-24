import torch
import torch.nn.functional as F


def box_filter(x, r):
    """
    Fast O(1) box filter using cumulative sum.
    x: [B, C, H, W]
    r: radius
    """
    return F.avg_pool2d(x, kernel_size=2*r+1, stride=1, padding=r)


def guided_filter(I, p, r=40, eps=1e-3):
    """
    Guided Filter (He et al.)

    I: guidance image   [B, 3, H, W] in [0,1]
    p: filtering input  [B, 1, H, W] (e.g., transmission map)
    r: window radius
    eps: regularization

    returns:
    q: filtered map     [B, 1, H, W]
    """
    B, C, H, W = I.shape

    # convert I to grayscale guidance
    I_gray = (
        0.2989 * I[:, 0:1] +
        0.5870 * I[:, 1:2] +
        0.1140 * I[:, 2:3]
    )

    mean_I = box_filter(I_gray, r)
    mean_p = box_filter(p, r)
    mean_Ip = box_filter(I_gray * p, r)

    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = box_filter(I_gray * I_gray, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)

    q = mean_a * I_gray + mean_b
    return q
