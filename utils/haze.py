import torch

def estimate_atmospheric_light(img, topk=0.001):
    """
    Estimate A from hazy image (DCP-style heuristic).
    img: [B,3,H,W] in [0,1]
    returns A: [B,3,1,1]
    """
    B, C, H, W = img.shape
    # dark channel
    dark = img.min(dim=1, keepdim=True).values  # [B,1,H,W]
    flat = dark.view(B, -1)
    k = max(1, int(topk * flat.shape[1]))
    _, idx = torch.topk(flat, k=k, dim=1, largest=True)

    img_flat = img.view(B, C, -1)  # [B,3,HW]
    A = []
    for b in range(B):
        pixels = img_flat[b, :, idx[b]]  # [3,k]
        # choose the brightest (max sum) among selected pixels
        s = pixels.sum(dim=0)
        j = torch.argmax(s)
        A.append(pixels[:, j])
    A = torch.stack(A, dim=0).view(B, C, 1, 1)
    return A

def recover_image(hazy, t_raw, beta, A=None, t0=0.1):
    """
    Recover dehazed image using atmospheric scattering model.
    hazy: [B,3,H,W], t_raw: [B,1,H,W], beta: bias/correction factor
    A: [B,3,1,1] if provided, else estimated.
    """

    if A is None:
        A = estimate_atmospheric_light(hazy).detach()

    # Transmission map processing
    t = torch.clamp(t_raw, 0.0, 1.0)
    t = t0 + (1.0 - t0) * torch.sigmoid(t_raw)

    # Basic Atmospheric Scattering Model: J = (I - A)/t + A
    J = (hazy - A) / t + A

    # --- SAFETY FIX FOR MEMORY CRASH ---
    # We ensure beta is treated as a compatible broadcasting shape.
    # If beta is accidentally a large tensor, we reduce it to a scalar or mean.
    if isinstance(beta, torch.Tensor):
        if beta.numel() > 3:
            # If beta is a full-sized map, reduce it to avoid [B,C,H,W] * [B,C,H,W] 
            # expansion errors if shapes don't align perfectly.
            v_beta = beta.mean().view(1, 1, 1, 1)
        elif beta.dim() == 1 and beta.shape[0] == 3:
            v_beta = beta.view(-1, 3, 1, 1)
        else:
            v_beta = beta.view(-1, 1, 1, 1)
    else:
        # If beta is just a float/int
        v_beta = beta

    # Final reconstruction with clamp to prevent pixel overflow
    pred = torch.clamp(J + v_beta, 0.0, 1.0)

    return pred, A