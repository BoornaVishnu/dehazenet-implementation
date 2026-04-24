import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class PerceptualVGG16(nn.Module):
    """
    VGG16 feature extractor for perceptual loss.
    Uses early layers suitable for image restoration.
    """
    def __init__(self, layers=(3, 8, 15), requires_grad=False):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features

        self.blocks = nn.ModuleList()
        prev = 0
        for l in layers:
            self.blocks.append(nn.Sequential(*vgg[prev:l]))
            prev = l

        if not requires_grad:
            for p in self.parameters():
                p.requires_grad_(False)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class DehazeLoss(nn.Module):
    """
    Combined L1 + optional perceptual loss for dehazing.
    """
    def __init__(self, use_perceptual=True, use_dcp = True, l1_weight=1.0, perceptual_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.use_perceptual = use_perceptual
        self.use_dcp = use_dcp
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.vgg = PerceptualVGG16() if use_perceptual else None

    def dark_channel(self, img):
        return img.min(dim=1, keepdim=True).values

    def forward(self, pred, target, t = None, beta = None):
        loss = self.l1_weight * self.l1(pred, target)

        if self.use_perceptual:
            pf = self.vgg(pred)
            tf = self.vgg(target)
            p_loss = 0.0
            for a, b in zip(pf, tf):
                p_loss += F.l1_loss(a, b)
            loss = loss + self.perceptual_weight * p_loss

        if t != None:
            # Regularization on t
            loss_t = torch.mean(t ** 2)
            loss = loss + 0.1 * loss_t

        if beta != None:
            # Regularization on brightness
            loss_beta = torch.mean(beta ** 2)
            loss = loss + 0.1 * loss_beta

        return loss