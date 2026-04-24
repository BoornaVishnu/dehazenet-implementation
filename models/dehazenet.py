import torch
import torch.nn as nn
import torch.nn.functional as F

class Maxout(nn.Module):
    """
    Maxout across groups of channels.
    If in_ch = out_ch * group, returns out_ch by max over group.
    """
    def __init__(self, group=4):
        super().__init__()
        self.group = group

    def forward(self, x):
        b, c, h, w = x.shape
        assert c % self.group == 0
        x = x.reshape(b, c // self.group, self.group, h, w)
        x = x.max(dim=2).values
        return x


class BReLU(nn.Module):
    """
    Bilateral ReLU: clamp to [0, 1].
    DehazeNet proposes BReLU to constrain transmission. [1](https://caibolun.github.io/DehazeNet/index.html)[4](https://caibolun.github.io/papers/DehazeNet.pdf)
    """
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)

class AffineHead(nn.Module):
    def __init__(self, in_ch=48, per_channel=True, beta_range=0.10):
        super().__init__()
        self.per_channel = per_channel
        self.beta_range  = beta_range
        out_ch = 3 if per_channel else 1

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1 * out_ch)
        )

    def forward(self, feat):
        raw_beta = self.net(feat)
        beta  = self.beta_range  * torch.tanh(raw_beta)
        return beta

class DehazeNet(nn.Module):
    """
    DehazeNet-style transmission estimator:
      - Feature extraction: conv + Maxout
      - Multi-scale mapping: parallel conv(3,5,7) then concat
      - Local extremum: max filter (implemented as maxpool stride=1)
      - Non-linear regression: conv -> BReLU => transmission map t(x)
    [1](https://caibolun.github.io/DehazeNet/index.html)[2](https://deepwiki.com/caibolun/DehazeNet/2.1-feature-extraction-pipeline)[3](https://deepwiki.com/caibolun/DehazeNet/4.1-transmission-map-estimation)
    """
    def __init__(self, in_ch=3):
        super().__init__()

        # F1: feature extraction (conv then Maxout)
        # conv1 produces 16 channels, maxout group=4 => 4 channels. [2](https://deepwiki.com/caibolun/DehazeNet/2.1-feature-extraction-pipeline)[3](https://deepwiki.com/caibolun/DehazeNet/4.1-transmission-map-estimation)
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=5, padding=2)
        self.maxout = Maxout(group=4)
        self.affine_head = AffineHead(in_ch=48, per_channel=False)

        # F2: multi-scale mapping (3x3, 5x5, 7x7) each to 16 channels. [2](https://deepwiki.com/caibolun/DehazeNet/2.1-feature-extraction-pipeline)[3](https://deepwiki.com/caibolun/DehazeNet/4.1-transmission-map-estimation)
        self.conv3 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(4, 16, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(4, 16, kernel_size=7, padding=3)

        # F3: local extremum (max filter 3x3) [2](https://deepwiki.com/caibolun/DehazeNet/2.1-feature-extraction-pipeline)[3](https://deepwiki.com/caibolun/DehazeNet/4.1-transmission-map-estimation)
        self.local_ext = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # F4: non-linear regression to 1-channel transmission map, followed by BReLU. [1](https://caibolun.github.io/DehazeNet/index.html)[3](https://deepwiki.com/caibolun/DehazeNet/4.1-transmission-map-estimation)
        self.conv_reg = nn.Conv2d(48, 1, kernel_size=5, padding=2)
        self.brelu = BReLU()

        # New
        self.use_brelu = True

        # init (safe defaults)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # DehazeNet reference implementation subtracts 0.5 before processing. [2](https://deepwiki.com/caibolun/DehazeNet/2.1-feature-extraction-pipeline)[3](https://deepwiki.com/caibolun/DehazeNet/4.1-transmission-map-estimation)
        x0 = x - 0.5

        f1 = self.conv1(x0)
        f1 = self.maxout(f1)  # -> [B,4,H,W]

        f2_3 = F.relu(self.conv3(f1), inplace=True)
        f2_5 = F.relu(self.conv5(f1), inplace=True)
        f2_7 = F.relu(self.conv7(f1), inplace=True)

        f2 = torch.cat([f2_3, f2_5, f2_7], dim=1)  # -> [B,48,H,W]
        f3 = self.local_ext(f2)  # local extremum

        t = self.conv_reg(f3)    # -> [B,1,H,W]

        if self.use_brelu: # New
            t = self.brelu(t)        # transmission in [0,1]

        beta = self.affine_head(f2)

        return t, beta

