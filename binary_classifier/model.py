import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# SRM filter bank
def _srm_kernels() -> torch.Tensor:
    """
    Six SRM-style 3×3 high-pass kernels, shape (6, 1, 3, 3).

    Each kernel is normalized so that a 1-unit pixel perturbation (= 1 LSB
    in [0,255] space) produces a residual of magnitude ≈ 1.  Specifically,
    each kernel is divided by the sum of its positive coefficients, so the
    response at the perturbed pixel equals exactly 1 when all neighbors
    are unmodified.
    """
    raw = [
        # 8-neighbour Laplacian — captures all second-order directions
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        # Horizontal second derivative
        [[0, 0, 0], [-1, 2, -1], [0, 0, 0]],
        # Vertical second derivative
        [[0, -1, 0], [0, 2, 0], [0, -1, 0]],
        # Full 2-D second derivative (covers diagonal neighbors twice)
        [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
        # Diagonal second derivative
        [[-1, 0, 0], [0, 2, 0], [0, 0, -1]],
        # Anti-diagonal second derivative
        [[0, 0, -1], [0, 2, 0], [-1, 0, 0]],
    ]
    out = []
    for k in raw:
        k = np.array(k, dtype=np.float32)
        k = k / k[k > 0].sum()
        out.append(torch.tensor(k).view(1, 1, 3, 3))
    return torch.cat(out, dim=0)


class SRMPreprocessing(nn.Module):
    def __init__(self, T: float = 3.0):
        super().__init__()
        srm = _srm_kernels()
        K = srm.shape[0]
        # Repeat once per input channel
        self.register_buffer("weight", srm.repeat(3, 1, 1, 1))
        self.out_channels = 3 * K
        self.T = T
        self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast frozen buffer to match input dtype (required under torch AMP)
        w = self.weight.to(x.dtype)
        # Scale to [0, 255] integer-pixel space before applying kernels
        out = F.conv2d(x * 255.0, w, padding=1, groups=3)
        out = out.clamp(-self.T, self.T)
        return self.bn(out)


# Residual block
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.conv(x) + self.shortcut(x)))


# Full steganalysis network
class SteganalysisNet(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()

        self.srm = SRMPreprocessing(T=3.0)
        srm_ch = self.srm.out_channels

        self.stem = nn.Sequential(
            nn.Conv2d(srm_ch, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResBlock(32, 64, pool=True)
        self.layer2 = ResBlock(64, 128, pool=True)
        self.layer3 = ResBlock(128, 256, pool=True)
        self.layer4 = ResBlock(256, 512, pool=False)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal for conv/linear; standard init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # SRM buffer was set by _srm_kernels() — intentionally not overwritten here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.srm(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)
