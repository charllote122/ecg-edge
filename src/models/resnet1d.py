"""
1D ResNet-style CNN for multi-label ECG classification.

Architecture:
    Input: (B, 12, 5000)
    Initial conv: 12 -> 64
    4 residual blocks (64 -> 128 -> 256 -> 512)
    Global average pool -> (B, 512)
    Linear -> (B, 5) with sigmoid activation

Designed for edge deployment: <5M params, INT8-friendly.

Usage:
    from src.models.resnet1d import ResNet1D
    model = ResNet1D(in_channels=12, num_classes=5)
    logits = model(torch.randn(4, 12, 5000))
    probs = torch.sigmoid(logits)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """
    Two Conv1d layers with BatchNorm + ReLU, plus a skip connection.

    If in_channels != out_channels or stride != 1, the skip connection uses a
    1x1 Conv1d to match shapes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=7, stride=stride, padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=7, stride=1, padding=3, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection: identity if shapes match, else 1x1 conv
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    1D ResNet for multi-label time-series classification.

    Args:
        in_channels: number of input channels (12 for 12-lead ECG)
        num_classes: number of output classes (5 superclasses)
        dropout: dropout probability
        base_channels: channels in the first conv layer (default 64)

    Output:
        Logits of shape (B, num_classes). Apply sigmoid for probabilities.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        dropout: float = 0.3,
        base_channels: int = 64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels,
                      kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        # 4 residual blocks
        # Block 1: 64 -> 64   (no downsampling)
        # Block 2: 64 -> 128  (stride 2)
        # Block 3: 128 -> 256 (stride 2)
        # Block 4: 256 -> 512 (stride 2)
        self.layer1 = ResidualBlock1D(base_channels, base_channels,
                                      stride=1, dropout=dropout)
        self.layer2 = ResidualBlock1D(base_channels, base_channels * 2,
                                      stride=2, dropout=dropout)
        self.layer3 = ResidualBlock1D(base_channels * 2, base_channels * 4,
                                      stride=2, dropout=dropout)
        self.layer4 = ResidualBlock1D(base_channels * 4, base_channels * 8,
                                      stride=2, dropout=dropout)

        # Classifier head
        self.gap = nn.AdaptiveAvgPool1d(1)          # (B, 512, 1)
        self.flatten = nn.Flatten()                  # (B, 512)
        self.classifier = nn.Linear(base_channels * 8, num_classes)

        # Weight init (Kaiming for convs, zeros for final bias)
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for conv layers, zeros for final classifier bias."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init the final classifier bias for stable training start
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 12, 5000) float32
        Returns:
            logits: (B, 5) float32 — apply sigmoid for probabilities
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    print("Building ResNet1D...")
    model = ResNet1D(in_channels=12, num_classes=5, dropout=0.3)
    n_params = count_parameters(model)
    print(f"  parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  size (FP32): ~{n_params * 4 / 1e6:.1f} MB")
    print(f"  size (INT8): ~{n_params / 1e6:.1f} MB (after quantization)")
    print()

    print("Forward pass test...")
    x = torch.randn(4, 12, 5000)  # batch of 4 random signals
    logits = model(x)
    probs = torch.sigmoid(logits)
    print(f"  input shape:  {x.shape}")
    print(f"  output shape: {logits.shape}")
    print(f"  logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"  probs range:  [{probs.min():.3f}, {probs.max():.3f}]")
    print()

    # Verify probs are in [0, 1]
    assert probs.min() >= 0 and probs.max() <= 1, "sigmoid output out of range"
    print("All checks passed.")
