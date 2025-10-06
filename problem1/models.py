"""
GAN models for font generation. (EE641 HW2 - Problem 1)
This file implements the exact architectures described in the starter:
- Generator: z -> 7x7x128 -> 14x14x64 -> 28x28x1 (Tanh)
- Discriminator: 28x28x1 -> 14x14x64 -> 7x7x128 -> 3x3x256 -> sigmoid
Both networks optionally support class-conditional inputs (one-hot 26-dim).
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim: int = 100, conditional: bool = False, num_classes: int = 26):
        """
        Generator network that produces 28×28 letter images.

        Args:
            z_dim: Dimension of latent vector z
            conditional: If True, concatenate one-hot class label to z
            num_classes: Number of classes (26 letters)
        """
        super().__init__()
        self.z_dim = z_dim
        self.conditional = conditional
        self.num_classes = num_classes

        input_dim = z_dim + (num_classes if conditional else 0)

        # Project and reshape: (B, input_dim) -> (B, 128*7*7) -> (B, 128, 7, 7)
        self.project = nn.Sequential(
            nn.Linear(input_dim, 128 * 7 * 7, bias=True),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(inplace=True),
        )

        # Upsample to 28x28:
        # (B,128,7,7) -> (B,64,14,14) -> (B,1,28,28)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 7 -> 14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=True),    # 14 -> 28
            nn.Tanh(),  # outputs in [-1, 1]
        )

    def forward(self, z: torch.Tensor, class_label: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z: [B, z_dim]
            class_label: [B, num_classes] one-hot (required if conditional=True)

        Returns:
            [B, 1, 28, 28] in [-1, 1]
        """
        if self.conditional:
            assert class_label is not None, "Generator requires one-hot class_label when conditional=True"
            x = torch.cat([z, class_label], dim=1)
        else:
            x = z
        x = self.project(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, conditional: bool = False, num_classes: int = 26):
        """
        Discriminator network that classifies 28×28 images as real/fake.
        If conditional=True, one-hot class vector is concatenated to features
        before the final linear classifier.
        """
        super().__init__()
        self.conditional = conditional
        self.num_classes = num_classes

        # Feature extractor:
        # 28x28x1 -> 14x14x64 -> 7x7x128 -> 3x3x256
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),   # 28 -> 14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), # 14 -> 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # kernel=4, stride=2, padding=1 : 7 -> 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        feature_dim = 256 * 3 * 3  # after the conv stack above

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + (num_classes if conditional else 0), 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, img: torch.Tensor, class_label: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            img: [B, 1, 28, 28]
            class_label: [B, num_classes] one-hot (required if conditional=True)

        Returns:
            prob_real: [B, 1] in [0,1] (sigmoid)
        """
        h = self.features(img)                  # [B, 256, 3, 3]
        h = h.view(h.size(0), -1)               # [B, 256*3*3]
        if self.conditional:
            assert class_label is not None, "Discriminator requires one-hot class_label when conditional=True"
            h = torch.cat([h, class_label], dim=1)
        out = self.classifier(h)                # [B,1]
        return out
