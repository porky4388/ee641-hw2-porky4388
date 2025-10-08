"""
GAN stabilization fixes.
This file implements Feature Matching (FM) with a mixed generator loss:
    g_loss = g_adv + fm_weight * fm_loss
and produces all required artifacts for the homework.

Side effects per run (under out_dir / results):
  - visualizations/grid_ep{10,30,50,100}.png
  - visualizations/mode_coverage_hist.png
  - mode_collapse_analysis.png
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Iterable
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# ---- shared helpers from training_dynamics ----
from training_dynamics import (
    _results_dirs,
    _resolve_fonts_dir,
    _CentroidModeAnalyzer,
    analyze_mode_coverage,
    _save_image_grid_from_noise,
    _save_mode_histogram,
    visualize_mode_collapse,
)


# ------------------------- feature helpers ------------------------- #
def _extract_features(discriminator: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Prefer discriminator.features(x) if available; otherwise fall back to forward()
    as a (weak) feature. Returns flattened features [B, F].
    """
    if hasattr(discriminator, "features"):
        h = discriminator.features(x)
    else:
        # fallback to logits/probabilities as features
        h = discriminator(x)
    if h.dim() > 2:
        h = h.view(h.size(0), -1)
    return h


def _feature_matching_loss(
    discriminator: nn.Module,
    real_imgs: torch.Tensor,
    fake_imgs: torch.Tensor,
) -> torch.Tensor:
    """
    L_FM = || E[f(x_real)] - E[f(x_fake)] ||_2^2
    Only fake path keeps grad so gradients flow into G.
    """
    with torch.no_grad():
        feat_real = _extract_features(discriminator, real_imgs).detach()  # [B,F]
        mu_real = feat_real.mean(dim=0, keepdim=True)                     # [1,F]
    feat_fake = _extract_features(discriminator, fake_imgs)               # [B,F]
    mu_fake = feat_fake.mean(dim=0, keepdim=True)                         # [1,F]
    return torch.mean((mu_fake - mu_real) ** 2)


# ------------------------- training with FM ------------------------- #
def train_gan_with_fix(
    generator: nn.Module,
    discriminator: nn.Module,
    data_loader,
    num_epochs: int = 100,
    device: str = "cuda",
    z_dim: int | None = None,
    # --- optimization ---
    lr_g: float = 2e-4,
    lr_d: float = 1e-4,
    betas: Tuple[float, float] = (0.5, 0.999),
    # --- FM control ---
    fix_type: str = "feature_matching",
    fm_weight: float = 15.0,
    # --- analysis / outputs ---
    analysis_every: int = 10,
    data_root: str | Path | None = None,
    normalize_generated: str = "neg_one_one",
    out_dir: str | Path | None = None,
    sample_epochs: Iterable[int] = (10, 30, 50, 100),
    # --- regularization ---
    real_label_smooth: float = 0.9,        # one-sided label smoothing for real
) -> Dict[str, List[float]]:
    """
    Train a GAN where the generator uses Feature Matching (with mixed loss).
    Returns:
        history dict with keys: 'd_loss', 'g_loss', 'epoch', 'mode_coverage', 'mode_epoch'
    """
    assert fix_type == "feature_matching", "Only 'feature_matching' is implemented."

    generator = generator.to(device).train()
    discriminator = discriminator.to(device).train()
    if z_dim is None:
        z_dim = getattr(generator, "z_dim", 100)

    # directories & analyzer
    results_dir, viz_dir = _results_dirs(out_dir)
    try:
        fonts_dir = _resolve_fonts_dir(data_root)
        analyzer = _CentroidModeAnalyzer(fonts_dir=fonts_dir, device=device)
    except Exception as e:
        print(f"[fix] Mode analyzer disabled: {e}")
        analyzer = None

    # losses & optimizers
    bce = nn.BCELoss()
    g_opt = optim.Adam(generator.parameters(), lr=lr_g, betas=betas)
    d_opt = optim.Adam(discriminator.parameters(), lr=lr_d, betas=betas)

    history = defaultdict(list)
    last_present_mask = None
    sample_epochs = set(sample_epochs)

    for epoch in range(1, num_epochs + 1):
        for batch_idx, (real, _) in enumerate(data_loader):
            real = real.to(device)                       # [B,1,28,28]
            bsz = real.size(0)

            # ------------------ Train D ------------------ #
            d_opt.zero_grad(set_to_none=True)
            y_real = torch.full((bsz, 1), real_label_smooth, device=device)  # 0.9
            y_fake = torch.zeros(bsz, 1, device=device)

            out_real = discriminator(real)               # expects Sigmoid() inside D
            d_real_loss = bce(out_real, y_real)

            with torch.no_grad():
                z = torch.randn(bsz, z_dim, device=device)
                fake = generator(z)
            out_fake = discriminator(fake.detach())
            d_fake_loss = bce(out_fake, y_fake)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_opt.step()

            # ------------------ Train G (FM + adversarial) ------------------ #
            g_opt.zero_grad(set_to_none=True)
            z = torch.randn(bsz, z_dim, device=device)
            fake = generator(z)

            # Feature Matching term
            fm_loss = _feature_matching_loss(discriminator, real, fake)

            # adversarial term (encourage D(fake) -> 1)
            out_fake_for_g = discriminator(fake)
            g_adv = bce(out_fake_for_g, torch.ones_like(out_fake_for_g, device=device))

            g_loss = g_adv + fm_weight * fm_loss
            g_loss.backward()
            g_opt.step()

            # ------------------ logging (fractional epoch) ------------------ #
            if batch_idx % 10 == 0:
                history["d_loss"].append(float(d_loss.item()))
                history["g_loss"].append(float(g_loss.item()))
                frac_epoch = epoch - 1 + batch_idx / max(1, len(data_loader))
                history["epoch"].append(float(frac_epoch))

        # ---- save milestone sample grids ----
        if epoch in sample_epochs:
            grid_path = viz_dir / f"grid_ep{epoch}.png"
            _save_image_grid_from_noise(
                generator, device, z_dim, n_samples=64, nrow=8,
                save_path=grid_path, rng_seed=641
            )
            print(f"[fix][ep{epoch:03d}] Saved sample grid -> {grid_path}")

        # ---- periodic mode coverage (robust; never NaN) ----
        if (analysis_every is not None) and (epoch % analysis_every == 0):
            if analyzer is not None:
                cov, present = analyze_mode_coverage(
                    generator,
                    device=device,
                    n_samples=1000,
                    analyzer=analyzer,
                    normalize_generated=normalize_generated,
                    return_presence=True,
                )
                last_present_mask = present
                history["mode_coverage"].append(float(cov))
                history["mode_epoch"].append(float(epoch))
                print(f"[fix][ep{epoch:03d}] Mode coverage = {cov:.3f}")
            else:
                history["mode_coverage"].append(0.0)
                history["mode_epoch"].append(float(epoch))

    # ---- end-of-training artifacts ----
    if last_present_mask is not None:
        _save_mode_histogram(last_present_mask, viz_dir / "mode_coverage_hist.png")
    visualize_mode_collapse(history, Path(results_dir) / "mode_collapse_analysis.png")

    return history
