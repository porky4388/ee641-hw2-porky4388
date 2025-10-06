"""
GAN training implementation with mode collapse analysis + required visualizations.
- Saves sample grids at epochs 10/30/50/100 to results/visualizations/
- Records mode coverage during training (never NaN)
- Saves final mode coverage histogram to results/visualizations/mode_coverage_hist.png
- visualize_mode_collapse(...) remains as the entry used by train.py to save mode_collapse_analysis.png
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Iterable, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image


# ------------------------- path helpers ------------------------- #
def _results_dirs(out_dir: str | Path | None) -> tuple[Path, Path]:
    """
    Return (results_dir, viz_dir). If out_dir is None, default to problem1/results.
    """
    base = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "results"
    viz = base / "visualizations"
    viz.mkdir(parents=True, exist_ok=True)
    return base, viz


def _resolve_fonts_dir(data_root: Optional[str | Path]) -> Path:
    """
    Return a Path to the *fonts directory* containing:
        metadata.json
        train/*.png
        val/*.png
    Tries multiple typical locations so it works whether you run from repo root or problem1/.
    """
    here = Path(__file__).resolve().parent      # problem1/
    candidates: List[Path] = []

    if data_root is not None:
        dr = Path(data_root)
        candidates += [dr, dr / "fonts"]  # allow .../data or .../data/fonts

    candidates += [
        here / "data" / "fonts",            # problem1/data/fonts
        here.parent / "data" / "fonts",     # ../data/fonts
        Path.cwd() / "data" / "fonts",      # CWD/data/fonts
    ]

    for c in candidates:
        if (c / "metadata.json").exists():
            return c

    tried = [str(c) for c in candidates]
    raise FileNotFoundError("Cannot locate fonts dataset. Tried:\n  - " + "\n  - ".join(tried))


# ------------------------- image utilities ------------------------- #
@torch.no_grad()
def _save_image_grid_from_noise(
    generator,
    device: str,
    z_dim: int,
    n_samples: int,
    nrow: int,
    save_path: Path,
    rng_seed: int = 641,
):
    """
    Sample n_samples images with a fixed RNG seed and save as a grid.
    Assumes generator outputs in [-1,1] (tanh). Converts to [0,1] for saving.
    """
    g = torch.Generator(device=device)
    g.manual_seed(rng_seed)
    z = torch.randn(n_samples, z_dim, generator=g, device=device)
    x = generator(z).detach().clamp(-1, 1)  # (N,1,28,28)

    # Make a simple grid with matplotlib (avoid torchvision dependency)
    imgs = [xi.squeeze(0).cpu().numpy() for xi in x]  # [-1,1]
    rows = (len(imgs) + nrow - 1) // nrow
    fig, axes = plt.subplots(rows, nrow, figsize=(nrow * 1.1, rows * 1.1))
    axes = np.array(axes).reshape(rows, nrow)

    for i in range(rows * nrow):
        ax = axes[i // nrow, i % nrow]
        ax.axis("off")
        if i < len(imgs):
            ax.imshow((imgs[i] + 1) / 2.0, cmap="gray", vmin=0, vmax=1)

    fig.tight_layout(pad=0.1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _save_mode_histogram(present_mask: np.ndarray, save_path: Path):
    """
    Save a histogram (bar plot) of which letters survive (1) vs missing (0).
    present_mask: numpy array (26,) with 0/1.
    """
    letters = [chr(65 + i) for i in range(26)]
    xs = np.arange(26)
    plt.figure(figsize=(10, 3))
    plt.bar(xs, present_mask.astype(np.int32), width=0.8)
    plt.xticks(xs, letters)
    plt.yticks([0, 1])
    plt.ylim(0, 1.05)
    plt.title("Mode Coverage Histogram (1 = letter present)")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# ------------------------- Core Training ------------------------- #
def train_gan(
    generator,
    discriminator,
    data_loader,
    num_epochs: int = 100,
    device: str = "cuda",
    z_dim: int | None = None,
    analysis_every: int = 10,
    data_root: str | Path | None = None,   # points to .../data  or  .../data/fonts
    normalize_generated: str = "neg_one_one",  # generator uses tanh by default
    # NEW: outputs & visuals (no change needed in train.py)
    out_dir: str | Path | None = None,
    sample_epochs: Iterable[int] = (10, 30, 50, 100),
    grid_samples: int = 64,
    grid_nrow: int = 8,
):
    """
    Standard GAN training (vanilla GAN) that tends to exhibit mode collapse.

    Side effects:
      - Save sample grids at epochs in `sample_epochs`
      - Save final mode coverage histogram at the end

    Returns:
      history: dict(list) with keys: 'd_loss', 'g_loss', 'epoch', 'mode_coverage', 'mode_epoch'
    """
    results_dir, viz_dir = _results_dirs(out_dir)

    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()

    if z_dim is None:
        z_dim = getattr(generator, "z_dim", 100)

    # Optimizers & loss
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Training history
    history = defaultdict(list)

    # Prepare analyzer (auto-resolve data even if data_root is None)
    try:
        fonts_dir = _resolve_fonts_dir(data_root)
        mode_analyzer = _CentroidModeAnalyzer(fonts_dir=fonts_dir, device=device)
    except Exception as e:
        print(f"[warn] Mode analyzer disabled: {e}")
        mode_analyzer = None

    # ----- training loop ----- #
    last_present_mask = None
    for epoch in range(1, num_epochs + 1):
        for batch_idx, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device)  # (B,1,28,28) in [-1,1] or [0,1]
            bsz = real_images.size(0)

            # labels
            real_labels = torch.ones(bsz, 1, device=device)
            fake_labels = torch.zeros(bsz, 1, device=device)

            # ========== Train D ==========
            d_optimizer.zero_grad(set_to_none=True)
            d_real = discriminator(real_images)
            d_real_loss = criterion(d_real, real_labels)

            z = torch.randn(bsz, z_dim, device=device)
            with torch.no_grad():
                fake_images = generator(z)
            d_fake = discriminator(fake_images.detach())
            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # ========== Train G ==========
            g_optimizer.zero_grad(set_to_none=True)
            z = torch.randn(bsz, z_dim, device=device)
            gen_images = generator(z)
            d_gen = discriminator(gen_images)
            g_loss = criterion(d_gen, real_labels)
            g_loss.backward()
            g_optimizer.step()

            # log every ~10 batches
            if batch_idx % 10 == 0:
                history["d_loss"].append(float(d_loss.item()))
                history["g_loss"].append(float(g_loss.item()))
                frac_epoch = epoch - 1 + batch_idx / max(1, len(data_loader))
                history["epoch"].append(float(frac_epoch))

        # Save sample grids at specific milestones (no change in train.py needed)
        if epoch in set(sample_epochs):
            grid_path = viz_dir / f"grid_ep{epoch}.png"
            _save_image_grid_from_noise(
                generator, device, z_dim,
                n_samples=grid_samples, nrow=grid_nrow,
                save_path=grid_path, rng_seed=641
            )
            print(f"[Epoch {epoch:03d}] Saved sample grid -> {grid_path}")

        # periodic coverage analysis (robust; never NaN)
        if (analysis_every is not None) and (epoch % analysis_every == 0):
            if mode_analyzer is not None:
                cov, present = analyze_mode_coverage(
                    generator,
                    device=device,
                    n_samples=1000,
                    analyzer=mode_analyzer,
                    normalize_generated=normalize_generated,
                    return_presence=True,
                )
                last_present_mask = present  # keep last mask for histogram
                history["mode_coverage"].append(float(cov))
                history["mode_epoch"].append(float(epoch))
                print(f"[Epoch {epoch:03d}] Mode coverage = {cov:.3f}")
            else:
                history["mode_coverage"].append(0.0)
                history["mode_epoch"].append(float(epoch))

    # Save final mode coverage histogram (which letters survive)
    if last_present_mask is not None:
        _save_mode_histogram(last_present_mask, viz_dir / "mode_coverage_hist.png")

    # ------ auto-save mode collapse plot to results/ ------
    try:
        out_png = Path(__file__).resolve().parent / "results" / "mode_collapse_analysis.png"
        visualize_mode_collapse(history, out_png)
        print(f"[train] Saved mode collapse analysis -> {out_png}")
    except Exception as e:
        print(f"[warn] failed to save mode collapse analysis: {e}")

    return history


# ------------------------- Mode Coverage ------------------------- #
class _CentroidModeAnalyzer:
    """
    Build class centroids (A..Z) from training images and use 1-NN to label
    generated samples. Does not require any external classifier.
    """
    def __init__(
        self,
        fonts_dir: str | Path,               # <- directly points to .../fonts
        device: str = "cpu",
        max_per_class: int = 50
    ):
        self.fonts_dir = Path(fonts_dir)
        self.device = device
        self.max_per_class = max_per_class
        self.letters = [chr(65 + i) for i in range(26)]
        self.centroids = self._build_centroids()    # [26,1,28,28] in [-1,1]

    def _build_centroids(self) -> torch.Tensor:
        meta_path = self.fonts_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        samples = meta["train_samples"]

        per_class_imgs = {L: [] for L in self.letters}
        for s in samples:
            L = s["letter"]
            if len(per_class_imgs[L]) >= self.max_per_class:
                continue
            img_path = self.fonts_dir / "train" / s["filename"]
            if not img_path.exists():
                continue
            with Image.open(img_path) as im:
                im = im.convert("L")
                if im.size != (28, 28):
                    im = im.resize((28, 28))
                arr = np.asarray(im, dtype=np.float32) / 255.0
                arr = arr * 2.0 - 1.0  # -> [-1,1]
            per_class_imgs[L].append(arr[None, ...])  # (1,28,28)

        centroids = []
        for L in self.letters:
            imgs = per_class_imgs[L]
            if len(imgs) == 0:
                centroids.append(np.zeros((1, 28, 28), dtype=np.float32))
            else:
                arr = np.stack(imgs, axis=0)  # (K,1,28,28)
                mean_img = arr.mean(axis=0)   # (1,28,28)
                centroids.append(mean_img)

        C = np.stack(centroids, axis=0)  # (26,1,28,28)
        return torch.from_numpy(C).to(self.device)

    @torch.no_grad()
    def predict_letters(self, imgs: torch.Tensor) -> np.ndarray:
        """
        imgs: [N,1,28,28] in [-1,1]
        returns: numpy [N,] with values 0..25
        """
        C = self.centroids.view(26, -1)         # (26,D)
        X = imgs.view(imgs.size(0), -1)         # (N,D)
        x2 = (X ** 2).sum(dim=1, keepdim=True)  # (N,1)
        c2 = (C ** 2).sum(dim=1)[None, :]       # (1,26)
        dist2 = x2 + c2 - 2.0 * (X @ C.t())     # (N,26)
        pred = torch.argmin(dist2, dim=1)       # (N,)
        return pred.cpu().numpy()


@torch.no_grad()
def analyze_mode_coverage(
    generator,
    device: str,
    n_samples: int = 1000,
    analyzer: _CentroidModeAnalyzer | None = None,
    normalize_generated: str = "neg_one_one",
    return_presence: bool = False,
):
    """
    Measure mode coverage by counting unique letters in generated samples.
    Returns a float in [0,1]; never NaN.
    If return_presence=True, also returns a (26,) 0/1 numpy mask of present letters.
    """
    try:
        assert analyzer is not None, "analyzer is required for mode coverage"
        generator.eval()

        z_dim = getattr(generator, "z_dim", 100)

        # Generate
        bs = 256
        imgs = []
        num = 0
        while num < n_samples:
            cur = min(bs, n_samples - num)
            z = torch.randn(cur, z_dim, device=device)
            x = generator(z)  # (cur,1,28,28) usually in [-1,1]
            if normalize_generated == "zero_one":
                # convert to [-1,1] to match centroids
                x = (x.clamp(0, 1) * 2.0) - 1.0
            imgs.append(x.detach())
            num += cur
        imgs = torch.cat(imgs, dim=0)  # (N,1,28,28)

        # Classify
        pred = analyzer.predict_letters(imgs)  # numpy (N,)
        uniq = np.unique(pred)
        coverage = float(len(uniq) / 26.0)
        if return_presence:
            present = np.zeros(26, dtype=np.uint8)
            present[uniq] = 1
            generator.train()
            return coverage, present
        generator.train()
        return coverage
    except Exception as e:
        print(f"[warn] analyze_mode_coverage failed: {e}")
        if return_presence:
            return 0.0, np.zeros(26, dtype=np.uint8)
        return 0.0


# ------------------------- Visualization for train.py ------------------------- #
def visualize_mode_collapse(history: dict, save_path: str | Path):
    """
    Visualize training curves & (sampled) mode coverage points.
    This function is invoked by train.py to create results/mode_collapse_analysis.png
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs_f = history.get("epoch", [])
    d_loss = history.get("d_loss", [])
    g_loss = history.get("g_loss", [])
    mode_cov = history.get("mode_coverage", [])
    mode_ep = history.get("mode_epoch", [])

    plt.figure(figsize=(10, 6))

    # Loss curves
    if epochs_f and d_loss and g_loss:
        plt.plot(epochs_f, d_loss, label="D loss")
        plt.plot(epochs_f, g_loss, label="G loss")

    # Mode coverage points
    if mode_cov and mode_ep:
        plt.scatter(mode_ep, mode_cov, marker="o", s=40, label="Mode coverage (unique/26)")

    plt.xlabel("Epoch")
    plt.ylabel("Loss / Coverage")
    plt.title("GAN Training Dynamics & Mode Collapse")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
