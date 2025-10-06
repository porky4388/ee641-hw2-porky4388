# ==== evaluate.py : plotting + interpolation helpers ====
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import string

def plot_training_history(history: dict, save_path: str | Path, suptitle: str | None = None):
    """2x2 面板：moving-avg D/G loss、coverage、loss ratio、raw losses；可加抬頭。"""
    def _moving_avg(x, k=21):
        if len(x) == 0:
            return x
        k = max(1, min(k, len(x)//2*2+1))
        w = np.ones(k)/k
        return np.convolve(x, w, mode='same')

    epochs = np.array(history.get("epoch", []), dtype=float)
    d_loss = np.array(history.get("d_loss", []), dtype=float)
    g_loss = np.array(history.get("g_loss", []), dtype=float)
    m_epochs = np.array(history.get("mode_epoch", []), dtype=float)
    m_cov = np.array(history.get("mode_coverage", []), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ax = axes[0, 0]
    if len(epochs) and len(d_loss) and len(g_loss):
        ax.plot(epochs, _moving_avg(d_loss), label='D loss (MA)')
        ax.plot(epochs, _moving_avg(g_loss), label='G loss (MA)')
        ax.set_title("Loss (moving average)"); ax.set_xlabel("epoch")
        ax.legend(); ax.grid(alpha=0.3)
    else: ax.set_visible(False)

    ax = axes[0, 1]
    if len(m_epochs) and len(m_cov):
        ax.plot(m_epochs, m_cov, 'o-', alpha=0.9)
        ax.set_ylim(0, 1.05)
        ax.set_title("Mode coverage"); ax.set_xlabel("epoch"); ax.set_ylabel("coverage")
        ax.grid(alpha=0.3)
    else: ax.set_visible(False)

    ax = axes[1, 0]
    if len(epochs) and len(d_loss) and len(g_loss):
        ratio = np.divide(g_loss, d_loss + 1e-8)
        ax.plot(epochs, _moving_avg(ratio))
        ax.set_title("G/D loss ratio (MA)"); ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    else: ax.set_visible(False)

    ax = axes[1, 1]
    if len(epochs) and len(d_loss) and len(g_loss):
        ax.plot(epochs, d_loss, '.', alpha=0.4, ms=2, label='D')
        ax.plot(epochs, g_loss, '.', alpha=0.4, ms=2, label='G')
        ax.set_title("Raw losses"); ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=0.3)
    else: ax.set_visible(False)

    # 先排版一次
    fig.tight_layout()
    # 可選抬頭
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def save_ratio_hist_from_counts(
    counts: dict, n_samples: int, out_path: str | Path, title: str
):
    """
    Draw A–Z occurrence ratios (counts / N). Replaces legacy 0/1 coverage hist.
    counts keys may be int or str (0..25).
    """
    letters = list(string.ascii_uppercase)
    def _get(c, i): return int(c.get(i, c.get(str(i), 0)))
    ratios = [_get(counts, i) / max(1, n_samples) for i in range(26)]

    plt.figure(figsize=(12, 3))
    plt.bar(letters, ratios)
    ymax = max(ratios) if ratios else 0.0
    plt.ylim(0, max(0.05, ymax + 0.02))
    plt.ylabel("ratio")
    plt.title(f"{title} (N={n_samples})")
    for i, y in enumerate(ratios):
        if y > 0:
            plt.text(i, y, f"{100*y:.1f}%", ha="center", va="bottom", fontsize=7, rotation=90)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def interpolation_experiment(
    generator: torch.nn.Module,
    device: str | torch.device = "cuda",
    z_dim: int = 100,
    n_cols: int = 13,      # default 13 columns
    n_rows: int = 2,       # default 2 rows -> 26 images
    save_path: str | Path = "results/visualizations/interp_A_to_Z.png",
    seed: int = 641,
):
    """
    Interpolate between two random latent vectors and plot as a grid (default 2x13),
    so it won't be a single super-wide strip.
    """
    total = n_cols * n_rows
    g = torch.Generator(device=device); g.manual_seed(seed)
    zA = torch.randn(1, z_dim, device=device, generator=g)
    zZ = torch.randn(1, z_dim, device=device, generator=g)
    alphas = torch.linspace(0, 1, steps=total, device=device).view(-1, 1)
    Z = (1 - alphas) * zA + alphas * zZ   # [total, z_dim]

    generator.eval()
    with torch.no_grad():
        X = generator(Z).detach().clamp(-1, 1).cpu()  # [total,1,28,28]

    letters = list(string.ascii_uppercase)[:total]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.8, n_rows * 0.9))
    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c] if n_rows > 1 else axes[c]
            ax.axis("off")
            ax.imshow((X[idx, 0].numpy() + 1) / 2, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(letters[idx], fontsize=8, pad=1)
            idx += 1
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.1)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
