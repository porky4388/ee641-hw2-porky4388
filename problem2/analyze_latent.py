"""
EE641 HW2 Problem 2 — Latent analysis (t-SNE only)

Outputs (saved under results/):
- tsne_high.png  : t-SNE of z_high (style)
- tsne_low.png   : t-SNE of z_low  (variation)
- disentangle_metrics.json (uses silhouette from sklearn)
- interpolate_*  .npy/.png and controllable generation helpers (unchanged)

"""

from __future__ import annotations
import os
import json
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- strict t-SNE / silhouette (no fallback) ---
try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
except Exception as e:
    raise ImportError("scikit-learn is required for t-SNE. Install with `conda install scikit-learn` "
                      "or `pip install scikit-learn`.") from e


# ----------------------------- Utils -----------------------------

def _results_dir() -> str:
    here = os.path.dirname(__file__)
    out = os.path.join(here, "results")
    os.makedirs(out, exist_ok=True)
    return out

def _as_xy(batch):
    """Support dataloaders that return x or (x, y, ...)."""
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 2 and torch.is_tensor(batch[1]):
            return batch[0], batch[1]
        return batch[0], None
    return batch, None

def _to_input(x: torch.Tensor) -> torch.Tensor:
    """Ensure float input in [B,1,16,9]."""
    if x.dim() == 3:  # [B,16,9]
        x = x.unsqueeze(1)  # [B,1,16,9]
    if x.dim() != 4 or x.size(1) != 1 or x.size(2) != 16 or x.size(3) != 9:
        raise ValueError(f"Expected x as [B,1,16,9] or [B,16,9], got {tuple(x.size())}")
    return x.float()

def _sigmoid_logits_to_binary(logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return (probs > thr).float()

def _save_quick_grid(pats: np.ndarray, path: str, title: str = "", max_cols: int = 8):
    N = pats.shape[0]
    cols = min(max_cols, max(1, N))
    rows = int(np.ceil(N / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i < N:
            ax.imshow(pats[i], interpolation="nearest", aspect="auto", cmap="gray_r")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

@torch.no_grad()
def _collect_latents(model, data_loader, device="cuda", max_batches: int | None = None):
    """
    Encode the loader and return arrays:
      z_low, z_high, y (or None), mu/logvar (both levels).
    """
    model.eval().to(device)
    zs_low, zs_high = [], []
    ys = []
    mus_l, logs_l = [], []
    mus_h, logs_h = [], []

    for b_idx, batch in enumerate(data_loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        x, y = _as_xy(batch)
        x = _to_input(x).to(device)
        z_low, mu_low, logvar_low, z_high, mu_high, logvar_high = model.encode_hierarchy(x)

        zs_low.append(z_low.detach().cpu().numpy())
        zs_high.append(z_high.detach().cpu().numpy())
        mus_l.append(mu_low.detach().cpu().numpy())
        logs_l.append(logvar_low.detach().cpu().numpy())
        mus_h.append(mu_high.detach().cpu().numpy())
        logs_h.append(logvar_high.detach().cpu().numpy())
        if y is not None:
            ys.append(y.detach().cpu().numpy())

    ZL = np.concatenate(zs_low, axis=0) if zs_low else np.zeros((0, model.z_low_dim))
    ZH = np.concatenate(zs_high, axis=0) if zs_high else np.zeros((0, model.z_high_dim))
    MU_L = np.concatenate(mus_l, axis=0) if mus_l else np.zeros_like(ZL)
    LV_L = np.concatenate(logs_l, axis=0) if logs_l else np.zeros_like(ZL)
    MU_H = np.concatenate(mus_h, axis=0) if mus_h else np.zeros_like(ZH)
    LV_H = np.concatenate(logs_h, axis=0) if logs_h else np.zeros_like(ZH)
    Y = np.concatenate(ys, axis=0) if ys else None

    return {
        "z_low": ZL, "z_high": ZH, "y": Y,
        "mu_low": MU_L, "logvar_low": LV_L,
        "mu_high": MU_H, "logvar_high": LV_H,
    }

def _plot_embed(emb2d: np.ndarray, labels: np.ndarray | None, fname: str, title: str):
    outdir = _results_dir()
    fig, ax = plt.subplots(figsize=(6, 5))
    if labels is None:
        ax.scatter(emb2d[:, 0], emb2d[:, 1], s=8)
    else:
        labs = np.asarray(labels).astype(int)
        for lab in np.unique(labs):
            m = labs == lab
            ax.scatter(emb2d[m, 0], emb2d[m, 1], s=10, label=f"class {int(lab)}")
        ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[saved] {path}")


# ----------------------------- Analyses -----------------------------

@torch.no_grad()
def visualize_latent_hierarchy(model, data_loader, device: str = 'cuda',
                               perplexity: int = 30, random_state: int = 41,
                               max_points: int = 4000):
    """
    Visualize z_high and z_low with t-SNE only.
    Saves: tsne_high.png, tsne_low.png
    """
    pack = _collect_latents(model, data_loader, device=device)
    ZH, ZL, Y = pack["z_high"], pack["z_low"], pack["y"]

    # Subsample for speed
    idx = np.arange(ZH.shape[0])
    if ZH.shape[0] > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(ZH.shape[0], size=max_points, replace=False)
    ZH_, ZL_ = ZH[idx], ZL[idx]
    Y_ = None if Y is None else Y[idx]

    # t-SNE (sklearn)
    def tsne2d(X):
        if X.shape[0] < 2:
            return np.zeros((X.shape[0], 2), dtype=np.float32)
        perpl = min(perplexity, max(5, X.shape[0] // 10))
        tsne = TSNE(n_components=2, perplexity=perpl, random_state=random_state,
                    init="random", learning_rate="auto")
        return tsne.fit_transform(X)

    embH = tsne2d(ZH_)
    embL = tsne2d(ZL_)

    _plot_embed(embH, Y_, "tsne_high.png", "t-SNE of z_high (style)")
    _plot_embed(embL, Y_, "tsne_low.png",  "t-SNE of z_low (variation)")

    return {"count": int(ZH.shape[0]), "saved": ["tsne_high.png", "tsne_low.png"]}


@torch.no_grad()
def interpolate_styles(model, pattern1: torch.Tensor, pattern2: torch.Tensor,
                       n_steps: int = 10, device: str = 'cuda',
                       save_prefix: str = "interpolate"):
    """
    Interpolate between two patterns at both levels:
      - Path A (style): interpolate z_high, keep z_low from p(z_low | z_high_t)
      - Path B (variation): fix z_high of pat1, interpolate z_low between pat1/pat2
    """
    outdir = _results_dir()
    model.eval().to(device)

    p1 = _to_input(pattern1).to(device)
    p2 = _to_input(pattern2).to(device)

    z1_l, mu1_l, _, z1_h, mu1_h, _ = model.encode_hierarchy(p1)
    z2_l, mu2_l, _, z2_h, mu2_h, _ = model.encode_hierarchy(p2)

    def lin(a, b, t): return a * (1.0 - t) + b * t

    seq_high, seq_low = [], []
    for i in range(n_steps):
        t = i / max(1, n_steps - 1)
        # style path
        z_h_t = lin(mu1_h, mu2_h, t)
        logits = model.decode_hierarchy(z_h_t, z_low=None)
        patt = _sigmoid_logits_to_binary(logits).squeeze(0).cpu().numpy()
        seq_high.append(patt)
        # variation path
        z_l_t = lin(mu1_l, mu2_l, t)
        logits = model.decode_hierarchy(mu1_h, z_low=z_l_t)
        patt = _sigmoid_logits_to_binary(logits).squeeze(0).cpu().numpy()
        seq_low.append(patt)

    seq_high = np.stack(seq_high, axis=0)
    seq_low  = np.stack(seq_low, axis=0)
    np.save(os.path.join(outdir, f"{save_prefix}_high.npy"), seq_high)
    np.save(os.path.join(outdir, f"{save_prefix}_low.npy"),  seq_low)
    _save_quick_grid(seq_high, os.path.join(outdir, f"{save_prefix}_high.png"), title="Interpolate z_high")
    _save_quick_grid(seq_low,  os.path.join(outdir, f"{save_prefix}_low.png"),  title="Interpolate z_low (fixed style)")

    print(f"[saved] {save_prefix}_high/low .npy/.png in {outdir}")
    return {"paths": [f"{save_prefix}_high", f"{save_prefix}_low"], "steps": n_steps}


@torch.no_grad()
def measure_disentanglement(model, data_loader, device: str = 'cuda',
                            max_batches: int | None = None, save_name: str = "disentangle_metrics.json"):
    """
    Quantify disentanglement:
      - For z_high: between-class vs within-class scatter (trace ratios)
      - For z_low : within-class variance
      - Silhouette score on z_high (sklearn)
      - Also report per-dimension mean KL to diagnose collapse
    """
    outdir = _results_dir()
    pack = _collect_latents(model, data_loader, device=device, max_batches=max_batches)
    ZH, ZL, Y = pack["z_high"], pack["z_low"], pack["y"]

    metrics: Dict[str, Any] = {}
    if Y is None:
        metrics["note"] = "No labels provided in data_loader; metrics limited."
        metrics["z_high_var_global"] = float(np.trace(np.cov(ZH.T))) if ZH.size else 0.0
        metrics["z_low_var_global"]  = float(np.trace(np.cov(ZL.T))) if ZL.size else 0.0
    else:
        labels = np.asarray(Y).astype(int)
        classes = np.unique(labels)

        Sw_high = 0.0
        Sw_low  = 0.0
        means_high: Dict[int, np.ndarray] = {}
        for c in classes:
            Xc_h = ZH[labels == c]
            Xc_l = ZL[labels == c]
            if Xc_h.shape[0] >= 2:
                Sw_high += np.trace(np.cov(Xc_h.T))
            if Xc_l.shape[0] >= 2:
                Sw_low  += np.trace(np.cov(Xc_l.T))
            means_high[c] = Xc_h.mean(axis=0, keepdims=True)
        Sw_high /= max(1, len(classes))
        Sw_low  /= max(1, len(classes))

        M = np.stack([means_high[c].squeeze(0) for c in classes], axis=0)
        Sb_high = np.trace(np.cov(M.T)) if M.shape[0] >= 2 else 0.0

        metrics.update({
            "classes": classes.tolist(),
            "Sw_high_trace": float(Sw_high),
            "Sw_low_trace":  float(Sw_low),
            "Sb_high_trace": float(Sb_high),
            "fisher_ratio_high": float(Sb_high / max(1e-8, Sw_high)),
            "fisher_like_low":  float((np.trace(np.cov(ZL.T)) - Sw_low) / max(1e-8, Sw_low)) if ZL.shape[0] >= 2 else 0.0,
        })

        if ZH.shape[0] >= 10 and len(classes) >= 2:
            metrics["silhouette_high"] = float(silhouette_score(ZH, labels, metric="euclidean"))

    mu_h, lv_h = pack["mu_high"], pack["logvar_high"]
    mu_l, lv_l = pack["mu_low"],  pack["logvar_low"]
    if mu_h.size and lv_h.size:
        kl_high_dim = 0.5 * (-lv_h + np.exp(lv_h) + mu_h**2 - 1.0)  # [N,D]
        metrics["kl_high_dim_mean"] = np.mean(kl_high_dim, axis=0).tolist()
    if mu_l.size and lv_l.size:
        kl_low_dim = 0.5 * (-lv_l + np.exp(lv_l) + mu_l**2 - 1.0)
        metrics["kl_low_dim_mean"] = np.mean(kl_low_dim, axis=0).tolist()

    save_path = os.path.join(outdir, save_name)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {save_path}")
    return metrics


@torch.no_grad()
def controllable_generation(model,
                            data_loader,
                            device: str = 'cuda',
                            n_per_genre: int = 16,
                            temperatures: List[float] = (1.0, 1.2, 1.5),
                            save_prefix: str = "controllable"):
    outdir = _results_dir()
    model.eval().to(device)

    # prototypes in z_high
    pack = _collect_latents(model, data_loader, device=device)
    ZH, Y = pack["z_high"], pack["y"]
    if Y is None:
        raise ValueError("controllable_generation requires labels (return_label=True).")

    labels = np.asarray(Y).astype(int)
    classes = np.unique(labels)
    protos = np.stack([ZH[labels == c].mean(axis=0) for c in classes], axis=0)

    all_reports = {}
    for T in temperatures:
        grids = []
        for i, c in enumerate(classes):
            z_high = torch.tensor(protos[i:i+1], dtype=torch.float32, device=device)
            row = []
            for _ in range(n_per_genre):
                logits = model.decode_hierarchy(z_high, z_low=None, temperature=T)
                patt = _sigmoid_logits_to_binary(logits).squeeze(0).cpu().numpy()
                row.append(patt)
            grids.append(np.stack(row, axis=0))
        grids = np.stack(grids, axis=0)
        np.save(os.path.join(outdir, f"{save_prefix}_samples_T{T:.2f}.npy"), grids)
        _save_quick_grid(grids.reshape(-1, 16, 9),
                         os.path.join(outdir, f"{save_prefix}_samples_T{T:.2f}.png"),
                         title=f"Controllable gen (T={T:.2f})")

        preds, gts = [], []
        for i, c in enumerate(classes):
            batch = torch.tensor(grids[i], dtype=torch.float32).unsqueeze(1).to(device)
            _, _, _, z_high_gen, _, _ = model.encode_hierarchy(batch)
            Z = z_high_gen.detach().cpu().numpy()
            d2 = ((Z[:, None, :] - protos[None, :, :]) ** 2).sum(axis=2)
            pred = d2.argmin(axis=1)
            preds.append(pred); gts.append(np.full((Z.shape[0],), i))
        preds = np.concatenate(preds, 0); gts = np.concatenate(gts, 0)
        acc = float((preds == gts).mean())
        all_reports[f"T={T:.2f}"] = {"accuracy_nearest_proto": acc, "num_generated": int(gts.size)}

    report = {"temperatures": list(map(float, temperatures)), "metrics": all_reports}
    save_path = os.path.join(outdir, f"{save_prefix}_report.json")
    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[saved] {save_path}")
    return report
