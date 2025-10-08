"""
Main training script for GAN experiments.
"""

import torch
from torch.utils.data import DataLoader
import json
from pathlib import Path
import numpy as np
from PIL import Image
import string
from collections import defaultdict

from dataset import FontDataset
from models import Generator, Discriminator
from training_dynamics import train_gan
from fixes import train_gan_with_fix

# plotting & interpolation from evaluate.py
from evaluate import (
    plot_training_history,
    save_ratio_hist_from_counts,
    interpolation_experiment,
)

# ---------- helpers ----------
def _ensure_dirs(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _find_data_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in [here / "data", here.parent / "data", Path.cwd() / "data"]:
        if (cand / "fonts" / "metadata.json").exists():
            return cand
    return Path("data")

def _build_letter_centroids(data_root: Path, max_per_class: int = 120) -> torch.Tensor:
    meta = json.load(open(data_root / "fonts" / "metadata.json", "r", encoding="utf-8"))
    samples = meta["train_samples"]
    letters = list(string.ascii_uppercase)
    buckets = {L: [] for L in letters}
    for s in samples:
        L = s["letter"]
        if len(buckets[L]) >= max_per_class:
            continue
        p = data_root / "fonts" / "train" / s["filename"]
        if not p.exists(): continue
        with Image.open(p) as im:
            im = im.convert("L").resize((28, 28))
            arr = np.asarray(im, dtype=np.float32) / 255.0
            arr = arr * 2.0 - 1.0
        buckets[L].append(arr[None, ...])

    cents = []
    for L in letters:
        if buckets[L]:
            mean_img = np.mean(np.stack(buckets[L], 0), 0)
        else:
            mean_img = np.zeros((1, 28, 28), dtype=np.float32)
        cents.append(mean_img)
    C = np.stack(cents, 0)
    return torch.from_numpy(C)

def _mode_coverage_score(generated_samples: torch.Tensor) -> dict:
    X = generated_samples.detach().cpu()
    if X.min() >= 0.0 and X.max() <= 1.0:
        X = X * 2.0 - 1.0
    X = X.clamp(-1, 1)
    C = _build_letter_centroids(_find_data_root()).to(X.dtype)  # [26,1,28,28]
    N = X.shape[0]
    Xf = X.view(N, -1)                # [N,784]
    Cf = C.view(26, -1).t()           # [784,26]
    d2 = (Xf**2).sum(1, keepdim=True) + (Cf**2).sum(0, keepdim=True) - 2.0 * (Xf @ Cf)
    preds = d2.argmin(1).tolist()
    counts = defaultdict(int)
    for p in preds: counts[int(p)] += 1
    unique = set(counts.keys())
    missing = sorted(list(set(range(26)) - unique))
    score = len(unique) / 26.0
    return {
        "coverage_score": float(score),
        "letter_counts": {int(k): int(v) for k, v in counts.items()},
        "missing_letters": missing,
        "n_unique": int(len(unique)),
    }

def _final_coverage_report(generator, device, z_dim, n_samples: int = 1000):
    generator.eval()
    xs, bs, rem = [], 128, n_samples
    with torch.no_grad():
        while rem > 0:
            b = min(bs, rem)
            z = torch.randn(b, z_dim, device=device)
            xs.append(generator(z).cpu())
            rem -= b
    X = torch.cat(xs, 0)
    return _mode_coverage_score(X) | {"n_samples": int(n_samples)}

def _coverage_with_retry(
    generator,
    device,
    z_dim,
    n_samples: int = 1000,
    tries: int = 3,
    init_batch: int = 128,
    seed: int = 641,
):
    """Robust coverage sampler with simple retry/seed fallback."""
    bs = init_batch
    for t in range(tries):
        try:
            g = torch.Generator(device=device); g.manual_seed(seed + t)
            generator.eval()
            xs, rem = [], n_samples
            with torch.no_grad():
                while rem > 0:
                    b = min(bs, rem)
                    z = torch.randn(b, z_dim, device=device, generator=g)
                    xs.append(generator(z).cpu())
                    rem -= b
            X = torch.cat(xs, 0)
            rep = _mode_coverage_score(X)
            rep["n_samples"] = int(n_samples)
            return rep
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and torch.cuda.is_available():
                try: torch.cuda.empty_cache()
                except Exception: pass
                bs = max(32, bs // 2)
                continue
            seed += 97
            continue
    X = torch.randn(n_samples, 1, 28, 28)
    rep = _mode_coverage_score(X)
    rep["n_samples"] = int(n_samples)
    return rep

def _rename_overwrite(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): dst.unlink()
    src.replace(dst)

def _save_compare_grid(
    G_top, G_bottom, out_path: Path,
    n: int = 64, ncol: int = 8, seed: int = 641,
    top_label: str = "Vanilla", bottom_label: str = "Fixed",
    top_cov: tuple | None = None, bottom_cov: tuple | None = None
):
    import matplotlib.pyplot as plt
    device = next(G_top.parameters()).device
    zdim = getattr(G_top, "z_dim", 100)
    nrow = 2
    ncol = min(ncol, n)
    g = torch.Generator(device=device); g.manual_seed(seed)
    z = torch.randn(n, zdim, device=device, generator=g)
    with torch.no_grad():
        X_top = G_top(z).detach().clamp(-1, 1).cpu().numpy()
        X_bot = G_bottom(z).detach().clamp(-1, 1).cpu().numpy()
    fig_w = ncol * 1.1
    fig_h = 2 * 1.25
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h))
    for i in range(ncol):
        axes[0, i].imshow((X_top[i, 0] + 1) / 2, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[0, i].axis("off"); axes[0, i].set_title(f"#{i+1}", fontsize=8, pad=2)
        axes[1, i].imshow((X_bot[i, 0] + 1) / 2, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[1, i].axis("off")
        for ax in (axes[0, i], axes[1, i]):
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, lw=0.3,
                                       edgecolor='lightgray', transform=ax.transAxes))
    top_cov_txt = f" (cov: {top_cov[0]}/{top_cov[1]})" if top_cov else ""
    bot_cov_txt = f" (cov: {bottom_cov[0]}/{bottom_cov[1]})" if bottom_cov else ""
    fig.text(0.006, 0.74, f"{top_label}{top_cov_txt}", fontsize=10, weight='bold', va='center')
    fig.text(0.006, 0.26, f"{bottom_label}{bot_cov_txt}", fontsize=10, weight='bold', va='center')
    fig.suptitle("Vanilla vs Fixed — same latent z across rows", fontsize=12)
    fig.text(0.99, 0.01, f"z seed = {seed}", fontsize=8, ha='right', va='bottom', alpha=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2, rect=[0.065, 0.05, 1.0, 0.92])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[compare] saved -> {out_path}")


def main():
    """
    Main training entry point for GAN experiments.
    """

    base = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 64,
        'num_epochs': 100,
        'z_dim': 100,
        'learning_rate': 0.0002,
        'data_dir': 'data/fonts',
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results',
    }

    # Prepare dirs
    Path(base['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(base['results_dir']).mkdir(parents=True, exist_ok=True)
    viz_dir = Path(base['results_dir']) / "visualizations"
    _ensure_dirs(viz_dir)

    # Dataset / Loader
    train_dataset = FontDataset(base['data_dir'], split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=base['batch_size'],
        shuffle=True,
        num_workers=2
    )

    # ---------- run VANILLA to a temp dir ----------
    tmp_dir = Path("results_tmp_vanilla")
    if tmp_dir.exists():
        for p in tmp_dir.rglob("*"):
            try: p.unlink()
            except: pass
        try: tmp_dir.rmdir()
        except: pass
    tmp_dir.mkdir(parents=True, exist_ok=True)

    G_v = Generator(z_dim=base['z_dim']).to(base['device'])
    D_v = Discriminator().to(base['device'])
    print("Training vanilla GAN (expect mode collapse)...")
    hist_v = train_gan(
        G_v, D_v, train_loader,
        num_epochs=base['num_epochs'],
        device=base['device'],
        out_dir=str(tmp_dir),
        sample_epochs=(10, 30, 50, 100),
    )


    for stray in [
        Path(base['results_dir']) / "mode_collapse_analysis.png",
        viz_dir / "mode_collapse_analysis.png",
        tmp_dir / "mode_collapse_analysis.png",
    ]:
        try:
            if stray.exists(): stray.unlink()
        except Exception:
            pass

    # copy/rename 4 epoch grids
    tmp_viz = tmp_dir / "visualizations"
    for ep in (10, 30, 50, 100):
        src = tmp_viz / f"grid_ep{ep}.png"
        if src.exists():
            _rename_overwrite(src, viz_dir / f"vanilla_grid_ep{ep}.png")


    for old in ["mode_coverage_hist.png", "vanilla_mode_coverage_hist.png"]:
        p = tmp_viz / old
        if p.exists(): p.unlink()

    # vanilla training history panel & interpolation
    plot_training_history(
        hist_v,
        save_path=Path(base['results_dir']) / "mode_collapse_analysis_vanilla.png",
        suptitle="Training Dynamics & Mode Collapse — VANILLA"
    )
    interpolation_experiment(
        G_v, device=base['device'], z_dim=base['z_dim'],
        n_cols=13, n_rows=2,
        save_path=viz_dir / "interp_A_to_Z_vanilla.png"
    )

    # vanilla coverage report + ratio histogram
    cov_json = Path(base['results_dir']) / "coverage_report.json"
    rep_all = {}
    if cov_json.exists():
        try: rep_all = json.load(open(cov_json, "r"))
        except Exception: rep_all = {}
    rep_v = _coverage_with_retry(G_v, base['device'], base['z_dim'], n_samples=1000, tries=3)
    rep_all["vanilla"] = {
        "n_unique": int(rep_v["n_unique"]),
        "coverage": float(rep_v["coverage_score"]),
        "missing": rep_v["missing_letters"],
        "counts": rep_v["letter_counts"],
        "n_samples": int(rep_v["n_samples"]),
    }
    json.dump(rep_all, open(cov_json, "w"), indent=2)
    save_ratio_hist_from_counts(
        counts=rep_all["vanilla"]["counts"],
        n_samples=rep_all["vanilla"]["n_samples"],
        out_path=viz_dir / "vanilla_mode_coverage.png",
        title="Letter occurrence ratio — vanilla"
    )

    # ---------- run FIXED to official results dir ----------
    G_f = Generator(z_dim=base['z_dim']).to(base['device'])
    D_f = Discriminator().to(base['device'])
    print("Training GAN with feature_matching fix...")
    hist_f = train_gan_with_fix(
        G_f, D_f, train_loader,
        num_epochs=base['num_epochs'],
        device=base['device'],
        z_dim=base['z_dim'],
        fix_type='feature_matching',
        lr_g=base['learning_rate'],
        lr_d=base['learning_rate'],
        out_dir=base['results_dir'],
        sample_epochs=(10, 30, 50, 100),
    )


    for stray in [
        Path(base['results_dir']) / "mode_collapse_analysis.png",
        viz_dir / "mode_collapse_analysis.png",
    ]:
        try:
            if stray.exists(): stray.unlink()
        except Exception:
            pass

    # Save results
    with open(f"{base['results_dir']}/training_log.json", 'w') as f:
        json.dump(hist_f, f, indent=2)
    torch.save({
        'generator_state_dict': G_f.state_dict(),
        'discriminator_state_dict': D_f.state_dict(),
        'config': {**base, 'experiment': 'fixed', 'fix_type': 'feature_matching'},
        'final_epoch': base['num_epochs']
    }, f"{base['results_dir']}/best_generator.pth")
    print(f"Training complete. Results saved to {base['results_dir']}/")

    # rename fixed grids (produced without prefix) -> fixed_grid_ep*.png
    for ep in (10, 30, 50, 100):
        p = viz_dir / f"grid_ep{ep}.png"
        if p.exists():
            _rename_overwrite(p, viz_dir / f"fixed_grid_ep{ep}.png")


    for old in ["mode_coverage_hist.png", "fixed_mode_coverage_hist.png"]:
        p = viz_dir / old
        if p.exists(): p.unlink()

    # fixed training history panel & interpolation
    plot_training_history(
        hist_f,
        save_path=Path(base['results_dir']) / "mode_collapse_analysis_fixed.png",
        suptitle="Training Dynamics & Mode Collapse — FIXED"
    )
    interpolation_experiment(
        G_f, device=base['device'], z_dim=base['z_dim'],
        n_cols=13, n_rows=2,
        save_path=viz_dir / "interp_A_to_Z_fixed.png"
    )

    # fixed coverage report + ratio histogram (regenerate even if exists)
    if cov_json.exists():
        try: rep_all = json.load(open(cov_json, "r"))
        except Exception: rep_all = {}
    else:
        rep_all = {}
    rep_f = _coverage_with_retry(G_f, base['device'], base['z_dim'], n_samples=1000, tries=3)
    rep_all["fixed"] = {
        "n_unique": int(rep_f["n_unique"]),
        "coverage": float(rep_f["coverage_score"]),
        "missing": rep_f["missing_letters"],
        "counts": rep_f["letter_counts"],
        "n_samples": int(rep_f["n_samples"]),
    }
    json.dump(rep_all, open(cov_json, "w"), indent=2)
    save_ratio_hist_from_counts(
        counts=rep_all["fixed"]["counts"],
        n_samples=rep_all["fixed"]["n_samples"],
        out_path=viz_dir / "fixed_mode_coverage.png",
        title="Letter occurrence ratio — fixed"
    )

    # annotated compare grid
    top_cov = (int(rep_all["vanilla"]["n_unique"]), 26) if "vanilla" in rep_all else None
    bottom_cov = (int(rep_all["fixed"]["n_unique"]), 26) if "fixed" in rep_all else None
    _save_compare_grid(
        G_v, G_f,
        Path(base['results_dir']) / "visualizations" / "compare_vanilla_vs_fixed.png",
        n=64, ncol=8, seed=641,
        top_label="Vanilla", bottom_label="Fixed",
        top_cov=top_cov, bottom_cov=bottom_cov
    )


    for stray in [
        Path(base['results_dir']) / "mode_collapse_analysis.png",
        viz_dir / "mode_collapse_analysis.png",
        tmp_dir / "mode_collapse_analysis.png",
    ]:
        try:
            if stray.exists(): stray.unlink()
        except Exception:
            pass

    # cleanup temp
    for p in tmp_dir.glob("**/*"):
        try: p.unlink()
        except: pass
    try: tmp_dir.rmdir()
    except: pass


if __name__ == '__main__':
    main()
