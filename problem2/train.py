# train.py  — EE641 HW2 Problem 2 (run-tag aware outputs)
from __future__ import annotations
import os, json, shutil, argparse, wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from glob import glob
import torch.nn.functional as F


from dataset import get_drum_dataloaders, DrumPatternDataset
from hierarchical_vae import HierarchicalDrumVAE
from training_utils import (
    train_hierarchical_vae,
    sample_diverse_patterns,
)
# analyze_latent.py is t-SNE only (as requested)
from analyze_latent import (
    visualize_latent_hierarchy,
    measure_disentanglement,
    interpolate_styles,
)

# ========================== Config ==========================

def default_config():
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 64,
        "epochs": 100,
        "lr": 2e-3,
        "z_high_dim": 4,
        "z_low_dim": 12,
        "kl_mode": "cyclical",
        "kl_cycles": 4,
        "free_bits": 0.5,
        "use_temperature_anneal": True,
        "samples_per_style": 10,
        "interpolate_steps": 12,
        "audio_per_style": 3,
        "bpm": 120,
    }

# ---------- Run-aware path mapping ----------

def apply_run_paths(cfg, tag: str):
    ROOT = Path(__file__).parent / "results"
    cfg["results_root"] = str(ROOT)              # fixed: problem2/results
    tag = tag or "default"
    cfg["run_tag"] = tag

    # Visual/sample categories
    cfg["gen_dir"]    = str(ROOT / "generated_patterns" / tag)
    cfg["latent_dir"] = str(ROOT / "latent_analysis" / tag)
    cfg["audio_dir"]  = str(ROOT / "audio_samples" / tag)

    # File categories (each has its own folder)
    cfg["weights_path"]     = str(ROOT / "best_model" / tag / "best_model.pth")
    cfg["collapse_path"]    = str(ROOT / "posterior_collapse" / tag / "posterior_collapse.json")
    cfg["log_path"]         = str(ROOT / "training_log" / tag / "training_log.json")
    cfg["val_metrics_path"] = str(ROOT / "val_metrics" / tag / "val_metrics.json")

def ensure_dirs(cfg):
    to_make = [
        cfg["gen_dir"], cfg["latent_dir"], cfg["audio_dir"],
        Path(cfg["weights_path"]).parent,
        Path(cfg["collapse_path"]).parent,
        Path(cfg["log_path"]).parent,
        Path(cfg["val_metrics_path"]).parent,
    ]
    for p in to_make:
        Path(p).mkdir(parents=True, exist_ok=True)

# ========================== Small helpers ==========================

def save_grid(pats: np.ndarray, path: str, title: str = "", max_cols: int = 10):
    """pats: [N,16,9] in {0,1}"""
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

def _safe_xy(sample):
    """Handle (x,y,...) or x"""
    if isinstance(sample, (list, tuple)):
        if len(sample) >= 2:
            return sample[0], sample[1]
        return sample[0], None
    return sample, None

# ========================== Inline plugin equivalents ==========================

def plot_drum_pattern(pattern, title='Drum Pattern'):
    """Quick piano-roll figure for a [16,9] pattern."""
    if torch.is_tensor(pattern):
        pattern = pattern.detach().cpu().numpy()
    pattern = (pattern > 0.5).astype(np.float32)

    instruments = ["Kick","Snare","CHH","OHH","Tom1","Tom2","Ride","Crash","Perc"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for j in range(16):
        for i in range(9):
            if pattern[j, i] > 0.5:
                ax.add_patch(plt.Rectangle((j, i), 1, 1,
                    facecolor='tab:blue', edgecolor='black', linewidth=0.5))
    for j in range(17):
        ax.axvline(j, color='gray', linewidth=0.5, alpha=0.5)
        if j % 4 == 0:
            ax.axvline(j, color='black', linewidth=1.0, alpha=0.6)
    for i in range(10):
        ax.axhline(i, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xlim(0,16); ax.set_ylim(0,9)
    ax.set_xticks(range(16)); ax.set_xticklabels([str(i+1) for i in range(16)])
    ax.set_yticks(range(9));  ax.set_yticklabels(instruments)
    ax.set_xlabel('Time'); ax.set_ylabel('Instrument'); ax.set_title(title)
    ax.invert_yaxis(); fig.tight_layout()
    return fig

def drum_pattern_validity(pattern):
    """Heuristic plausibility score [0,1] for [16,9] pattern(s)."""
    if torch.is_tensor(pattern): x = pattern.detach().cpu()
    else: x = torch.tensor(pattern)
    if x.dim() == 3:
        return float(np.mean([drum_pattern_validity(x[i]) for i in range(x.size(0))]))
    x = (x > 0.5).to(torch.int32).numpy()
    score = 1.0
    density = float(x.mean())
    if density < 0.03 or density > 0.70: score *= 0.6
    elif density < 0.06 or density > 0.55: score *= 0.8
    if int((x.sum(axis=0) > 0).sum()) < 2: score *= 0.7
    if int(x[:,0].sum()) == 0: score *= 0.7                       # need kick
    if not (x[4,1] or x[12,1]): score *= 0.9                      # encourage backbeat
    if float((x[:8] == x[8:]).mean()) < 0.30: score *= 0.8        # half-bar repetition
    return float(np.clip(score, 0.0, 1.0))

def sequence_diversity(patterns):
    """Mean pairwise Hamming distance (≤100 samples)."""
    if torch.is_tensor(patterns):
        X = (patterns.detach().cpu() > 0.5).to(torch.int8).view(patterns.size(0), -1)
    else:
        X = (torch.tensor(patterns) > 0.5).to(torch.int8).view(len(patterns), -1)
    n = X.size(0)
    if n < 2: return 0.0
    m = min(100, n); dists = []
    for i in range(m):
        xi = X[i]
        for j in range(i+1, m):
            xj = X[j]
            dists.append((xi != xj).float().mean().item())
    return float(np.mean(dists)) if dists else 0.0

# ========================== Audio synth ==========================

def _sine(freq, n, sr):  return np.sin(2*np.pi*freq*(np.arange(n)/sr))
def _noise(n):           return np.random.randn(n)
def _env(n, attack=0.002, decay=0.05, sr=22050):
    a = int(max(1, attack*sr)); d = int(max(1, decay*sr)); s = max(0, n-a-d)
    env = np.concatenate([np.linspace(0,1,a,endpoint=False), np.ones(s), np.linspace(1,0,d,endpoint=True)])
    if env.size < n: env = np.pad(env, (0, n-env.size))
    return env[:n]

def pattern_to_wav(pattern: np.ndarray, path: str, bpm=120, bars=4, sr=22050):
    step_dur = 60.0/bpm/4.0; step_samps = int(sr*step_dur)
    total_samps = step_samps*16*bars; audio = np.zeros(total_samps, dtype=np.float32)
    freqs = [60, 180, 8000, 10000, 120, 200, 4000, 6000, 2200]
    is_noise = [0, 0, 1, 1, 0, 0, 1, 1, 1]
    for rep in range(bars):
        for step in range(16):
            start = (rep*16+step)*step_samps; end = start+step_samps
            for inst in range(9):
                if pattern[step, inst] > 0.5:
                    tone = _noise(step_samps) if is_noise[inst] else _sine(freqs[inst], step_samps, sr)
                    audio[start:end] += 0.2 * tone * _env(step_samps, sr=sr)
    m = np.max(np.abs(audio))
    if m>0: audio = 0.95*audio/m
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((audio*32767.0).astype(np.int16).tobytes())

# ========================== Posterior collapse (inline) ==========================

@torch.no_grad()
def posterior_collapse_report(model, loader: DataLoader, device: str, out_path: str,
                              high_thr: float = 0.02, low_thr: float = 0.02) -> dict:
    model.eval().to(device)
    mus_h, lvs_h, mus_l, lvs_l = [], [], [], []
    for batch in loader:
        x = batch if not isinstance(batch, (list, tuple)) else batch[0]
        x = x.to(device)
        z_l, mu_l, lv_l, z_h, mu_h, lv_h = model.encode_hierarchy(x)
        mus_h.append(mu_h.detach().cpu()); lvs_h.append(lv_h.detach().cpu())
        mus_l.append(mu_l.detach().cpu()); lvs_l.append(lv_l.detach().cpu())
    if not mus_h: raise RuntimeError("posterior_collapse_report: empty loader.")
    mu_h = torch.cat(mus_h, 0).numpy(); lv_h = torch.cat(lvs_h, 0).numpy()
    mu_l = torch.cat(mus_l, 0).numpy(); lv_l = torch.cat(lvs_l, 0).numpy()
    kl_high = 0.5 * (-lv_h + np.exp(lv_h) + mu_h**2 - 1.0)
    kl_low  = 0.5 * (-lv_l + np.exp(lv_l) + mu_l**2 - 1.0)
    kl_high_mean = kl_high.mean(axis=0); kl_low_mean = kl_low.mean(axis=0)
    collapsed_high = (kl_high_mean <= high_thr).nonzero()[0].astype(int).tolist()
    collapsed_low  = (kl_low_mean  <= low_thr).nonzero()[0].astype(int).tolist()
    report = {
        "kl_high_dim_mean": kl_high_mean.tolist(),
        "kl_low_dim_mean":  kl_low_mean.tolist(),
        "collapsed_high_indices": collapsed_high,
        "collapsed_low_indices":  collapsed_low,
        "thresholds": {"high": float(high_thr), "low": float(low_thr)},
        "num_samples": int(mu_h.shape[0]),
    }
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f: json.dump(report, f, indent=2)
    print(f"[saved] {out_path}")
    return report

# ========================== Dimension interpretation ==========================

def interpret_dimensions(model: HierarchicalDrumVAE, outdir: str, device: str,
                         sweep_vals: List[float] = (-2.0, 0.0, 2.0)) -> Dict:
    os.makedirs(outdir, exist_ok=True)
    model.eval().to(device)
    D = model.z_high_dim
    report = {"sweep_vals": list(map(float, sweep_vals)), "dims": []}
    for j in range(D):
        acts, imgs = [], []
        for v in sweep_vals:
            z_high = torch.zeros(1, D, device=device); z_high[0, j] = float(v)
            mu_p, logvar_p = model._prior_low_given_high(z_high)
            z_low = mu_p
            logits = model.decode_hierarchy(z_high, z_low=z_low)
            probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
            imgs.append((probs > 0.5).astype(np.float32))
            acts.append(probs.mean(axis=0).tolist())
        grid = np.stack(imgs, axis=0)
        save_grid(grid, os.path.join(outdir, f"z_high_dim{j}_sweep.png"),
                  title=f"z_high[{j}] sweep")
        report["dims"].append({"dim": j, "mean_activation_per_instrument": acts})
    with open(os.path.join(outdir, "dimension_interpretation.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report

# ========================== Eval helpers ==========================

def evaluate_elbo(model: HierarchicalDrumVAE, loader: DataLoader, device: str) -> dict:
    model.eval()
    tot_loss = tot_bce = tot_kll = tot_klh = 0.0; n = 0
    with torch.no_grad():
        for batch in loader:
            x = batch if not isinstance(batch, (list, tuple)) else batch[0]
            x = x.to(device)
            out = model(x, beta=1.0)
            B = x.size(0)
            tot_loss += float(out["loss"].item()) * B
            tot_bce  += float(out["recon_bce"].item()) * B
            tot_kll  += float(out["kl_low"].item()) * B
            tot_klh  += float(out["kl_high"].item()) * B
            n += B
    return {
        "loss": tot_loss / max(1, n),
        "recon_bce": tot_bce / max(1, n),
        "kl_low": tot_kll / max(1, n),
        "kl_high": tot_klh / max(1, n),
        "num_samples": n,
    }

def _encode_z_high(model, x, device):
    with torch.no_grad():
        z_l, mu_l, lv_l, z_h, mu_h, lv_h = model.encode_hierarchy(x.to(device))
    return mu_h  # [B, z_high_dim]

def _kick_jaccard(a, b):
    ka = a[:, 0] > 0.5; kb = b[:, 0] > 0.5
    inter = (ka & kb).sum(); union = (ka | kb).sum()
    return float(inter / max(1, union))

def _step_energy_corr(a, b):
    ea = a.sum(axis=1); eb = b.sum(axis=1)
    if ea.std() == 0 or eb.std() == 0:
        return 1.0 if np.allclose(ea, eb) else 0.0
    return float(np.corrcoef(ea, eb)[0, 1])

def step_style_eval(cfg, device, model, val_labeled_loader,
                    kick_thr=0.80, energy_thr=0.90):
    """
    讀取 generated_patterns/<tag>/style_transfer_*.npy，
    以 val-set 的 z_high 建 1-NN gallery 檢查 style 是否到位，
    並以 Kick Jaccard + step-energy correlation 檢查節奏是否保留。
    輸出到 generated_patterns/<tag>/style_transfer_eval.json
    """
    # 1) z_high gallery
    zs, ys = [], []
    for xb, yb in val_labeled_loader:
        mu_h = _encode_z_high(model, xb, device)
        zs.append(mu_h.cpu()); ys.append(yb.cpu())
    Z = torch.cat(zs, dim=0); Y = torch.cat(ys, dim=0)          # [N,D], [N]
    Zn = F.normalize(Z, p=2, dim=1)

    # 2) 掃檔
    gen_dir = cfg["gen_dir"]
    files = sorted(glob(os.path.join(gen_dir, "style_transfer_*_to_*.npy")))
    if not files:
        print(f"[StyleEval] No style_transfer_*.npy under {gen_dir}. Run --mode transfer first.")
        return None

    total=ok_style=ok_rhythm=ok_both=0; details=[]
    for fp in files:
        arr = np.load(fp)  # [3,16,9]: [src, transferred, tgt]
        src, trf, tgt = arr[0], arr[1], arr[2]

        # style：val gallery 1-NN（cosine）
        xt = torch.tensor(trf, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,16,9]
        zt = F.normalize(_encode_z_high(model, xt, device), p=2, dim=1).cpu()  # [1,D]
        sims = (Zn @ zt.T).squeeze(1)
        pred = int(Y[torch.argmax(sims)].item())

        target = int(os.path.basename(fp).split("_to_")[1].split(".")[0])
        style_ok = (pred == target)

        # rhythm：Kick Jaccard + step-energy corr
        j = _kick_jaccard(src, trf)
        r = _step_energy_corr(src, trf)
        rhythm_ok = (j >= kick_thr) and (r >= energy_thr)

        ok_style += int(style_ok); ok_rhythm += int(rhythm_ok)
        ok_both += int(style_ok and rhythm_ok); total += 1
        details.append({
            "file": os.path.basename(fp),
            "pred_style": pred, "target_style": target,
            "kick_jaccard": j, "energy_corr": r,
            "ok_style": bool(style_ok), "ok_rhythm": bool(rhythm_ok),
            "ok_both": bool(style_ok and rhythm_ok)
        })

    out = {
        "run_tag": cfg["run_tag"],
        "thresholds": {"kick_jaccard": kick_thr, "energy_corr": energy_thr},
        "counts": {"total_pairs": total, "style_success": ok_style,
                   "rhythm_success": ok_rhythm, "both_success": ok_both},
        "rates": {
            "style": ok_style/total if total else None,
            "rhythm": ok_rhythm/total if total else None,
            "both": ok_both/total if total else None
        },
        "details": details
    }
    with open(os.path.join(gen_dir, "style_transfer_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[StyleEval] total={total} style={out['rates']['style']:.2f} "
          f"rhythm={out['rates']['rhythm']:.2f} both={out['rates']['both']:.2f}")
    return out

# ========================== Step functions ==========================

def step_train(cfg, device, train_loader):
    model = HierarchicalDrumVAE(cfg["z_high_dim"], cfg["z_low_dim"]).to(device)
    print(f"[Device] torch.cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available(): print(f"[Device] GPU={torch.cuda.get_device_name(0)}")
    print(f"[Device] model on = {next(model.parameters()).device}")
    hist = train_hierarchical_vae(
        model=model, data_loader=train_loader, num_epochs=cfg["epochs"], device=device,
        lr=cfg["lr"], kl_mode=cfg["kl_mode"], kl_cycles=cfg["kl_cycles"],
        free_bits=cfg["free_bits"], use_temperature_anneal=cfg["use_temperature_anneal"],
    )
    Path(cfg["log_path"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["log_path"], "w") as f: json.dump({k: v for k, v in hist.items()}, f, indent=2)
    Path(cfg["weights_path"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg["weights_path"])
    print(f"[Saved] log → {cfg['log_path']}")
    print(f"[Saved] weights → {cfg['weights_path']}")
    return model

def load_or_init_model(cfg, device, require_weights=True):
    model = HierarchicalDrumVAE(cfg["z_high_dim"], cfg["z_low_dim"]).to(device)
    w = cfg["weights_path"]
    if os.path.isfile(w):
        model.load_state_dict(torch.load(w, map_location=device))
        print(f"[Info] Loaded weights: {w}")
    elif require_weights:
        raise FileNotFoundError(f"Cannot find weights at {w}. Run with --mode train first.")
    return model

def step_val(cfg, device, model, val_loader):
    metrics = evaluate_elbo(model, val_loader, device=device)
    Path(cfg["val_metrics_path"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["val_metrics_path"], "w") as f: json.dump(metrics, f, indent=2)
    print(f"[Val] saved → {cfg['val_metrics_path']}")

def step_gen_samples(cfg, device, model):
    n_styles, n_vars = 5, cfg["samples_per_style"]
    samples = sample_diverse_patterns(model, n_styles=n_styles, n_variations=n_vars,
                                      device=device, threshold=0.5).numpy()
    Path(cfg["gen_dir"]).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(cfg["gen_dir"], f"samples_5x{n_vars}.npy"), samples)
    flat = []
    for i in range(n_styles):
        np.save(os.path.join(cfg["gen_dir"], f"style_{i}_{n_vars}.npy"), samples[i])
        save_grid(samples[i], os.path.join(cfg["gen_dir"], f"style_{i}_{n_vars}.png"),
                  title=f"Style {i}: {n_vars} samples")
        flat.append(samples[i])
    save_grid(np.concatenate(flat, axis=0),
              os.path.join(cfg["gen_dir"], "all_styles_grid.png"),
              title=f"All styles ({n_vars} each)")
    print("[Gen] samples saved.")

def step_plugin_metrics_and_visuals(cfg):
    gen_dir = cfg["gen_dir"]
    samples_path = None
    for fn in os.listdir(gen_dir):
        if fn.startswith("samples_5x") and fn.endswith(".npy"):
            samples_path = os.path.join(gen_dir, fn); break
    if samples_path is None:
        raise FileNotFoundError(f"Need samples_5x*.npy under {gen_dir}. Run --mode gen first (or all).")
    arr = np.load(samples_path)            # [5,N,16,9]
    all_patterns = arr.reshape(-1, 16, 9)
    val_scores = [drum_pattern_validity(torch.tensor(p)) for p in all_patterns]
    validity_mean = float(np.mean(val_scores))
    diversity = sequence_diversity(torch.tensor(all_patterns))
    per_style_validity = []
    for i in range(arr.shape[0]):
        vs = [drum_pattern_validity(torch.tensor(p)) for p in arr[i]]
        per_style_validity.append(float(np.mean(vs)))
    out = {
        "validity_mean": validity_mean,
        "diversity": diversity,
        "per_style_validity_mean": per_style_validity,
        "num_samples": int(all_patterns.shape[0]),
    }
    with open(os.path.join(gen_dir, "plugin_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    # piano-roll
    for s in range(arr.shape[0]):
        for k in range(min(3, arr.shape[1])):
            fig = plot_drum_pattern(arr[s, k], title=f"Style {s} • Sample {k}")
            fig.savefig(os.path.join(gen_dir, f"style{s}_sample{k}_roll.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
    print("[Plugin] metrics & piano-roll saved.")

def step_interpolate(cfg, device, model, val_labeled_ds):
    x1, y1 = _safe_xy(val_labeled_ds[0]); idx2 = 1
    if y1 is not None:
        while idx2 < len(val_labeled_ds):
            x2, y2 = _safe_xy(val_labeled_ds[idx2])
            if y2 is None or int(y2) != int(y1): break
            idx2 += 1
    else:
        x2, _ = _safe_xy(val_labeled_ds[1])
    # Save under generated_patterns/<tag>/...
    save_prefix = f"generated_patterns/{cfg['run_tag']}/interpolate"
    Path(os.path.join(Path(cfg["results_root"]), "generated_patterns", cfg["run_tag"])).mkdir(parents=True, exist_ok=True)
    interpolate_styles(model, x1.unsqueeze(0), x2.unsqueeze(0),
                       n_steps=cfg["interpolate_steps"], device=device,
                       save_prefix=save_prefix)
    print("[Interpolate] sequences saved.")

def step_style_transfer(cfg, device, model, val_labeled_ds, max_pairs=5):
    buckets = {}
    for i in range(len(val_labeled_ds)):
        xi, yi = _safe_xy(val_labeled_ds[i])
        if yi is None: continue
        yi = int(yi)
        if yi not in buckets: buckets[yi] = xi
        if len(buckets) == 5: break
    styles = sorted(buckets.keys()); count = 0
    for a in styles:
        for b in styles:
            if a == b or count >= max_pairs: continue
            src, tgt = buckets[a], buckets[b]
            with torch.no_grad():
                z_l, mu_l, _, z_h, mu_h, _ = model.encode_hierarchy(
                    torch.stack([src, tgt], dim=0).to(device))
                pat = torch.sigmoid(model.decode_hierarchy(mu_h[1:2], z_low=mu_l[0:1]))
                out = (pat > 0.5).float().squeeze(0).cpu().numpy()
            trio = np.stack([src.squeeze(0).numpy(), out, tgt.squeeze(0).numpy()], axis=0)
            np.save(os.path.join(cfg["gen_dir"], f"style_transfer_{a}_to_{b}.npy"), trio)
            save_grid(trio, os.path.join(cfg["gen_dir"], f"style_transfer_{a}_to_{b}.png"),
                      title=f"Transfer: {a}->{b} (src, transferred, tgt)", max_cols=3)
            count += 1
    print("[Transfer] examples saved.")

def step_latent(cfg, device, model, val_labeled_loader):
    # 1) Generate t-SNE (analyze_latent saves to results root → we will move)
    visualize_latent_hierarchy(model, val_labeled_loader, device=device)
    # 2) Move to run-specific latent_dir
    for name in ("tsne_high.png", "tsne_low.png"):
        src = os.path.join(cfg["results_root"], name)  # analyze_latent default
        dst = os.path.join(cfg["latent_dir"], name)
        if os.path.exists(src):
            Path(cfg["latent_dir"]).mkdir(parents=True, exist_ok=True)
            if os.path.exists(dst): os.remove(dst)
            shutil.move(src, dst)
    # 3) Disentanglement metrics saved directly to run latent_dir
    dst_json_abs = os.path.abspath(os.path.join(cfg["latent_dir"], "disentangle_metrics.json"))
    measure_disentanglement(model, val_labeled_loader, device=device, save_name=dst_json_abs)
    # 4) Dimension interpretation
    interpret_dimensions(model, outdir=cfg["latent_dir"], device=device, sweep_vals=[-2.0, 0.0, 2.0])
    # 5) Posterior collapse per-run
    posterior_collapse_report(model, val_labeled_loader, device=device, out_path=cfg["collapse_path"])
    print("[Latent] t-SNE / disentangle / interpretation saved to", cfg["latent_dir"])

def step_plot_training_curves(cfg):
    with open(cfg["log_path"], "r") as f: hist = json.load(f)
    xs = list(range(1, len(hist["loss"])+1))
    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.plot(xs, hist["loss"], label="loss")
    ax.plot(xs, hist["recon_bce"], label="recon_bce")
    ax.plot(xs, hist["kl_low"], label="kl_low")
    ax.plot(xs, hist["kl_high"], label="kl_high")
    ax.set_xlabel("epoch"); ax.set_title("Training Curves"); ax.legend()
    fig.tight_layout()
    out = os.path.join(cfg["latent_dir"], "training_curves.png")
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f"[Curves] saved → {out}")

def step_audio(cfg):
    gen_dir = cfg["gen_dir"]; Path(cfg["audio_dir"]).mkdir(parents=True, exist_ok=True)
    samples_path = None
    for fn in os.listdir(gen_dir):
        if fn.startswith("samples_5x") and fn.endswith(".npy"):
            samples_path = os.path.join(gen_dir, fn); break
    if samples_path is None:
        raise FileNotFoundError(f"Need samples_5x*.npy under {gen_dir}. Run --mode gen first (or all).")
    arr = np.load(samples_path)  # [5, N, 16, 9]
    for i in range(arr.shape[0]):
        for k in range(min(cfg["audio_per_style"], arr.shape[1])):
            wav_path = os.path.join(cfg["audio_dir"], f"style{i}_sample{k}.wav")
            pattern_to_wav(arr[i,k], wav_path, bpm=cfg["bpm"], bars=4, sr=22050)
    print("[Audio] wav files saved.")

# ========================== CLI/Main ==========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="all",
        choices=["all","train","gen","interpolate","transfer","latent","audio","val","plugin","curves","styleeval"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-train", dest="skip_train", action="store_true",
                        help="Skip training and load saved weights")
    parser.add_argument("--run-tag", type=str, default="", help="store outputs under results/<category>/<tag>")
    parser.add_argument("--kl-mode", type=str, choices=["cyclical", "linear", "constant"])
    parser.add_argument("--free-bits", type=float)
    args = parser.parse_args()

    # config
    cfg = default_config()
    if args.epochs is not None: cfg["epochs"] = args.epochs
    if args.kl_mode: cfg["kl_mode"] = args.kl_mode
    if args.free_bits is not None: cfg["free_bits"] = args.free_bits

    # run-aware paths
    apply_run_paths(cfg, args.run_tag)
    ensure_dirs(cfg)

    device = cfg["device"]

    # DataLoaders
    train_loader, val_loader = get_drum_dataloaders(
        batch_size=cfg["batch_size"], add_channel_dim=True, return_label=False,
        shuffle_train=True, num_workers=2, pin_memory=torch.cuda.is_available()
    )
    val_labeled_loader = DataLoader(
        DrumPatternDataset(split='val', add_channel_dim=True, return_label=True),
        batch_size=256, shuffle=False
    )
    val_labeled_ds = val_labeled_loader.dataset

    # Train or load
    if args.mode in ("all","train") and not args.skip_train:
        model = step_train(cfg, device, train_loader)
        step_val(cfg, device, model, val_loader)
    else:
        model = load_or_init_model(cfg, device, require_weights=True)
        if args.mode == "val":
            step_val(cfg, device, model, val_loader)

    # Artifacts
    if args.mode in ("all","gen"):
        step_gen_samples(cfg, device, model)
        step_plugin_metrics_and_visuals(cfg)

    if args.mode in ("all","interpolate"):
        step_interpolate(cfg, device, model, val_labeled_ds)

    if args.mode in ("all","transfer"):
        step_style_transfer(cfg, device, model, val_labeled_ds)

    if args.mode in ("all", "styleeval"):
        step_style_eval(cfg, device, model, val_labeled_loader)

    if args.mode in ("all","latent"):
        step_latent(cfg, device, model, val_labeled_loader)

    if args.mode in ("all","curves"):
        step_plot_training_curves(cfg)

    if args.mode in ("all","audio"):
        step_audio(cfg)

    print(f"[Done] Mode={args.mode}. Artifacts under {cfg['results_root']} (tag={cfg['run_tag']})")

if __name__ == "__main__":
    main()
