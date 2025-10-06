"""
Training implementations for hierarchical VAE with posterior collapse prevention.
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# -------------------------- Utilities --------------------------

def _as_x(batch):
    """Support dataloaders that return x or (x, ...)"""
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch

def _kld_normal(mu_q, logvar_q, mu_p=None, logvar_p=None):
    """
    KL( N(mu_q, var_q) || N(mu_p, var_p) ) summed over last dim → [B]
    Default prior: N(0, I) if mu_p/logvar_p is None.
    """
    if mu_p is None:
        mu_p = torch.zeros_like(mu_q)
    if logvar_p is None:
        logvar_p = torch.zeros_like(logvar_q)
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kld = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
    return kld.sum(dim=-1)  # [B]


# -------------------------- Schedules --------------------------

def make_kl_anneal_schedule(total_epochs: int,
                            mode: str = "cyclical",
                            n_cycles: int = 4,
                            start: float = 0.0,
                            end: float = 1.0,
                            ratio: float = 0.5):
    """
    Create a function epoch -> beta.
    - cyclical: beta goes start->end within each cycle, flat 0 the rest portion.
      'ratio' controls the increasing portion of each cycle (e.g., 0.5 means
      first 50% epochs in a cycle increase linearly to 'end', then stay at 'end').
    - linear: single linear warmup to 'end' then keep.
    """

    if mode == "linear":
        def sched(epoch: int) -> float:
            t = min(1.0, max(0.0, epoch / max(1, total_epochs - 1)))
            return float(start + (end - start) * t)
        return sched

    # cyclical
    cycle_len = max(1, total_epochs // max(1, n_cycles))

    def sched(epoch: int) -> float:
        cycle_idx = epoch % cycle_len
        ramp_len = int(cycle_len * ratio)
        if ramp_len <= 0:
            return end
        if cycle_idx < ramp_len:
            t = cycle_idx / max(1, ramp_len - 1)
            return float(start + (end - start) * t)
        return float(end)

    return sched


def make_temperature_schedule(total_epochs: int,
                              start_t: float = 1.5,
                              end_t: float = 1.0):
    """Linear temperature annealing: epoch -> temperature."""
    def sched(epoch: int) -> float:
        t = min(1.0, max(0.0, epoch / max(1, total_epochs - 1)))
        return float(start_t + (end_t - start_t) * t)
    return sched


# -------------------------- Training --------------------------

def train_hierarchical_vae(model,
                           data_loader: DataLoader,
                           num_epochs: int = 100,
                           device: str = 'cuda',
                           lr: float = 2e-3,
                           kl_mode: str = "cyclical",
                           kl_cycles: int = 4,
                           free_bits: float = 0.5,
                           use_temperature_anneal: bool = True) -> Dict[str, List[float]]:
    """
    Train hierarchical VAE with:
      - KL annealing
      - Free bits per latent dimension
      - (optional) temperature annealing

    Assumptions:
      - `model` implements:
          encode_hierarchy(x) -> z_low, mu_low, logvar_low, z_high, mu_high, logvar_high
          decode_hierarchy(z_high, z_low=None, temperature=1.0) -> logits [B,16,9]
          _prior_low_given_high(z_high) -> (mu_p_low, logvar_p_low)
      - Input batches are x in shape [B,1,16,9] or [B,16,9] with {0,1}

    Returns:
      history dict with lists for 'loss','recon_bce','kl_low','kl_high','beta','temperature'
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    kl_sched = make_kl_anneal_schedule(num_epochs, mode=kl_mode, n_cycles=kl_cycles)
    temp_sched = make_temperature_schedule(num_epochs) if use_temperature_anneal else (lambda e: 1.0)

    history = defaultdict(list)

    for epoch in range(num_epochs):
        beta = float(kl_sched(epoch))
        temperature = float(temp_sched(epoch))

        epoch_loss = epoch_bce = epoch_kll = epoch_klh = 0.0
        n_samples = 0

        for batch in data_loader:
            x = _as_x(batch).to(device)          # [B,1,16,9] or [B,16,9]
            B = x.size(0)
            n_samples += B

            # ----- Encode -----
            z_low, mu_low, logvar_low, z_high, mu_high, logvar_high = model.encode_hierarchy(x)

            # Conditional prior for z_low
            mu_p_low, logvar_p_low = model._prior_low_given_high(z_high)

            # ----- Decode with temperature anneal -----
            logits = model.decode_hierarchy(z_high, z_low=z_low, temperature=temperature)  # [B,16,9]
            target = x.squeeze(1) if x.dim() == 4 else x
            target = target.float()

            # Reconstruction BCE with logits
            recon_bce = F.binary_cross_entropy_with_logits(logits, target, reduction="sum") / B

            # ----- KL terms -----
            # Per-sample KL, then average
            kl_low_all = _kld_normal(mu_low, logvar_low, mu_p_low, logvar_p_low)   # [B]
            kl_high_all = _kld_normal(mu_high, logvar_high, None, None)            # [B]

            # Free bits per-dimension:
            # compute elementwise kl per dim, clamp, then sum and mean over batch
            kl_low_dim = 0.5 * (
                (logvar_p_low - logvar_low)
                + (logvar_low.exp() + (mu_low - mu_p_low) ** 2) / logvar_p_low.exp()
                - 1.0
            )  # [B, z_low_dim]
            kl_high_dim = 0.5 * (
                (-logvar_high)
                + (logvar_high.exp() + (mu_high ** 2))
                - 1.0
            )  # [B, z_high_dim] (prior is std normal)

            # clamp with free bits (nats)
            kl_low_fb = torch.clamp(kl_low_dim, min=free_bits).sum(dim=-1).mean()
            kl_high_fb = torch.clamp(kl_high_dim, min=free_bits).sum(dim=-1).mean()

            # total loss (use fb'd KLs for optimization; keep raw KL for logging)
            loss = recon_bce + beta * (kl_low_fb + kl_high_fb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * B
            epoch_bce  += float(recon_bce.item()) * B
            epoch_kll  += float(kl_low_all.mean().item()) * B
            epoch_klh  += float(kl_high_all.mean().item()) * B

        # epoch averages
        history['loss'].append(epoch_loss / n_samples)
        history['recon_bce'].append(epoch_bce / n_samples)
        history['kl_low'].append(epoch_kll / n_samples)
        history['kl_high'].append(epoch_klh / n_samples)
        history['beta'].append(beta)
        history['temperature'].append(temperature)

        print(f"[Epoch {epoch+1:03d}/{num_epochs}] "
              f"loss={history['loss'][-1]:.4f} | "
              f"bce={history['recon_bce'][-1]:.4f} | "
              f"kl_low={history['kl_low'][-1]:.4f} | "
              f"kl_high={history['kl_high'][-1]:.4f} | "
              f"beta={beta:.3f} T={temperature:.2f}")

    return history


# -------------------------- Sampling --------------------------

@torch.no_grad()
def sample_diverse_patterns(model,
                            n_styles: int = 5,
                            n_variations: int = 8,
                            device: str = 'cuda',
                            threshold: float = 0.5) -> torch.Tensor:
    """
    Generate a grid of patterns with consistent high-level style.

    Returns:
        patterns_bin: [n_styles, n_variations, 16, 9] (0/1 float tensor on CPU)
    """
    model.eval()
    model = model.to(device)

    # Sample z_high from N(0, I)
    z_high = torch.randn(n_styles, model.z_high_dim, device=device)

    out = []
    for i in range(n_styles):
        row = []
        for _ in range(n_variations):
            # decode_hierarchy will sample z_low ~ p(z_low|z_high) if None
            logits = model.decode_hierarchy(z_high[i:i+1], z_low=None)  # [1,16,9]
            probs = torch.sigmoid(logits)
            patt = (probs > threshold).float()  # binarize
            row.append(patt.squeeze(0))         # [16,9]
        out.append(torch.stack(row, dim=0))     # [n_variations,16,9]

    patterns_bin = torch.stack(out, dim=0).cpu()  # [n_styles,n_variations,16,9]
    return patterns_bin


# -------------------------- Posterior Collapse Analysis --------------------------

@torch.no_grad()
def analyze_posterior_collapse(model,
                               data_loader: DataLoader,
                               max_batches: int | None = 50,
                               device: str = 'cuda',
                               collapse_threshold: float = 1e-2) -> Dict[str, Any]:
    """
    Diagnose latent utilization by average per-dimension KL (nats).
    A dimension is considered "collapsed" if its mean KL < threshold.

    Returns:
        {
          'kl_low_dim_mean':  [z_low_dim] list,
          'kl_high_dim_mean': [z_high_dim] list,
          'collapsed_low_idx':  [...],
          'collapsed_high_idx': [...]
        }
    """
    model.eval()
    model = model.to(device)

    sum_low = None
    sum_high = None
    count = 0

    for b_idx, batch in enumerate(data_loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        x = _as_x(batch).to(device)

        z_low, mu_low, logvar_low, z_high, mu_high, logvar_high = model.encode_hierarchy(x)
        mu_p_low, logvar_p_low = model._prior_low_given_high(z_high)

        # per-dim KL
        kl_low_dim = 0.5 * (
            (logvar_p_low - logvar_low)
            + (logvar_low.exp() + (mu_low - mu_p_low) ** 2) / logvar_p_low.exp()
            - 1.0
        )  # [B, z_low_dim]
        kl_high_dim = 0.5 * (
            (-logvar_high)
            + (logvar_high.exp() + (mu_high ** 2))
            - 1.0
        )  # [B, z_high_dim]

        batch_sum_low = kl_low_dim.mean(dim=0)   # [z_low_dim]
        batch_sum_high = kl_high_dim.mean(dim=0) # [z_high_dim]

        if sum_low is None:
            sum_low = batch_sum_low
            sum_high = batch_sum_high
        else:
            sum_low += batch_sum_low
            sum_high += batch_sum_high

        count += 1

    kl_low_mean = (sum_low / max(1, count)).detach().cpu().tolist()
    kl_high_mean = (sum_high / max(1, count)).detach().cpu().tolist()

    collapsed_low = [i for i, v in enumerate(kl_low_mean) if v < collapse_threshold]
    collapsed_high = [i for i, v in enumerate(kl_high_mean) if v < collapse_threshold]

    return {
        "kl_low_dim_mean": kl_low_mean,
        "kl_high_dim_mean": kl_high_mean,
        "collapsed_low_idx": collapsed_low,
        "collapsed_high_idx": collapsed_high,
        "threshold": collapse_threshold,
        "batches_evaluated": count,
    }
