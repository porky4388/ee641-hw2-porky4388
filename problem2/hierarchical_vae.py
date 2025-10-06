"""
Hierarchical VAE for drum pattern generation.

- q(z_low | x)  with Conv1d encoder
- q(z_high | z_low) with MLP
- p(z_low | z_high) with MLP  (conditional prior)
- p(x | z_low, z_high) with ConvTranspose1d decoder (logits → Bernoulli)

Input x can be [B, 16, 9] or [B, 1, 16, 9].
Output recon is sigmoid(logits) with shape [B, 16, 9].
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _kld_normal(mu_q, logvar_q, mu_p=None, logvar_p=None):
    """
    KL( N(mu_q, var_q) || N(mu_p, var_p) ), elementwise sum over dim.
    mu_p/logvar_p default to 0 if None (i.e., standard Normal prior).
    """
    if mu_p is None:
        mu_p = torch.zeros_like(mu_q)
    if logvar_p is None:
        logvar_p = torch.zeros_like(logvar_q)

    # log( var_q / var_p ) + (var_q + (mu_q - mu_p)^2)/var_p - 1
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kld = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)
    return kld.sum(dim=1)  # sum over latent dims → [B]


class HierarchicalDrumVAE(nn.Module):
    def __init__(self, z_high_dim: int = 4, z_low_dim: int = 12):
        """
        Two-level VAE for drum patterns.

        z_high : style/genre
        z_low  : pattern variation (conditioned on z_high in the prior)

        Args:
            z_high_dim: Dimension of high-level latent (style)
            z_low_dim:  Dimension of low-level latent (variation)
        """
        super().__init__()
        self.z_high_dim = z_high_dim
        self.z_low_dim = z_low_dim

        # ---------------- Encoder: x -> z_low ----------------
        # Treat pattern as sequence of length 16 with 9 channels
        # Input to Conv1d should be [B, 9, 16]
        self.encoder_low = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=3, padding=1),            # [B, 32, 16]
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), # [B, 64, 8]
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),# [B,128, 4]
            nn.ReLU(inplace=True),
            nn.Flatten(),                                          # [B, 128*4=512]
        )
        self.fc_mu_low     = nn.Linear(512, z_low_dim)
        self.fc_logvar_low = nn.Linear(512, z_low_dim)

        # ---------------- Encoder: z_low -> z_high ------------
        self.encoder_high = nn.Sequential(
            nn.Linear(z_low_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_mu_high     = nn.Linear(32, z_high_dim)
        self.fc_logvar_high = nn.Linear(32, z_high_dim)

        # ---------------- Conditional prior: p(z_low | z_high) -------------
        self.prior_low_net = nn.Sequential(
            nn.Linear(z_high_dim, 64),
            nn.ReLU(inplace=True),
        )
        self.prior_mu_low     = nn.Linear(64, z_low_dim)
        self.prior_logvar_low = nn.Linear(64, z_low_dim)

        # ---------------- Decoder: (z_low, z_high) -> x --------------------
        # Concatenate latents then up-project to [B,128,4], then ConvT to length 16.
        dec_in = z_low_dim + z_high_dim
        self.decoder_fc = nn.Sequential(
            nn.Linear(dec_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),   # 128 * 4
            nn.ReLU(inplace=True),
        )
        self.dec_deconv = nn.Sequential(
            # start [B, 128, 4]
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # -> [B,64,8]
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # -> [B,32,16]
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 9, kernel_size=3, padding=1),                       # -> [B, 9, 16] (logits)
        )

    # ---------------- utilities ----------------

    @staticmethod
    def _canonicalize_input(x: torch.Tensor) -> torch.Tensor:
        """
        Accept [B, 16, 9] or [B, 1, 16, 9]; return [B, 9, 16] for Conv1d.
        """
        if x.dim() == 4:
            # [B, 1, 16, 9] -> [B, 16, 9]
            if x.size(1) != 1:
                raise ValueError(f"Expected channel=1 for 4D input, got {x.size()}")
            x = x.squeeze(1)
        if x.dim() != 3 or x.size(1) != 16 or x.size(2) != 9:
            raise ValueError(f"Expected x shape [B,16,9] or [B,1,16,9], got {tuple(x.size())}")
        # to [B, 9, 16]
        return x.transpose(1, 2).contiguous().float()

    @staticmethod
    def _to_bernoulli_target(x: torch.Tensor) -> torch.Tensor:
        """
        Return target in shape [B,16,9], float in [0,1].
        Accepts [B,1,16,9] or [B,16,9].
        """
        if x.dim() == 4:
            x = x.squeeze(1)
        return x.float()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        z = mu + std * eps, eps ~ N(0, I)
        """
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---------------- main pieces ----------------

    def encode_hierarchy(self, x: torch.Tensor):
        """
        Encode pattern to both latent levels.

        Args:
            x: [B,16,9] or [B,1,16,9]

        Returns:
            z_low, mu_low, logvar_low,
            z_high, mu_high, logvar_high
        """
        x_1d = self._canonicalize_input(x)        # [B,9,16]
        h = self.encoder_low(x_1d)                # [B,512]
        mu_low, logvar_low = self.fc_mu_low(h), self.fc_logvar_low(h)
        z_low = self.reparameterize(mu_low, logvar_low)

        g = self.encoder_high(z_low)              # [B,32]
        mu_high, logvar_high = self.fc_mu_high(g), self.fc_logvar_high(g)
        z_high = self.reparameterize(mu_high, logvar_high)

        return z_low, mu_low, logvar_low, z_high, mu_high, logvar_high

    def _prior_low_given_high(self, z_high: torch.Tensor):
        """
        Conditional prior parameters for z_low given z_high.
        Returns mu_p, logvar_p with shape [B, z_low_dim].
        """
        p = self.prior_low_net(z_high)
        return self.prior_mu_low(p), self.prior_logvar_low(p)

    def decode_hierarchy(self, z_high: torch.Tensor, z_low: torch.Tensor | None = None,
                         temperature: float = 1.0):
        """
        Decode to pattern logits [B,16,9].

        If z_low is None, sample it from p(z_low|z_high).
        """
        if z_low is None:
            mu_p, logvar_p = self._prior_low_given_high(z_high)
            z_low = self.reparameterize(mu_p, logvar_p)

        z = torch.cat([z_low, z_high], dim=1)     # [B, z_low+z_high]
        h = self.decoder_fc(z)                    # [B,512]
        h = h.view(h.size(0), 128, 4)             # [B,128,4]
        logits = self.dec_deconv(h)               # [B, 9,16]
        logits = logits / max(1e-6, float(temperature))
        # Return [B,16,9]
        return logits.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor, beta: float = 1.0):
        """
        Full forward with losses.

        Args:
            x:    [B,16,9] or [B,1,16,9] with values in {0,1}
            beta: KL weight (β-VAE). Use <1 to reduce collapse risk.

        Returns:
            dict with:
              - recon: sigmoid(logits) in [0,1], shape [B,16,9]
              - loss: scalar
              - recon_bce, kl_low, kl_high: scalars
              - mu/logvar for both levels (for analysis)
        """
        target = self._to_bernoulli_target(x)     # [B,16,9]

        # Encode
        z_low, mu_low, logvar_low, z_high, mu_high, logvar_high = self.encode_hierarchy(x)

        # Conditional prior for z_low
        mu_p_low, logvar_p_low = self._prior_low_given_high(z_high)

        # Decode
        logits = self.decode_hierarchy(z_high, z_low=z_low)   # [B,16,9]
        recon = torch.sigmoid(logits)

        # Reconstruction (Bernoulli with logits)
        # F.binary_cross_entropy_with_logits expects input logits and target probs
        recon_bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="sum"
        ) / x.size(0)  # average over batch

        # KL terms
        kl_low  = _kld_normal(mu_low,  logvar_low,  mu_p_low, logvar_p_low).mean()
        kl_high = _kld_normal(mu_high, logvar_high, None, None).mean()

        loss = recon_bce + beta * (kl_low + kl_high)

        return {
            "recon": recon,
            "loss": loss,
            "recon_bce": recon_bce,
            "kl_low": kl_low,
            "kl_high": kl_high,
            "mu_low": mu_low, "logvar_low": logvar_low,
            "mu_high": mu_high, "logvar_high": logvar_high,
        }
