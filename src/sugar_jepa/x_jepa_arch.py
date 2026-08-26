"""X-CGM-JEPA architecture. Self-contained, no repo-internal imports."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def _sinusoidal_table(n_positions, dim):
    pe = torch.zeros(n_positions, dim)
    pos = torch.arange(0, n_positions, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class JepaBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class JepaEncoder(nn.Module):
    def __init__(self, n_time_steps, patch_size=8, embed_dim=96, n_layers=3, n_heads=6,
                 mlp_ratio=4.0, dropout=0.0, norm="instance"):
        super().__init__()
        if n_time_steps % patch_size != 0:
            raise ValueError(f"n_time_steps ({n_time_steps}) must be divisible by patch_size ({patch_size})")
        if norm not in ("instance", "none"):
            raise ValueError(f"norm must be 'instance' or 'none', got {norm!r}")

        self.n_patches = n_time_steps // patch_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.norm_mode = norm

        self.patch_embed = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=self.n_patches)
        self.blocks = nn.ModuleList(
            [JepaBlock(embed_dim, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, glucose, keep=None):
        if self.norm_mode == "instance":
            mean = glucose.mean(dim=1, keepdim=True)
            std = glucose.std(dim=1, keepdim=True).clamp_min(1e-6)
            glucose = (glucose - mean) / std

        x = self.patch_embed(glucose.unsqueeze(1)).transpose(1, 2)
        x = self.pos_enc(x)
        if keep is not None:
            x = x.gather(1, keep.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class JepaPredictor(nn.Module):
    def __init__(self, embed_dim, n_patches, pred_dim=None, n_layers=2, n_heads=4,
                 mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        pred_dim = pred_dim or max(embed_dim // 2, n_heads)
        self.pred_dim = pred_dim
        self.in_proj = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos_table", _sinusoidal_table(n_patches, pred_dim))
        self.blocks = nn.ModuleList(
            [JepaBlock(pred_dim, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(pred_dim)
        self.out_proj = nn.Linear(pred_dim, embed_dim)

    def forward(self, context, context_idx, target_idx):
        batch, n_ctx, _ = context.shape
        n_tgt = target_idx.numel()
        ctx = self.in_proj(context) + self.pos_table[context_idx].unsqueeze(0)
        tgt = self.mask_token.expand(batch, n_tgt, -1) + self.pos_table[target_idx].unsqueeze(0)
        x = torch.cat([ctx, tgt], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, n_ctx:, :])
        return self.out_proj(x)


class GlucoEncoder(nn.Module):
    def __init__(self, gridsize=32, patch=8, in_ch=3, embed_dim=96,
                 n_layers=3, n_heads=6, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.patch = patch
        self.n_patches = (gridsize // patch) ** 2
        self.embed = nn.Linear(patch * patch * in_ch, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len=self.n_patches)
        self.blocks = nn.ModuleList(
            [JepaBlock(embed_dim, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def patchify(self, img):
        b, h, w, c = img.shape
        p = self.patch
        x = img.reshape(b, h // p, p, w // p, p, c).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(b, (h // p) * (w // p), p * p * c)

    def forward(self, img, keep=None):
        x = self.pos_enc(self.embed(self.patchify(img)))
        if keep is not None:
            x = x.gather(1, keep.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class GluPredictor(nn.Module):
    def __init__(self, num_gluco_patches=16, embed_dim=96, pred_dim=48,
                 n_layers=1, n_heads=2, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.num_gluco_patches = num_gluco_patches
        self.predictor_embed = nn.Linear(embed_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", _sinusoidal_table(num_gluco_patches, pred_dim))
        self.blocks = nn.ModuleList(
            [JepaBlock(pred_dim, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(pred_dim)
        self.predictor_proj = nn.Linear(pred_dim, embed_dim)

    def forward(self, cgm_context, gluco_masks=None):
        b, lc, _ = cgm_context.shape
        x = self.predictor_embed(cgm_context)
        blanks = self.mask_token.expand(b, self.num_gluco_patches, -1) + self.pos.unsqueeze(0)
        x = torch.cat([x, blanks], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, lc:]
        if gluco_masks is not None:
            idx = gluco_masks.unsqueeze(-1).expand(-1, -1, x.size(-1))
            x = x.gather(1, idx)
        return self.predictor_proj(x)


def sample_block_mask(n_patches, n_targets, min_block, max_block, rng, max_attempts=100):
    for _ in range(max_attempts):
        occupied = set()
        placed = 0
        for _ in range(n_targets):
            size = rng.randint(min_block, max_block)
            for _ in range(20):
                start = rng.randint(0, n_patches - size)
                block = set(range(start, start + size))
                if not (block & occupied):
                    occupied |= block
                    placed += 1
                    break
        context = [i for i in range(n_patches) if i not in occupied]
        if placed == n_targets and context:
            return context, sorted(occupied)
    raise RuntimeError(f"Could not place {n_targets} blocks of {min_block}-{max_block} in {n_patches} patches")


@torch.no_grad()
def ema_update(target, online, momentum):
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.mul_(momentum).add_(p_o.detach(), alpha=1.0 - momentum)
    for b_t, b_o in zip(target.buffers(), online.buffers()):
        b_t.copy_(b_o)


def momentum_at(step, total_steps, base, final=1.0):
    if total_steps <= 1:
        return final
    progress = min(step / (total_steps - 1), 1.0)
    return final - (final - base) * (math.cos(math.pi * progress) + 1) / 2


@torch.no_grad()
def collapse_metrics(latents):
    """(latent_std, effective_rank) of an encoder's output, for collapse watch."""
    z = latents.reshape(-1, latents.size(-1)).float()
    std = z.std(dim=0).mean().item()

    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(z.size(0) - 1, 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    denom = (ev**2).sum()
    eff_rank = (ev.sum() ** 2 / denom).item() if denom > 0 else 0.0
    return std, eff_rank


def variance_penalty(latents, target_std):
    """VICReg-style hinge: penalise any latent dim whose std falls below target_std."""
    z = latents.reshape(-1, latents.size(-1))
    std = torch.sqrt(z.var(dim=0) + 1e-8)
    return F.relu(target_std - std).mean()


def sigreg(z, num_slices=256, k=17):
    n, d = z.shape
    a = torch.randn(d, num_slices, device=z.device, dtype=z.dtype)
    a = a / a.norm(dim=0, keepdim=True)
    proj = z @ a
    t = torch.linspace(-5, 5, k, device=z.device, dtype=z.dtype)
    phi = torch.exp(-0.5 * t ** 2)
    xt = proj.unsqueeze(-1) * t
    re = torch.cos(xt).mean(0)
    im = torch.sin(xt).mean(0)
    diff_sq = (re - phi) ** 2 + im ** 2
    per_dir = torch.trapz(diff_sq * phi, t, dim=1)
    return per_dir.mean()


def cgm_forward_loss(encoder, target_encoder, predictor, glucose, ctx_idx, tgt_idx,
                      var_weight=0.0, var_target=0.5):
    """Verbatim port of jepa_pretrain.py's _forward_loss — the plain CGM-only
    JEPA objective, unchanged. Returns (total, pred_loss, var_loss, full, context)."""
    batch = glucose.size(0)

    with torch.no_grad():
        full = target_encoder(glucose)
        targets = full[:, tgt_idx, :].detach()

    keep = ctx_idx.unsqueeze(0).expand(batch, -1)
    context = encoder(glucose, keep=keep)
    pred = predictor(context, ctx_idx, tgt_idx)
    pred_loss = F.smooth_l1_loss(pred, targets)

    if var_weight > 0.0:
        var_loss = variance_penalty(context, var_target)
    else:
        var_loss = torch.zeros((), device=pred_loss.device, dtype=pred_loss.dtype)

    return pred_loss + var_weight * var_loss, pred_loss.detach(), var_loss.detach(), full.detach(), context


def x_forward_loss(
    cgm_encoder, cgm_encoder_ema, cgm_predictor,
    glu_encoder, glu_predictor,
    glucose, gluco_img,
    cgm_ctx_idx, cgm_tgt_idx, gluco_tgt_idx,
    gluco_loss_weight=1.0,
    sigreg_weight=0.0,
    cgm_var_weight=0.0,
    cgm_var_target=0.5,
):
    """CGM half is exactly cgm_forward_loss (plain jepa_pretrain.py's objective,
    untouched); the glucodensity half is a pure addition on top — cgm_context is
    detached before it reaches glu_predictor, so no gradient from the cross-modal
    loss ever reaches cgm_encoder. Returns (total, cgm_loss, gluco_loss, reg,
    cgm_var_loss, cgm_full) — cgm_full is the EMA target's full output, for
    collapse_metrics.
    """
    cgm_total, cgm_loss, cgm_var_loss, cgm_full, cgm_context = cgm_forward_loss(
        cgm_encoder, cgm_encoder_ema, cgm_predictor,
        glucose, cgm_ctx_idx, cgm_tgt_idx,
        var_weight=cgm_var_weight, var_target=cgm_var_target,
    )
    cgm_context = cgm_context.detach()

    glu_full = glu_encoder(gluco_img)
    reg = sigreg(glu_full.reshape(-1, glu_full.size(-1))) if sigreg_weight > 0 else glu_full.new_zeros(())
    glu_full = F.layer_norm(glu_full, (glu_full.size(-1),))
    glu_targets = glu_full[:, gluco_tgt_idx, :]

    gluco_masks = gluco_tgt_idx.unsqueeze(0).expand(cgm_context.size(0), -1)
    glu_pred = glu_predictor(cgm_context, gluco_masks)
    gluco_loss = F.l1_loss(glu_pred, glu_targets)

    total = cgm_total + gluco_loss_weight * gluco_loss + sigreg_weight * reg
    return total, cgm_loss, gluco_loss.detach(), reg.detach(), cgm_var_loss, cgm_full
