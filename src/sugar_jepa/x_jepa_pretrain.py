"""X-CGM-JEPA pretraining.

Step 1: GlucoEncoder  — encodes a glucodensity image into patch embeddings.
Step 2: GluPredictor  — cross-modal predictor (P_Glu): predicts masked glucodensity
        patches from the CGM context.
Step 3: x_forward_loss — one training block: shared CGM context drives both
        predictors; total loss = L_CGM + w * L_Glu. Faithful to CGM-JEPA
        pretrain/pretrain_x_cgm_jepa.py.

The CGM half reuses jepa_pretrain.py / sugar_jepa_model.py unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from sugar_jepa.jepa_pretrain import _sinusoidal_table
from sugar_jepa.sugar_jepa_model import JepaBlock, PositionalEncoding


class GlucoEncoder(nn.Module):
    """(batch, 32, 32, 3) glucodensity image -> (batch, 16, embed_dim) patch embeddings."""

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
    """P_Glu — cross-modal predictor. Faithful port of CGM-JEPA
    models/cross_modal_predictor.py (CrossModalPredictor).

    Input is the CGM encoder context only. The glucodensity patches enter as
    learned mask tokens + positional encoding (no glucodensity embedding, no
    X_Glu / proj_b). The predictor maps CGM -> glucodensity latents; the loss is
    taken against the target-encoder glucodensity latents at the masked patches.

    Local substitutions vs upstream: JepaBlock for their Block,
    _sinusoidal_table for their PositionalEmbedding.
    """

    def __init__(self, num_gluco_patches=16, embed_dim=96, pred_dim=48,
                 n_layers=1, n_heads=2, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.num_gluco_patches = num_gluco_patches
        self.predictor_embed = nn.Linear(embed_dim, pred_dim)      # CGM context -> pred width
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("pos", _sinusoidal_table(num_gluco_patches, pred_dim))
        self.blocks = nn.ModuleList(
            [JepaBlock(pred_dim, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(pred_dim)
        self.predictor_proj = nn.Linear(pred_dim, embed_dim)       # pred width -> encoder width

    def forward(self, cgm_context, gluco_masks=None):
        """
        cgm_context  (B, Lc, embed_dim)  CGM encoder context patches
        gluco_masks  (B, K) int or None  masked glucodensity indices to keep for the loss;
                                         None returns all num_gluco_patches
        returns      (B, K, embed_dim)   predicted glucodensity latents
        """
        b, lc, _ = cgm_context.shape
        x = self.predictor_embed(cgm_context)                      # (B, Lc, pred_dim)
        blanks = self.mask_token.expand(b, self.num_gluco_patches, -1) + self.pos.unsqueeze(0)
        x = torch.cat([x, blanks], dim=1)                          # (B, Lc + num_gluco, pred_dim)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, lc:]                                              # keep glucodensity slice
        if gluco_masks is not None:
            idx = gluco_masks.unsqueeze(-1).expand(-1, -1, x.size(-1))
            x = x.gather(1, idx)                                   # (B, K, pred_dim)
        return self.predictor_proj(x)


def sigreg(z, num_slices: int = 256, k: int = 17):
    """SIGReg (LeJEPA, arXiv:2511.08544): push embeddings toward an isotropic
    unit Gaussian so they cannot collapse.

    Project z onto `num_slices` random unit directions; on each 1-D projection
    run an Epps-Pulley characteristic-function test against N(0,1) and average.
    Implemented with cos/sin (not complex tensors) so it runs on MPS. The
    test-statistic's *N scaling is dropped — as a loss term we want the
    batch-size-independent CF distance, not a p-value.

    z: (N, D) embeddings.  Returns a scalar >= 0 (0 == already isotropic N(0,1)).
    """
    n, d = z.shape
    a = torch.randn(d, num_slices, device=z.device, dtype=z.dtype)
    a = a / a.norm(dim=0, keepdim=True)
    proj = z @ a                                              # (N, num_slices)
    t = torch.linspace(-5, 5, k, device=z.device, dtype=z.dtype)   # (k,)
    phi = torch.exp(-0.5 * t ** 2)                            # N(0,1) CF, real  (k,)
    xt = proj.unsqueeze(-1) * t                               # (N, num_slices, k)
    re = torch.cos(xt).mean(0)                                # Re(empirical CF)  (num_slices, k)
    im = torch.sin(xt).mean(0)                                # Im(empirical CF)
    diff_sq = (re - phi) ** 2 + im ** 2                       # |ecf - phi|^2
    per_dir = torch.trapz(diff_sq * phi, t, dim=1)            # weighted integral per direction
    return per_dir.mean()


def x_forward_loss(
    cgm_encoder, cgm_encoder_ema, cgm_predictor,   # CGM half: JepaEncoder x2 (EMA) + JepaPredictor
    glu_encoder, glu_predictor,                    # glucodensity half: GlucoEncoder + GluPredictor
    glucose,        # (B, T)          raw CGM window
    gluco_img,      # (B, 32, 32, 3)  its glucodensity image
    cgm_ctx_idx,    # (C,)  visible CGM patch positions (context)
    cgm_tgt_idx,    # (Tc,) masked CGM patch positions to predict
    gluco_tgt_idx,  # (Tg,) masked glucodensity patch positions to predict
    gluco_loss_weight: float = 1.0,
    sigreg_weight: float = 0.0,
):
    """One X-CGM-JEPA step. Returns (total, cgm_loss, gluco_loss, reg).

    Mirrors pretrain_x_cgm_jepa.py: CGM targets come from the EMA encoder
    (stop-grad); glucodensity targets come from the TRAINABLE glu_encoder WITH
    gradients (no detach) — the cross-modal loss trains that encoder too. Both
    predictors share the same CGM context. Loss is L1.

    sigreg_weight > 0 adds a SIGReg penalty on the glucodensity embeddings only,
    which is the anti-collapse guard the released code lacks (0 == off, exactly
    the released behaviour).
    """
    b = glucose.size(0)

    # CGM target: EMA encoder, stop-grad, layernorm, keep masked patches.
    with torch.no_grad():
        cgm_full = cgm_encoder_ema(glucose)
        cgm_full = F.layer_norm(cgm_full, (cgm_full.size(-1),))
        cgm_targets = cgm_full[:, cgm_tgt_idx, :]

    # Shared CGM context: online encoder on the visible patches only.
    keep = cgm_ctx_idx.unsqueeze(0).expand(b, -1)
    cgm_context = cgm_encoder(glucose, keep=keep)

    cgm_pred = cgm_predictor(cgm_context, cgm_ctx_idx, cgm_tgt_idx)
    cgm_loss = F.l1_loss(cgm_pred, cgm_targets)

    # Glucodensity target: TRAINABLE encoder, gradients flow (NOT detached).
    glu_full = glu_encoder(gluco_img)
    # SIGReg acts on the raw embedding distribution, before per-token layernorm.
    reg = sigreg(glu_full.reshape(-1, glu_full.size(-1))) if sigreg_weight > 0 else glu_full.new_zeros(())
    glu_full = F.layer_norm(glu_full, (glu_full.size(-1),))
    glu_targets = glu_full[:, gluco_tgt_idx, :]

    # Same CGM context predicts the masked glucodensity patches (cross-modal).
    gluco_masks = gluco_tgt_idx.unsqueeze(0).expand(b, -1)
    glu_pred = glu_predictor(cgm_context, gluco_masks)
    gluco_loss = F.l1_loss(glu_pred, glu_targets)

    total = cgm_loss + gluco_loss_weight * gluco_loss + sigreg_weight * reg
    return total, cgm_loss.detach(), gluco_loss.detach(), reg.detach()


if __name__ == "__main__":
    import copy

    from sugar_jepa.jepa_pretrain import JepaPredictor
    from sugar_jepa.sugar_jepa_model import JepaEncoder

    b = 4

    enc = GlucoEncoder()
    img = torch.randn(b, 32, 32, 3)
    print("GlucoEncoder:", tuple(img.shape), "->", tuple(enc(img).shape))

    pred = GluPredictor()
    cgm_ctx = torch.randn(b, 16, 96)                    # CGM context (encoder output)
    gluco_masks = torch.arange(12, 16).expand(b, 4)     # 4 masked glucodensity patches per sample
    out = pred(cgm_ctx, gluco_masks)
    print("GluPredictor:", "-> predicts", tuple(out.shape), "(expected (4, 4, 96))")

    # Joined training block: CGM (128 -> 16 patches) + glucodensity (16 patches).
    cgm_encoder = JepaEncoder(n_time_steps=128, patch_size=8, embed_dim=96, n_layers=3, n_heads=6)
    cgm_encoder_ema = copy.deepcopy(cgm_encoder)
    for p in cgm_encoder_ema.parameters():
        p.requires_grad = False
    cgm_predictor = JepaPredictor(embed_dim=96, n_patches=16)

    glucose = torch.randn(b, 128)
    cgm_ctx_idx = torch.arange(0, 12)        # 12 visible CGM patches
    cgm_tgt_idx = torch.arange(12, 16)       # 4 masked CGM patches
    gluco_tgt_idx = torch.arange(8, 16)      # 8 masked glucodensity patches

    total, cgm_loss, gluco_loss, reg = x_forward_loss(
        cgm_encoder, cgm_encoder_ema, cgm_predictor, enc, pred,
        glucose, img, cgm_ctx_idx, cgm_tgt_idx, gluco_tgt_idx, sigreg_weight=1.0,
    )
    total.backward()
    print(f"x_forward_loss: total={total.item():.4f} cgm={cgm_loss.item():.4f} "
          f"gluco={gluco_loss.item():.4f} sigreg={reg.item():.4f}")
    print("glu_encoder gets grad:", enc.embed.weight.grad is not None)  # trainable teacher
