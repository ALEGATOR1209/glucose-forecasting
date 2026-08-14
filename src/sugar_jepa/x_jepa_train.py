"""Train X-CGM-JEPA on the dev CSV — quick "is it improving?" check.

Plain loop in the style of jepa_pretrain.py (no framework). Builds paired
(glucose window, glucodensity image) data, then trains the four modules from
x_jepa_pretrain.py with L_total = L_CGM + w * L_Glu. EMA on the CGM encoder
only; the glucodensity encoder trains as a live teacher (see x_forward_loss).
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sugar_jepa.glucodensity import build, GRIDSIZE
from sugar_jepa.jepa_pretrain import (
    JepaPredictor,
    ema_update,
    momentum_at,
    sample_block_mask,
)
from sugar_jepa.sugar_jepa_model import JepaEncoder
from sugar_jepa.x_jepa_pretrain import GlucoEncoder, GluPredictor, x_forward_loss
from sugar_one.train_sugar_one import impute_and_sort, load_splits_streaming

CSV = "data/input/loop_ai_ready_joined2_dev.csv"
WINDOW = 128
PATCH = 8
N_CGM_PATCHES = WINDOW // PATCH          # 16
N_GLU_PATCHES = (GRIDSIZE // 8) ** 2     # 16
CACHE = Path("data/output/x_jepa_paired_train.npz")


class PairedDataset(Dataset):
    """(glucose[128], glucodensity[32,32,3]) pairs for the same window."""

    def __init__(self, windows: np.ndarray, images: np.ndarray):
        self.windows = torch.from_numpy(windows).float()
        self.images = torch.from_numpy(images).float()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        return self.windows[i], self.images[i]


def load_paired():
    if CACHE.exists():
        d = np.load(CACHE)
        return d["windows"], d["images"]
    train_df, _, _ = load_splits_streaming(CSV, "sequence_id", False)
    images, windows = build(impute_and_sort(train_df))
    images, windows = np.stack(images), np.stack(windows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, windows=windows, images=images)
    return windows, images


def main(epochs=20, batch_size=256, embed_dim=96, lr=1e-3, gluco_loss_weight=1.0,
         sigreg_weight=1.0, n_cgm_targets=4, n_glu_targets=8, ema_base=0.999, seed=0):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    windows, images = load_paired()
    print(f"paired windows: {len(windows)}  glucose{windows.shape[1:]}  gluco{images.shape[1:]}  | {device}")
    loader = DataLoader(PairedDataset(windows, images), batch_size=batch_size,
                        shuffle=True, drop_last=True)

    cgm_encoder = JepaEncoder(n_time_steps=WINDOW, patch_size=PATCH,
                              embed_dim=embed_dim, n_layers=3, n_heads=6).to(device)
    cgm_encoder_ema = copy.deepcopy(cgm_encoder)
    for p in cgm_encoder_ema.parameters():
        p.requires_grad = False
    cgm_predictor = JepaPredictor(embed_dim=embed_dim, n_patches=N_CGM_PATCHES).to(device)
    glu_encoder = GlucoEncoder(gridsize=GRIDSIZE, patch=8, embed_dim=embed_dim).to(device)
    glu_predictor = GluPredictor(num_gluco_patches=N_GLU_PATCHES, embed_dim=embed_dim).to(device)

    params = (list(cgm_encoder.parameters()) + list(cgm_predictor.parameters())
              + list(glu_encoder.parameters()) + list(glu_predictor.parameters()))
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.04)
    total_steps = epochs * len(loader)

    step = 0
    for epoch in range(epochs):
        cgm_encoder.train(); cgm_predictor.train(); glu_encoder.train(); glu_predictor.train()
        s_total, s_cgm, s_glu, s_reg, n = 0.0, 0.0, 0.0, 0.0, 0
        for glucose, gluco_img in loader:
            glucose, gluco_img = glucose.to(device), gluco_img.to(device)
            ctx, tgt = sample_block_mask(N_CGM_PATCHES, n_cgm_targets, 2, 4, rng)
            glu_tgt = sorted(rng.sample(range(N_GLU_PATCHES), n_glu_targets))

            total, cgm_loss, gluco_loss, reg = x_forward_loss(
                cgm_encoder, cgm_encoder_ema, cgm_predictor, glu_encoder, glu_predictor,
                glucose, gluco_img,
                torch.tensor(ctx, device=device), torch.tensor(tgt, device=device),
                torch.tensor(glu_tgt, device=device),
                gluco_loss_weight=gluco_loss_weight, sigreg_weight=sigreg_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            ema_update(cgm_encoder_ema, cgm_encoder, momentum_at(step, total_steps, ema_base))

            s_total += total.item(); s_cgm += cgm_loss.item()
            s_glu += gluco_loss.item(); s_reg += reg.item()
            n += 1; step += 1

        # Collapse watch: std of the (trainable) glucodensity teacher's latents.
        with torch.no_grad():
            glu_encoder.eval()
            sample = torch.from_numpy(images[:256]).float().to(device)
            glu_std = glu_encoder(sample).reshape(-1, embed_dim).std(0).mean().item()
        print(f"epoch {epoch:2d} | total={s_total/n:.4f} cgm={s_cgm/n:.4f} "
              f"gluco={s_glu/n:.4f} sigreg={s_reg/n:.4f} | glu_latent_std={glu_std:.4f}")


if __name__ == "__main__":
    main()
