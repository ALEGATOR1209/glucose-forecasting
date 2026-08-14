"""Per-epoch visual diagnostics for the JEPA encoder.

One PNG per epoch, four panels, answering the only question that matters early:
is the encoder learning structure, or quietly collapsing?

  1. Activation distribution — the spread of latent values. Narrowing toward a
     spike at one value is collapse.
  2. Per-dimension std across windows — a healthy encoder uses its dimensions.
     Bars flattening to zero mean those dimensions carry no information.
  3. PCA of the pooled embeddings, coloured by the window's mean glucose. If the
     colour gradient follows the scatter, the embedding has at least recovered
     glucose level. A single blob with no structure means it has not.
  4. Cumulative explained variance. A curve that hits 100% in two or three
     components is the visual form of a low effective rank.

matplotlib is imported lazily so that a missing plotting dependency degrades to
"no plots" rather than breaking training.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _to_2d(latents: np.ndarray) -> np.ndarray:
    """(N, n_patches, embed_dim) -> (N, embed_dim) by mean-pooling patches."""
    if latents.ndim == 3:
        return latents.mean(axis=1)
    if latents.ndim != 2:
        raise ValueError(f"expected (N, P, E) or (N, E) latents, got {latents.shape}")
    return latents


def window_trend(windows: np.ndarray) -> np.ndarray:
    """Normalized trend of each glucose window: (last - first) / std.

    A scale-invariant colour for the PCA panel, and the choice matters: the
    encoder instance-normalizes every window (per-window z-score), which removes
    both the absolute level and the amplitude. Colouring by mean glucose would
    therefore be null *by construction* — it would look identical for a perfect
    encoder and a collapsed one. Shape is what survives the normalization, so
    shape is what we colour by.
    """
    w = np.asarray(windows, dtype=np.float64)
    return (w[:, -1] - w[:, 0]) / np.maximum(w.std(axis=1), 1e-6)


def plot_encoder_diagnostics(
    latents: np.ndarray,
    out_path: Path,
    epoch: int,
    color_values: np.ndarray | None = None,
    color_label: str = "window trend  (last − first) / std",
    subtitle: str = "",
) -> Path | None:
    """Write the 4-panel diagnostic figure. Returns the path, or None if plotting
    is unavailable (never raises into the training loop)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: no display on a training box
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        return None

    arr = np.asarray(latents, dtype=np.float64)
    pooled = _to_2d(arr)                       # (N, E) — one point per window, for PCA
    tokens = arr.reshape(-1, arr.shape[-1])    # (N*P, E) — per-patch, for the std metric
    n_samples, embed_dim = pooled.shape
    if n_samples < 3:
        return None

    # Same quantity `collapse_metrics` logs as latent_std, so the figure and the
    # CSV cannot disagree. (Pooling patches first would average the variance away
    # and show a much smaller number under the same name.)
    per_dim_std = tokens.std(axis=0)
    n_components = min(embed_dim, n_samples, 20)
    pca = PCA(n_components=n_components).fit(pooled)
    projected = pca.transform(pooled)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"JEPA encoder — epoch {epoch}" + (f"\n{subtitle}" if subtitle else ""),
        fontsize=13,
    )

    # 1. activation distribution
    ax = axes[0, 0]
    ax.hist(tokens.ravel(), bins=80, color="#4C72B0")
    ax.set_title("Latent activation distribution (all patches)")
    ax.set_xlabel("activation")
    ax.set_ylabel("count")

    # 2. per-dimension std — collapse shows up as bars going to zero
    ax = axes[0, 1]
    ax.bar(np.arange(embed_dim), np.sort(per_dim_std)[::-1], color="#DD8452", width=1.0)
    ax.axhline(per_dim_std.mean(), color="k", ls="--", lw=1,
               label=f"mean = {per_dim_std.mean():.3f}  (= latent_std)")
    ax.set_title("Per-dimension std across patches (sorted)")
    ax.set_xlabel("dimension (sorted)")
    ax.set_ylabel("std")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    # 3. PCA scatter, coloured by glucose level
    ax = axes[1, 0]
    var = pca.explained_variance_ratio_
    if color_values is not None and len(color_values) == n_samples:
        sc = ax.scatter(
            projected[:, 0], projected[:, 1], c=np.asarray(color_values),
            cmap="viridis", s=10, alpha=0.75,
        )
        fig.colorbar(sc, ax=ax, label=color_label)
    else:
        ax.scatter(projected[:, 0], projected[:, 1], s=10, alpha=0.75, color="#55A868")
    ax.set_title("PCA of pooled embeddings")
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}% var)" if len(var) > 1 else "PC2")

    # 4. cumulative explained variance — the visual form of effective rank
    ax = axes[1, 1]
    cumulative = np.cumsum(var)
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", ms=3, color="#C44E52")
    ax.axhline(0.95, color="k", ls="--", lw=1, label="95%")
    ax.set_title("Cumulative explained variance")
    ax.set_xlabel("component")
    ax.set_ylabel("cumulative ratio")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path