"""Map glucose windows to glucodensity images.

KDE math adapted from CGM-JEPA utils/glucodensity_utils.py (MIT).
"""
import numpy as np
import torch
from scipy import interpolate
from scipy.stats import gaussian_kde

from sugar_one.train_sugar_one import impute_and_sort, load_splits_streaming

CSV = "data/input/loop_ai_ready_joined2_dev.csv"
WINDOW = 128
GRIDSIZE = 32
MAX_GAP_MIN = 60.0  # drop a window if it has a hole bigger than this
OUT = "data/output/glucodensity"


def compute_2d_kde_grid(a, b, gridsize=64):
    kde = gaussian_kde(np.vstack([a, b]), bw_method="scott")
    a_min, a_max = np.percentile(a, [1, 99])
    b_min, b_max = np.percentile(b, [1, 99])
    Ag, Bg = np.meshgrid(np.linspace(a_min, a_max, gridsize), np.linspace(b_min, b_max, gridsize))
    Z = kde(np.vstack([Ag.ravel(), Bg.ravel()])).reshape(Ag.shape)
    return Z / (Z.max() + 1e-12)


def glucodensity(cgm_sequence, smoothing_factor=5.0, gridsize=64):
    if isinstance(cgm_sequence, torch.Tensor):
        cgm_sequence = cgm_sequence.cpu().numpy()
    cgm_sequence = np.asarray(cgm_sequence).flatten()
    t = np.arange(len(cgm_sequence)) * 5.0 / 60.0
    spline = interpolate.UnivariateSpline(t, cgm_sequence, s=smoothing_factor)
    g = spline(t)
    dg = spline.derivative(1)(t)
    ddg = spline.derivative(2)(t)
    return np.stack([
        compute_2d_kde_grid(g, dg, gridsize),
        compute_2d_kde_grid(g, ddg, gridsize),
        compute_2d_kde_grid(dg, ddg, gridsize),
    ], axis=-1)


def good_windows(glucose, ds, window, max_gap_min):
    step_min = np.diff(ds).astype("timedelta64[s]").astype(float) / 60.0
    for start in range(0, max(len(glucose) - window + 1, 0), window):
        w = glucose[start:start + window]
        gaps = step_min[start:start + window - 1]
        if np.isfinite(w).all() and (gaps <= max_gap_min).all():
            yield w


def save_previews(images, windows, out_dir, n=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    titles = ["glucose x speed", "glucose x accel", "speed x accel"]
    for i in range(min(n, len(images))):
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
        ax[0].plot(windows[i])
        ax[0].set_title("glucose")
        for c in range(3):
            ax[c + 1].imshow(images[i][:, :, c], origin="lower", aspect="auto")
            ax[c + 1].set_title(titles[c])
        fig.tight_layout()
        fig.savefig(f"{out_dir}/sample_{i}.png", dpi=110)
        plt.close(fig)


def build(df):
    images, windows = [], []
    for _, grp in df.sort(["unique_id", "ds"]).group_by(["unique_id"], maintain_order=True):
        g = grp["glucose"].to_numpy().astype(np.float64)
        ds = grp["ds"].to_numpy()
        for w in good_windows(g, ds, WINDOW, MAX_GAP_MIN):
            try:
                img = glucodensity(w, gridsize=GRIDSIZE).astype(np.float32)
            except np.linalg.LinAlgError:
                continue  # flat window -> singular KDE
            images.append(img)
            windows.append(w.astype(np.float32))
    return images, windows


def main():
    splits = load_splits_streaming(CSV, "sequence_id", False)
    for name, df in zip(["train", "val", "test"], splits):
        images, windows = build(impute_and_sort(df))
        if not images:
            print(f"{name}: no windows")
            continue
        arr = np.stack(images)
        np.save(f"{OUT}_{name}.npy", arr)
        if name == "train":
            save_previews(images, windows, f"{OUT}_preview")
        print(f"{name}: {len(images)} images {arr.shape} -> {OUT}_{name}.npy")


if __name__ == "__main__":
    main()
