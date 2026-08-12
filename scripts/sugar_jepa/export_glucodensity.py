"""Precompute paired (glucose window, glucodensity image) arrays for the full CSV.

Exports one .npz per split (windows + images) so Colab can load them directly
instead of running KDE. Parallelises the KDE across CPU cores.
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from scripts.sugar_jepa.glucodensity import GRIDSIZE, MAX_GAP_MIN, WINDOW, glucodensity, good_windows
from scripts.sugar_one.train_sugar_one import impute_and_sort, load_splits_streaming

CSV = "data/input/loop_ai_ready_joined2.csv"
OUT = "data/output/x_jepa_paired"


def collect_windows(df):
    wins = []
    for _, grp in df.sort(["unique_id", "ds"]).group_by(["unique_id"], maintain_order=True):
        g = grp["glucose"].to_numpy().astype(np.float64)
        ds = grp["ds"].to_numpy()
        for w in good_windows(g, ds, WINDOW, MAX_GAP_MIN):
            wins.append(w.astype(np.float32))
    return wins


def _kde(w):
    try:
        return glucodensity(w, gridsize=GRIDSIZE).astype(np.float32)
    except Exception:
        return None  # degenerate window (singular KDE / spline failure)


def main():
    splits = load_splits_streaming(CSV, "sequence_id", False)
    for name, df in zip(["train", "val", "test"], splits):
        wins = collect_windows(impute_and_sort(df))
        print(f"{name}: {len(wins)} candidate windows -> computing KDE")
        with Pool() as pool:
            imgs = list(tqdm(pool.imap(_kde, wins, chunksize=16), total=len(wins), desc=name))
        keep = [(w, i) for w, i in zip(wins, imgs) if i is not None]
        if not keep:
            print(f"{name}: no valid windows")
            continue
        W = np.stack([w for w, _ in keep])
        I = np.stack([i for _, i in keep])
        path = f"{OUT}_{name}.npz"
        np.savez_compressed(path, windows=W, images=I)
        print(f"{name}: {len(W)} pairs ({len(wins) - len(W)} dropped) -> {path}")


if __name__ == "__main__":
    main()
