#!/usr/bin/env python3
"""Shared checkpoint save/load/metadata utilities.

Generalizes the near-identical ``save_full_checkpoint`` / ``load_full_checkpoint``
/ ``read_checkpoint_meta`` / ``update_latest_symlink`` helpers that used to live
inline in ``train_glumind.py`` and ``train_sugar_one.py``. These operate on
generic ``nn.Module`` / optimizer / scheduler objects already, so no
model-specific logic is involved. Checkpoint dict shape (key names) is
preserved exactly per caller to keep existing checkpoints loadable.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def save_full_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_val_loss: float,
    config_dict: dict,
    *,
    config_key: str = "args",
    stringify_paths: bool = False,
    wait: int | None = None,
    best_epoch: int | None = None,
    atomic: bool = False,
) -> None:
    """Save a full checkpoint: model + optimizer + scheduler + metadata.

    ``config_key``: key name under which ``config_dict`` is stored
        ("args" for GluMind-style callers, "config" for SugarOne-style).
    ``stringify_paths``: if True, convert ``Path`` values in ``config_dict``
        to strings before saving (GluMind passes an argparse Namespace's
        ``vars()``; SugarOne already passes a plain, pre-stringified dict).
    ``wait`` / ``best_epoch``: optional extra fields (SugarOne-style
        checkpoints track early-stopping state for resume).
    ``atomic``: if True, write to a ``.tmp`` file and atomically replace
        (SugarOne-style); if False, write directly (GluMind-style).
    """
    cfg = config_dict
    if stringify_paths:
        cfg = {k: str(v) if isinstance(v, Path) else v for k, v in config_dict.items()}

    ckpt = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        config_key: cfg,
    }
    if wait is not None:
        ckpt["wait"] = wait
    if best_epoch is not None:
        ckpt["best_epoch"] = best_epoch

    if atomic:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)
    else:
        torch.save(ckpt, path)


def load_full_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
    *,
    return_wait_and_best_epoch: bool = False,
    log_fn=print,
):
    """Load a full checkpoint.

    By default returns ``(epoch, best_val_loss)`` (GluMind-style).
    If ``return_wait_and_best_epoch`` is True, returns
    ``(epoch, best_val_loss, wait, best_epoch)`` (SugarOne-style), where
    ``epoch`` is the last *completed* epoch.
    """
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    epoch = int(ckpt.get("epoch", 0))
    best_val = float(ckpt.get("best_val_loss", float("inf")))

    if return_wait_and_best_epoch:
        wait = int(ckpt.get("wait", 0))
        best_epoch = int(ckpt.get("best_epoch", epoch))
        log_fn(
            f"  Loaded checkpoint: last_completed_epoch={epoch} | "
            f"next_epoch={epoch + 1} | best_epoch={best_epoch} | "
            f"best_val_loss={best_val:.6f} | patience_wait={wait}"
        )
        return epoch, best_val, wait, best_epoch

    log_fn(f"  Resumed from checkpoint: epoch={epoch}, "
           f"best_val_loss={best_val:.6f}")
    return epoch, best_val


def read_checkpoint_meta(path: Path) -> dict[str, int | float] | None:
    """Lightweight read of a checkpoint file for tuning state (no model load)."""
    if not path.is_file():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "best_epoch": int(ckpt.get("best_epoch", 0)),
        "best_val_loss": float(ckpt.get("best_val_loss", float("inf"))),
        "wait": int(ckpt.get("wait", 0)),
    }


def update_latest_symlink(run_dir: Path, out_dir: Path, log_fn=print) -> None:
    """Write a 'latest.txt' pointer to the most recent run directory.

    Using a plain text file instead of a symlink avoids the Windows privilege
    requirement (WinError 1314) that blocks symlink creation for non-admin
    users without Developer Mode enabled.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_txt = out_dir / "latest.txt"
    latest_txt.write_text(str(run_dir) + "\n", encoding="utf-8")
    log_fn(f"Latest run pointer: {latest_txt} -> {run_dir}")


def strip_compile_prefix(state_dict: dict) -> dict:
    """Strip the ``_orig_mod.`` prefix added by ``torch.compile`` (if present).

    Duplicated inline in evaluate_glumind.py and evaluate_model.py prior to
    this refactor; both call sites do: ``state = torch.load(...); if any(k
    startswith _orig_mod.) then strip``.
    """
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict
