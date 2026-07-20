#!/usr/bin/env python3
"""
Random-search tuner for SugarOne (global training only).

One code path; behaviour comes entirely from the TOML config file.
Shipped configs: tune_sugar_one_full.toml (default, production) and
tune_sugar_one_dev.toml (laptop search; pass -c explicitly).
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
import tomllib
import torch
import typer

from glucose_forecasting.training import sugar_one as tg
from scripts.sugar_one.console_log import echo_plain

app = typer.Typer(
    name="tune-sugar-one",
    add_completion=False,
    help="Random hyperparameter search for SugarOne (global mode).",
)

STATE_VERSION = 1

STATUS_RUNNING = "running"
STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"

NON_RETRYABLE_ERROR_MARKERS = (
    "OutOfMemoryError",
    "CUDA out of memory",
    "cuda out of memory",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "TritonMissing",
    "skipped_by_user",
)

# Default config when --config is omitted: file next to this script, else cwd fallback.
DEFAULT_CONFIG_FILENAME = "tune_sugar_one_full.toml"
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _SCRIPT_DIR / DEFAULT_CONFIG_FILENAME


def resolve_config_path(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise typer.BadParameter(f"Config file not found: {path}")
        return path
    if DEFAULT_CONFIG_PATH.is_file():
        return DEFAULT_CONFIG_PATH
    cwd_fallback = Path.cwd() / "scripts" / "sugar_one" / DEFAULT_CONFIG_FILENAME
    if cwd_fallback.is_file():
        return cwd_fallback.resolve()
    raise typer.BadParameter(
        f"Config not found. Pass --config PATH or place {DEFAULT_CONFIG_FILENAME} in "
        f"{_SCRIPT_DIR} (or scripts/sugar_one/ under the repo root)."
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object in {path}")
    return data


def canonical_params_for_hash(params: Mapping[str, Any]) -> dict[str, Any]:
    """Stable dict for hashing (sorted keys, rounded floats)."""
    out: dict[str, Any] = {}
    for key in sorted(params):
        val = params[key]
        if isinstance(val, float):
            out[key] = round(val, 12)
        elif isinstance(val, bool):
            out[key] = val
        elif isinstance(val, int):
            out[key] = val
        elif isinstance(val, str):
            out[key] = val
        else:
            out[key] = val
    return out


def combo_hash(params: Mapping[str, Any]) -> str:
    blob = json.dumps(canonical_params_for_hash(params), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_user_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    cfg = tomllib.loads(raw.decode("utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid TOML root in {path}")
    return cfg


def resolve_csv_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def prepare_frames(
    csv_path: Path,
    *,
    unique_id: str,
    drop_interpolated: bool,
    study_groups: str,
    split_scheme: str,
    max_train_series: int,
    max_eval_series: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train_df, val_df, test_df = tg.load_splits_streaming(
        csv_path, unique_id, drop_interpolated
    )
    train_df = tg.normalize_study_groups_column(train_df)
    val_df = tg.normalize_study_groups_column(val_df)
    test_df = tg.normalize_study_groups_column(test_df)

    if study_groups.strip():
        group_list = [
            tg.normalize_study_group_label(g.strip())
            for g in study_groups.split(",")
            if g.strip()
        ]
        train_df = train_df.filter(pl.col("study_group").is_in(group_list))
        val_df = val_df.filter(pl.col("study_group").is_in(group_list))
        test_df = test_df.filter(pl.col("study_group").is_in(group_list))

    train_df, val_df, test_df = tg.apply_split_scheme(train_df, val_df, test_df, split_scheme)

    if max_train_series > 0:
        train_df = tg.limit_series(train_df, max_train_series)
    if max_eval_series > 0:
        val_df = tg.limit_series(val_df, max_eval_series)
        test_df = tg.limit_series(test_df, max_eval_series)

    train_df = tg.impute_and_sort(train_df)
    val_df = tg.impute_and_sort(val_df)
    test_df = tg.impute_and_sort(test_df)
    return train_df, val_df, test_df


def make_device(device_name: str) -> torch.device:
    if device_name == "mps" and not torch.backends.mps.is_available():
        echo_plain("MPS not available, falling back to CPU.")
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        echo_plain("CUDA not available, falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    return device


def setup_cuda_flags(disable_tf32: bool, device: torch.device) -> None:
    if device.type != "cuda":
        return
    if not disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        echo_plain("TF32 enabled.")


def read_split_metrics(run_dir: Path, split: str) -> dict[str, float] | None:
    path = run_dir / f"{split}_metrics_overall.csv"
    if not path.exists():
        return None
    df = pl.read_csv(path)
    if df.is_empty():
        return None
    row = df.row(0, named=True)
    return {
        "mae": float(row["mae"]),
        "rmse": float(row["rmse"]),
        "mard": float(row["mard"]),
    }


def sample_from_space(rng: random.Random, space: dict[str, list[Any]]) -> dict[str, Any]:
    return {key: rng.choice(values) for key, values in space.items()}


def derive_rng(master_seed: int, draw_index: int) -> random.Random:
    """Deterministic RNG stream slot — safe for resume via persisted next_draw_index."""
    digest = hashlib.sha256(f"{master_seed}:{draw_index}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def merge_defaults_and_sample(
    defaults: Mapping[str, Any],
    sampled: Mapping[str, Any],
) -> dict[str, Any]:
    """Config [defaults], then random draw from [tune.space] (sample wins on overlap)."""
    return {**dict(defaults), **dict(sampled)}


def format_trial_params(params: Mapping[str, Any]) -> str:
    parts = [f"{k}={params[k]}" for k in sorted(params.keys())]
    return " | ".join(parts)


def runtime_from_tune_section(tune: Mapping[str, Any]) -> dict[str, Any]:
    """Training loop scalars from [tune] (everything except nested [tune.space])."""
    return {
        "epochs": int(tune["epochs"]),
        "patience": int(tune["patience"]),
        "log_every": int(tune["log_every"]),
        "val_every_n_epochs": int(tune["val_every_n_epochs"]),
        "ckpt_every_n_epochs": int(tune["ckpt_every_n_epochs"]),
        "precision": str(tune["precision"]),
        "compile_mode": str(tune["compile_mode"]),
        "disable_tf32": bool(tune["disable_tf32"]),
        "num_workers": int(tune["num_workers"]),
        "prefetch_factor": int(tune["prefetch_factor"]),
        "batch_log_every": int(tune.get("batch_log_every", 0)),
        "eval_batch_log_every": int(tune.get("eval_batch_log_every", 0)),
    }


def build_train_cfg(
    *,
    csv_resolved: Path,
    dataset_cfg: dict[str, Any],
    params: dict[str, Any],
    runtime: dict[str, Any],
    device_name: str,
    seed: int,
    out_dir: Path,
    resume_from: str,
) -> dict[str, Any]:
    p = params
    r = runtime
    return {
        "csv": str(csv_resolved),
        "unique_id": dataset_cfg["unique_id"],
        "drop_interpolated": dataset_cfg["drop_interpolated"],
        "study_groups": dataset_cfg["study_groups"],
        "split_scheme": dataset_cfg["split_scheme"],
        "mode": "global",
        "horizon": int(p["horizon"]),
        "input_steps": int(p["input_steps"]),
        "d_model": int(p["d_model"]),
        "n_heads": int(p["n_heads"]),
        "n_blocks": int(p["n_blocks"]),
        "ff_units": int(p["ff_units"]),
        "dropout": float(p["dropout"]),
        "epochs": int(r["epochs"]),
        "batch_size": int(p["batch_size"]),
        "precision": str(r["precision"]),
        "compile_mode": str(r["compile_mode"]),
        "disable_tf32": bool(r["disable_tf32"]),
        "num_workers": int(r["num_workers"]),
        "prefetch_factor": int(r["prefetch_factor"]),
        "lr": float(p["lr"]),
        "weight_decay": float(p["weight_decay"]),
        "patience": int(r["patience"]),
        "log_every": int(r["log_every"]),
        "ckpt_every_n_epochs": int(r["ckpt_every_n_epochs"]),
        "val_every_n_epochs": int(r["val_every_n_epochs"]),
        "batch_log_every": int(r.get("batch_log_every", 0)),
        "eval_batch_log_every": int(r.get("eval_batch_log_every", 0)),
        "resume_from": resume_from,
        "lwf_lambda": 0.5,
        "continual_order": "default",
        "continual_val_scope": "current_group",
        "device": device_name,
        "seed": seed,
        "out_dir": str(out_dir),
    }


def run_one_global_trial(
    *,
    cfg: dict[str, Any],
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    device: torch.device,
    run_name: str,
    out_root: Path,
) -> Path:
    t_ds = time.perf_counter()
    train_ds, val_ds, test_ds = tg.build_datasets(
        train_df, val_df, test_df, cfg["input_steps"], cfg["horizon"]
    )
    echo_plain(
        f"  Dataset windows built in {time.perf_counter() - t_ds:.1f}s | "
        f"train={len(train_ds):,} val={len(val_ds) if val_ds else 0:,} "
        f"test={len(test_ds) if test_ds else 0:,}"
    )
    model = tg.make_model(**tg._model_kwargs(cfg), device=device)
    tg.run_train_and_eval(
        model,
        train_ds,
        val_ds,
        test_ds,
        cfg,
        device,
        run_name,
        out_root,
    )
    return out_root / run_name


def trial_params_from_record(trial: Mapping[str, Any]) -> dict[str, Any]:
    """Merged hyperparameters (legacy state used separate capacity_fixed)."""
    params = dict(trial.get("params", {}))
    params.update(trial.get("capacity_fixed", {}))
    return params


def leaderboard_row_from_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    params = trial_params_from_record(trial)
    row: dict[str, Any] = {
        "trial_index": int(trial["trial_index"]),
        "combo_hash": str(trial["combo_hash"]),
        "val_mae": float(trial["val_mae"]),
        "val_rmse": float(trial["val_rmse"]),
        "val_mard": float(trial["val_mard"]),
        "seconds": float(trial["duration_seconds"]),
    }
    row.update(params)
    return row


def write_leaderboard_csv(path: Path, trials_ok: list[Mapping[str, Any]]) -> None:
    """Rewrite leaderboard from all successful trials (sorted by val_mae)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not trials_ok:
        if path.exists():
            path.unlink()
        return
    rows = [leaderboard_row_from_trial(t) for t in trials_ok]
    pl.DataFrame(rows).sort("val_mae").write_csv(path)


def marginal_counts(top_trials: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key in keys:
        counts: dict[str, int] = {}
        for t in top_trials:
            raw = trial_params_from_record(t).get(key)
            label = str(raw)
            counts[label] = counts.get(label, 0) + 1
        out[key] = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return out


def write_tune_report(
    *,
    report_path: Path,
    config_path: Path,
    trials_ok: list[dict[str, Any]],
    timing_stats: dict[str, Any],
    report_param_keys: list[str] | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# SugarOne tuning report")
    lines.append("")
    lines.append(f"- Config: `{config_path}`")
    lines.append(f"- Generated: `{datetime.now().isoformat()}`")
    lines.append(f"- Successful trials: **{len(trials_ok)}**")
    lines.append(
        f"- Total wall time (success): **{timing_stats.get('total_seconds_success', 0.0):.1f}s** "
        f"| mean **{timing_stats.get('mean_seconds_per_trial', 0.0):.1f}s** / trial"
    )
    lines.append("")
    lines.append("## Leaderboard (sorted by val MAE)")
    lines.append("")
    if trials_ok:
        if report_param_keys is None:
            report_param_keys = sorted(
                {k for t in trials_ok for k in t.get("params", {}).keys()}
            )
        rows = []
        for t in sorted(trials_ok, key=lambda x: float(x["val_mae"])):
            row: dict[str, Any] = {
                "trial": t["trial_index"],
                "val_mae": float(t["val_mae"]),
                "val_rmse": float(t["val_rmse"]),
                "val_mard": float(t["val_mard"]),
                "seconds": round(float(t["duration_seconds"]), 2),
            }
            for key in report_param_keys:
                row[key] = trial_params_from_record(t).get(key)
            rows.append(row)
        buf = io.StringIO()
        pl.DataFrame(rows).write_csv(buf)
        lines.append("```csv")
        lines.append(buf.getvalue().rstrip("\n"))
        lines.append("```")
    else:
        lines.append("_No successful trials yet._")
    lines.append("")
    lines.append("## Parameter prioritization (top quartile of successful trials)")
    lines.append("")
    sorted_ok = sorted(trials_ok, key=lambda x: float(x["val_mae"]))
    q = max(1, len(sorted_ok) // 4)
    top_q = sorted_ok[:q]
    keys = report_param_keys if report_param_keys else []
    if top_q and keys:
        marg = marginal_counts(top_q, keys)
        lines.append(
            "Counts show how often each discrete value appeared among the "
            f"best **{len(top_q)}** runs."
        )
        lines.append("")
        for key in keys:
            lines.append(f"### `{key}`")
            lines.append("")
            for val, cnt in marg[key].items():
                lines.append(f"- `{val}` -> **{cnt}**")
            lines.append("")
    else:
        lines.append("_No data._")
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def timing_from_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [t for t in trials if t.get("status") == STATUS_OK]
    total = sum(float(t["duration_seconds"]) for t in ok)
    n = len(ok)
    mean = total / n if n else 0.0
    return {
        "total_seconds_success": total,
        "mean_seconds_per_trial": mean,
        "count_success": n,
    }


def is_non_retryable_failure(error: str | None) -> bool:
    if not error:
        return False
    return any(marker in error for marker in NON_RETRYABLE_ERROR_MARKERS)


def resolve_trial_checkpoint(run_dir: Path) -> str:
    last_ckpt = run_dir / "last_checkpoint.pt"
    if last_ckpt.is_file():
        return str(last_ckpt.resolve())
    return ""


def checkpoint_resume_summary(run_dir: Path) -> str:
    meta = tg.read_checkpoint_meta(run_dir / "last_checkpoint.pt")
    if meta is None:
        return ""
    last_ep = int(meta["epoch"])
    return (
        f"last_completed_epoch={last_ep} | resume_from_epoch={last_ep + 1} | "
        f"best_epoch={int(meta['best_epoch'])} | patience_wait={int(meta['wait'])}"
    )


def _hash_blocked_for_new_draw(trials: list[dict[str, Any]], ch: str) -> bool:
    for trial in trials:
        if str(trial["combo_hash"]) != ch:
            continue
        status = trial.get("status")
        if status == STATUS_OK:
            return True
        if status == STATUS_RUNNING:
            return True
        if status == STATUS_FAILED and trial.get("non_retryable"):
            return True
    return False


def _success_count(trials: list[dict[str, Any]]) -> int:
    return sum(1 for t in trials if t.get("status") == STATUS_OK)


@dataclass
class TuneContext:
    user_cfg: dict[str, Any]
    config_path: Path
    device_name: str
    csv_path: Path
    out_root: Path
    state_path: Path
    leaderboard_path: Path
    report_path: Path
    space: dict[str, list[Any]]
    param_defaults: dict[str, Any]
    runtime: dict[str, Any]
    dataset: dict[str, Any]
    report_param_keys: list[str]
    n_trials_target: int
    max_draws: int
    rng_seed: int
    resume_from: str


@dataclass
class ClaimedTrial:
    trial_index: int
    combo_hash: str
    trial_params: dict[str, Any]
    run_name: str
    resume_from_path: str = ""
    is_resume: bool = False


def build_tune_context(
    *,
    user_cfg: dict[str, Any],
    config_path: Path,
    device_name: str,
    seed_override: int | None,
) -> TuneContext:
    paths = user_cfg["paths"]
    dataset = user_cfg["dataset"]
    tune = user_cfg["tune"]
    space_raw = tune.get("space", {})
    param_defaults: dict[str, Any] = dict(user_cfg.get("defaults", {}))
    runtime = runtime_from_tune_section(tune)

    csv_path = resolve_csv_path(str(paths["csv"]))
    out_root = resolve_csv_path(str(paths["output_dir"]))
    out_root.mkdir(parents=True, exist_ok=True)

    space: dict[str, list[Any]] = {k: list(v) for k, v in space_raw.items()}
    if not space and not param_defaults:
        raise typer.BadParameter(
            "Config must define [defaults] when [tune.space] is omitted."
        )

    n_trials_target = int(tune["n_trials"])
    if not space and n_trials_target > 1:
        n_trials_target = 1

    return TuneContext(
        user_cfg=user_cfg,
        config_path=config_path,
        device_name=device_name,
        csv_path=csv_path,
        out_root=out_root,
        state_path=out_root / "state.json",
        leaderboard_path=out_root / "leaderboard.csv",
        report_path=out_root / "tune_report.md",
        space=space,
        param_defaults=param_defaults,
        runtime=runtime,
        dataset=dict(dataset),
        report_param_keys=sorted({*space.keys(), *param_defaults.keys()}),
        n_trials_target=n_trials_target,
        max_draws=int(tune["max_random_draws"]),
        rng_seed=int(seed_override if seed_override is not None else tune["random_seed"]),
        resume_from=str(tune.get("resume_from", "")),
    )


def ensure_state_initialized(ctx: TuneContext) -> None:
    if ctx.state_path.exists():
        return
    state = {
        "version": STATE_VERSION,
        "config_path": str(ctx.config_path.resolve()),
        "random_seed": ctx.rng_seed,
        "target_trials": ctx.n_trials_target,
        "trials": [],
        "next_draw_index": 0,
    }
    atomic_write_json(ctx.state_path, state)


def reconcile_trial_state(ctx: TuneContext) -> int:
    ensure_state_initialized(ctx)
    state = load_json_if_exists(ctx.state_path)
    if state is None:
        return 0
    changed = 0
    trials: list[dict[str, Any]] = state["trials"]
    for trial in trials:
        status = trial.get("status")
        run_dir = Path(str(trial.get("run_dir", "")))
        ckpt_path = resolve_trial_checkpoint(run_dir) if run_dir else ""
        meta = tg.read_checkpoint_meta(Path(ckpt_path)) if ckpt_path else None

        if status == STATUS_RUNNING:
            trial["status"] = STATUS_INTERRUPTED
            trial["resume_checkpoint"] = ckpt_path
            if meta is not None:
                trial["last_completed_epoch"] = int(meta["epoch"])
                trial["resume_from_epoch"] = int(meta["epoch"]) + 1
            changed += 1
            continue

        if status == STATUS_FAILED:
            non_retryable = is_non_retryable_failure(trial.get("error"))
            if trial.get("non_retryable") is not True and non_retryable:
                trial["non_retryable"] = True
                changed += 1
            continue

        if status == STATUS_INTERRUPTED:
            if trial.get("resume_checkpoint") != ckpt_path:
                trial["resume_checkpoint"] = ckpt_path
                changed += 1
            if meta is not None:
                last_ep = int(meta["epoch"])
                if trial.get("last_completed_epoch") != last_ep:
                    trial["last_completed_epoch"] = last_ep
                    trial["resume_from_epoch"] = last_ep + 1
                    changed += 1

    if changed:
        state["trials"] = trials
        atomic_write_json(ctx.state_path, state)
    return changed


def refresh_artifacts(ctx: TuneContext) -> None:
    state = load_json_if_exists(ctx.state_path)
    if state is None:
        return
    trials = state["trials"]
    successes = [t for t in trials if t.get("status") == STATUS_OK]
    stats = timing_from_trials(trials)
    write_leaderboard_csv(ctx.leaderboard_path, successes)
    write_tune_report(
        report_path=ctx.report_path,
        config_path=ctx.config_path,
        trials_ok=successes,
        timing_stats=stats,
        report_param_keys=ctx.report_param_keys,
    )


def _resumable_sort_key(trial: dict[str, Any]) -> tuple[int, int]:
    priority = 0 if trial.get("status") == STATUS_INTERRUPTED else 1
    return (priority, int(trial["trial_index"]))


def try_claim_resumable_trial(ctx: TuneContext) -> ClaimedTrial | None:
    state = load_json_if_exists(ctx.state_path)
    if state is None:
        return None
    trials: list[dict[str, Any]] = state["trials"]
    if _success_count(trials) >= ctx.n_trials_target:
        return None

    candidates = [
        t
        for t in trials
        if t.get("status") == STATUS_INTERRUPTED
        or (t.get("status") == STATUS_FAILED and not t.get("non_retryable"))
    ]
    candidates.sort(key=_resumable_sort_key)

    for trial in candidates:
        trial_index = int(trial["trial_index"])
        idx = next(i for i, t in enumerate(trials) if int(t["trial_index"]) == trial_index)
        run_dir = Path(str(trial["run_dir"]))
        ckpt = resolve_trial_checkpoint(run_dir) or str(trial.get("resume_checkpoint", ""))
        meta = tg.read_checkpoint_meta(Path(ckpt)) if ckpt else None
        trials[idx] = {
            **trial,
            "status": STATUS_RUNNING,
            "claimed_at": time.time(),
            "resume_checkpoint": ckpt,
            "error": None,
        }
        if meta is not None:
            trials[idx]["last_completed_epoch"] = int(meta["epoch"])
            trials[idx]["resume_from_epoch"] = int(meta["epoch"]) + 1
        state["trials"] = trials
        atomic_write_json(ctx.state_path, state)
        return ClaimedTrial(
            trial_index=trial_index,
            combo_hash=str(trial["combo_hash"]),
            trial_params=trial_params_from_record(trial),
            run_name=str(trial["run_name"]),
            resume_from_path=ckpt,
            is_resume=True,
        )
    return None


def claim_next_trial(ctx: TuneContext) -> ClaimedTrial | None:
    ensure_state_initialized(ctx)
    reconcile_trial_state(ctx)
    resumed = try_claim_resumable_trial(ctx)
    if resumed is not None:
        return resumed

    state = load_json_if_exists(ctx.state_path)
    if state is None:
        return None
    trials: list[dict[str, Any]] = state["trials"]
    if _success_count(trials) >= ctx.n_trials_target:
        return None

    draw_idx = int(state.get("next_draw_index", 0))
    attempts = 0
    next_index = max((int(t["trial_index"]) for t in trials), default=-1) + 1

    while attempts < ctx.max_draws:
        trial_rng = derive_rng(ctx.rng_seed, draw_idx)
        draw_idx += 1
        attempts += 1
        sampled = (
            sample_from_space(trial_rng, ctx.space)
            if ctx.space
            else {}
        )
        trial_params = merge_defaults_and_sample(ctx.param_defaults, sampled)
        ch = combo_hash(trial_params)
        if _hash_blocked_for_new_draw(trials, ch):
            continue

        run_name = f"trial_{next_index:04d}_{ch[:8]}"
        record: dict[str, Any] = {
            "trial_index": next_index,
            "combo_hash": ch,
            "params": trial_params,
            "run_name": run_name,
            "status": STATUS_RUNNING,
            "error": None,
            "non_retryable": False,
            "duration_seconds": 0.0,
            "val_mae": None,
            "val_rmse": None,
            "val_mard": None,
            "run_dir": str(ctx.out_root / run_name),
            "resume_checkpoint": "",
            "claimed_at": time.time(),
        }
        trials.append(record)
        state["trials"] = trials
        state["next_draw_index"] = draw_idx
        atomic_write_json(ctx.state_path, state)
        return ClaimedTrial(
            trial_index=next_index,
            combo_hash=ch,
            trial_params=trial_params,
            run_name=run_name,
            resume_from_path="",
            is_resume=False,
        )

    return None


def finalize_trial(ctx: TuneContext, record: dict[str, Any]) -> None:
    state = load_json_if_exists(ctx.state_path)
    if state is None:
        raise RuntimeError(f"Missing state file: {ctx.state_path}")
    trials: list[dict[str, Any]] = state["trials"]
    updated = False
    for i, t in enumerate(trials):
        if int(t["trial_index"]) == int(record["trial_index"]):
            trials[i] = record
            updated = True
            break
    if not updated:
        trials.append(record)
    state["trials"] = trials
    atomic_write_json(ctx.state_path, state)
    successes = [t for t in trials if t.get("status") == STATUS_OK]
    stats = timing_from_trials(trials)
    write_leaderboard_csv(ctx.leaderboard_path, successes)
    write_tune_report(
        report_path=ctx.report_path,
        config_path=ctx.config_path,
        trials_ok=successes,
        timing_stats=stats,
        report_param_keys=ctx.report_param_keys,
    )


def execute_claimed_trial(
    ctx: TuneContext,
    claim: ClaimedTrial,
    *,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    device: torch.device,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "trial_index": claim.trial_index,
        "combo_hash": claim.combo_hash,
        "params": claim.trial_params,
        "run_name": claim.run_name,
        "status": STATUS_FAILED,
        "error": None,
        "duration_seconds": 0.0,
        "val_mae": None,
        "val_rmse": None,
        "val_mard": None,
        "run_dir": str(ctx.out_root / claim.run_name),
        "non_retryable": False,
        "resume_checkpoint": claim.resume_from_path,
    }
    resume_path = claim.resume_from_path or ctx.resume_from
    if not resume_path:
        resume_path = resolve_trial_checkpoint(ctx.out_root / claim.run_name)
    cfg = build_train_cfg(
        csv_resolved=ctx.csv_path,
        dataset_cfg=ctx.dataset,
        params=claim.trial_params,
        runtime=ctx.runtime,
        device_name=ctx.device_name,
        seed=ctx.rng_seed,
        out_dir=ctx.out_root,
        resume_from=resume_path,
    )
    echo_plain("")
    echo_plain("=" * 72)
    ckpt_note = checkpoint_resume_summary(ctx.out_root / claim.run_name)
    resume_note = f" | resume={resume_path}" if resume_path else ""
    if ckpt_note:
        resume_note = f"{resume_note} | {ckpt_note}"
    echo_plain(
        f"Trial {claim.trial_index} | "
        f"{'RESUME' if claim.is_resume or resume_path else 'NEW'} | "
        f"hash={claim.combo_hash[:16]}... | {format_trial_params(claim.trial_params)}"
        f"{resume_note}"
    )
    echo_plain("=" * 72)

    t0 = time.perf_counter()
    try:
        run_dir = run_one_global_trial(
            cfg=cfg,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            device=device,
            run_name=claim.run_name,
            out_root=ctx.out_root,
        )
        metrics = read_split_metrics(run_dir, "val")
        elapsed = time.perf_counter() - t0
        if metrics is None:
            raise RuntimeError("Missing val_metrics_overall.csv after training.")
        record["status"] = STATUS_OK
        record["duration_seconds"] = elapsed
        record["val_mae"] = metrics["mae"]
        record["val_rmse"] = metrics["rmse"]
        record["val_mard"] = metrics["mard"]
        meta = tg.read_checkpoint_meta(run_dir / "last_checkpoint.pt")
        if meta is not None:
            record["last_completed_epoch"] = int(meta["epoch"])
        state_now = load_json_if_exists(ctx.state_path) or {"trials": []}
        ok_trials = [t for t in state_now["trials"] if t.get("status") == STATUS_OK]
        ok_n = len(ok_trials) + 1
        stats = timing_from_trials(ok_trials + [record])
        echo_plain(
            f"OK Done in {elapsed:.1f}s | val MAE={metrics['mae']:.4f} | "
            f"mean ok-trial time={stats['mean_seconds_per_trial']:.1f}s | "
            f"completed {ok_n}/{ctx.n_trials_target}"
        )
        eta_remain = stats["mean_seconds_per_trial"] * (ctx.n_trials_target - ok_n)
        if eta_remain > 0:
            unit = "min" if eta_remain < 7200 else "h"
            eta_val = eta_remain / 60 if unit == "min" else eta_remain / 3600
            echo_plain(f"  ETA (~remaining trials): {eta_val:.1f} {unit}")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        record["duration_seconds"] = elapsed
        record["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        record["non_retryable"] = is_non_retryable_failure(record["error"])
        if record["non_retryable"]:
            echo_plain("  Marked non_retryable (will not auto-resume).")
        echo_plain(f"FAIL Trial failed after {elapsed:.1f}s: {exc}")
        run_dir = Path(record["run_dir"])
        ckpt = resolve_trial_checkpoint(run_dir)
        if ckpt:
            record["resume_checkpoint"] = ckpt
            meta = tg.read_checkpoint_meta(Path(ckpt))
            if meta is not None:
                record["last_completed_epoch"] = int(meta["epoch"])
                record["resume_from_epoch"] = int(meta["epoch"]) + 1
    else:
        record["non_retryable"] = False
        record["resume_checkpoint"] = resolve_trial_checkpoint(Path(record["run_dir"]))
    return record


def tune_loop(
    *,
    user_cfg: dict[str, Any],
    config_path: Path,
    device_name: str,
    seed_override: int | None,
) -> None:
    ctx = build_tune_context(
        user_cfg=user_cfg,
        config_path=config_path,
        device_name=device_name,
        seed_override=seed_override,
    )
    tune = user_cfg["tune"]

    torch.manual_seed(ctx.rng_seed)
    np.random.seed(ctx.rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ctx.rng_seed)

    device = make_device(ctx.device_name)
    setup_cuda_flags(bool(tune["disable_tf32"]), device)

    echo_plain(f"Config: {ctx.config_path}")
    echo_plain(f"Device: {device}")
    echo_plain(f"CSV: {ctx.csv_path}")
    echo_plain(f"Output: {ctx.out_root}")
    if ctx.space:
        searched = sorted(ctx.space.keys())
        fixed_keys = sorted(k for k in ctx.param_defaults if k not in ctx.space)
        echo_plain(f"Search space ({len(searched)} keys): {', '.join(searched)}")
        if fixed_keys:
            echo_plain(f"Defaults (not in .space): {', '.join(fixed_keys)}")
    else:
        echo_plain(
            "No [tune.space]; single trial using [defaults] only "
            f"({len(ctx.param_defaults)} keys)."
        )
        if int(tune["n_trials"]) > 1:
            echo_plain(
                f"[tune].n_trials={tune['n_trials']} ignored without search space; "
                "running 1 trial."
            )

    ensure_state_initialized(ctx)
    n_reconciled = reconcile_trial_state(ctx)
    if n_reconciled:
        echo_plain(f"Reconciled {n_reconciled} trial state record(s).")

    state = load_json_if_exists(ctx.state_path)
    trials = state["trials"] if state else []
    if _success_count(trials) >= ctx.n_trials_target:
        echo_plain(
            f"Already completed {_success_count(trials)} >= target {ctx.n_trials_target}. "
            "Regenerating report."
        )
        refresh_artifacts(ctx)
        echo_plain(f"Report: {ctx.report_path}")
        echo_plain(f"Leaderboard: {ctx.leaderboard_path}")
        return

    train_df, val_df, test_df = prepare_frames(
        ctx.csv_path,
        unique_id=str(ctx.dataset["unique_id"]),
        drop_interpolated=bool(ctx.dataset["drop_interpolated"]),
        study_groups=str(ctx.dataset["study_groups"]),
        split_scheme=str(ctx.dataset["split_scheme"]),
        max_train_series=int(ctx.dataset["max_train_series"]),
        max_eval_series=int(ctx.dataset["max_eval_series"]),
    )
    echo_plain(
        f"Frames: train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,} rows"
    )

    while True:
        claim = claim_next_trial(ctx)
        if claim is None:
            break
        record = execute_claimed_trial(
            ctx,
            claim,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            device=device,
        )
        finalize_trial(ctx, record)
        echo_plain(f"  Updated leaderboard + report: {ctx.leaderboard_path}")

        state = load_json_if_exists(ctx.state_path)
        trials = state["trials"] if state else []
        if _success_count(trials) >= ctx.n_trials_target:
            break

    state = load_json_if_exists(ctx.state_path)
    trials = state["trials"] if state else []
    ok_n = _success_count(trials)
    if ok_n < ctx.n_trials_target:
        echo_plain(
            f"\nStopped: could not reach {ctx.n_trials_target} successes (have {ok_n})."
        )
        raise typer.Exit(code=1)

    echo_plain("\nTuning complete.")
    echo_plain(f"Report: {ctx.report_path}")
    echo_plain(f"Leaderboard: {ctx.leaderboard_path}")


@app.callback(invoke_without_command=True)
def cli(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help=f"TOML config (default: {DEFAULT_CONFIG_FILENAME} next to this script).",
    ),
    device_name: str = typer.Option(
        "cuda",
        "--device",
        help="cuda | cpu | mps",
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Override [tune].random_seed.",
    ),
) -> None:
    """Random hyperparameter search for SugarOne (global mode)."""
    config_path = resolve_config_path(config)
    user_cfg = load_user_config(config_path)
    if "tune" not in user_cfg:
        raise typer.BadParameter("Config must define [tune].")
    tune_loop(
        user_cfg=user_cfg,
        config_path=config_path,
        device_name=device_name,
        seed_override=seed,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
