"""Leaderboard and trial state for TOML-driven personalization tuning."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import polars as pl

STATE_VERSION = 1
STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_RUNNING = "running"

PARAM_KEYS = (
    "base_run_dir",
    "personal_csv",
    "lwf_lambda",
    "lr",
    "weight_decay",
    "patience",
    "epochs",
    "batch_size",
    "personal_days",
    "train_window_stride",
    "val_every_n_epochs",
    "precision",
    "eval_zero_shot",
)

METRIC_SORT_KEY = "ft_test_mae"


def resolve_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


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
        raise TypeError(f"Expected JSON object in {path}")
    return data


def _canonical_value(val: Any) -> Any:
    if isinstance(val, float):
        return round(val, 12)
    if isinstance(val, Path):
        return str(val.resolve())
    return val


def _normalize_param(key: str, val: Any) -> Any:
    if key in ("base_run_dir", "personal_csv") and isinstance(val, str):
        return str(resolve_path(val))
    return _canonical_value(val)


def combo_hash(params: Mapping[str, Any]) -> str:
    subset = {k: _normalize_param(k, params.get(k)) for k in PARAM_KEYS}
    blob = json.dumps(subset, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def params_from_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    params = {k: cfg.get(k) for k in PARAM_KEYS}
    if params.get("train_window_stride") is None:
        params["train_window_stride"] = 1
    if params.get("eval_zero_shot") is None:
        params["eval_zero_shot"] = True
    return params


def training_params_match(saved_cfg: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    """Match training identity; ``eval_zero_shot`` is not part of combo identity."""
    saved_p = params_from_config(saved_cfg)
    for key in PARAM_KEYS:
        if key == "eval_zero_shot":
            continue
        if _normalize_param(key, saved_p.get(key)) != _normalize_param(key, params.get(key)):
            return False
    return True


def find_resume_checkpoint(out_root: Path, params: Mapping[str, Any]) -> Path | None:
    """Return last_checkpoint.pt for a partial run matching ``params``, if any."""
    from scripts.common.checkpoint import read_checkpoint_meta

    for ckpt_path in sorted(out_root.rglob("last_checkpoint.pt")):
        run_dir = ckpt_path.parent
        cfg_path = run_dir / "config.json"
        if not cfg_path.is_file():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not cfg.get("personalization"):
            continue
        if not training_params_match(cfg, params):
            continue
        metrics_path = run_dir / "personalization_metrics.json"
        if metrics_path.is_file():
            results = json.loads(metrics_path.read_text(encoding="utf-8"))
            if results.get("finetuned_test") is not None:
                continue
        meta = read_checkpoint_meta(ckpt_path)
        if meta is None:
            continue
        if meta["epoch"] >= int(cfg.get("epochs", 0)):
            continue
        return ckpt_path
    return None


def merge_defaults(defaults: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    for key, val in override.items():
        if key == "name":
            merged["name"] = val
            continue
        merged[key] = val
    for key in PARAM_KEYS:
        merged.setdefault(key, None)
    return merged


def build_run_combos(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build explicit run list from [[runs]] or Cartesian product of [grid]."""
    defaults = dict(cfg.get("defaults", {}))
    paths = cfg.get("paths", {})
    for key in ("base_run_dir", "personal_csv"):
        if key in paths and key not in defaults:
            defaults[key] = paths[key]

    explicit = cfg.get("runs")
    if explicit:
        combos: list[dict[str, Any]] = []
        for entry in explicit:
            if not isinstance(entry, dict):
                raise ValueError("Each [[runs]] entry must be a table")
            combos.append(merge_defaults(defaults, entry))
        return combos

    grid = cfg.get("grid", {})
    if not grid:
        return [defaults]

    keys = sorted(grid.keys())
    value_lists = [list(grid[k]) for k in keys]
    combos = []
    for values in product(*value_lists):
        override = dict(zip(keys, values))
        combos.append(merge_defaults(defaults, override))
    return combos


def trial_record_from_run(
    *,
    run_index: int,
    params: Mapping[str, Any],
    run_name: str,
) -> dict[str, Any]:
    return {
        "run_index": run_index,
        "combo_hash": combo_hash(params),
        "run_name": run_name,
        "params": {k: params.get(k) for k in PARAM_KEYS if k in params},
        "status": STATUS_RUNNING,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "run_dir": None,
        "wall_time_s": None,
        "zs_test_mae": None,
        "ft_test_mae": None,
        "ft_val_mae": None,
        "error": None,
    }


def finalize_trial_from_results(
    record: dict[str, Any],
    *,
    run_dir: Path,
    results: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = results.get("config", {})
    zs = results.get("zero_shot_test") or {}
    ft_test = results.get("finetuned_test") or {}
    ft_val = results.get("finetuned_val") or {}
    record.update(
        {
            "status": STATUS_OK,
            "finished_at": datetime.now().isoformat(),
            "run_dir": str(run_dir),
            "wall_time_s": results.get("wall_time_s"),
            "zs_test_mae": zs.get("mae"),
            "ft_test_mae": ft_test.get("mae"),
            "ft_val_mae": ft_val.get("mae"),
            "train_windows": cfg.get("train_windows"),
            "error": None,
        }
    )
    return record


def grid_combo_hashes(cfg: Mapping[str, Any]) -> set[str]:
    """Hashes for every combo in the active TOML grid."""
    return {combo_hash(params) for params in build_run_combos(cfg)}


def leaderboard_row_from_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    params = trial.get("params", {})
    row: dict[str, Any] = {
        "run_index": trial.get("run_index"),
        "combo_hash": trial.get("combo_hash"),
        "status": trial.get("status"),
        "run_name": trial.get("run_name"),
        "zs_test_mae": trial.get("zs_test_mae"),
        "ft_test_mae": trial.get("ft_test_mae"),
        "ft_val_mae": trial.get("ft_val_mae"),
        "wall_time_s": trial.get("wall_time_s"),
        "train_windows": trial.get("train_windows"),
        "run_dir": trial.get("run_dir"),
    }
    for key in PARAM_KEYS:
        if key in params:
            row[key] = params[key]
    return row


def write_leaderboard_csv(
    path: Path,
    trials: list[Mapping[str, Any]],
    *,
    active_combo_hashes: set[str] | None = None,
) -> None:
    """Rewrite leaderboard from successful trials only (sorted by ft_test_mae).

    Failed/running trials stay in ``state.json``; errors are not written here.
    When ``active_combo_hashes`` is set, only trials matching the current grid
    are included (legacy runs from an old grid are omitted).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    successes = [t for t in trials if t.get("status") == STATUS_OK]
    if active_combo_hashes is not None:
        successes = [
            t for t in successes if str(t.get("combo_hash", "")) in active_combo_hashes
        ]
    if not successes:
        if path.exists():
            path.unlink()
        return
    rows = [leaderboard_row_from_trial(t) for t in successes]
    df = pl.DataFrame(rows)
    if METRIC_SORT_KEY in df.columns:
        df = df.sort(METRIC_SORT_KEY, nulls_last=True)
    df.write_csv(path)


def completed_hashes(
    trials: list[Mapping[str, Any]],
    *,
    active_combo_hashes: set[str] | None = None,
) -> set[str]:
    done = {
        str(t["combo_hash"])
        for t in trials
        if t.get("status") == STATUS_OK and t.get("combo_hash")
    }
    if active_combo_hashes is not None:
        done &= active_combo_hashes
    return done


def import_existing_runs(out_root: Path, trials: list[dict[str, Any]]) -> int:
    """Scan output_dir for finished runs and add missing OK trials to state."""
    known_hashes = completed_hashes(trials)
    known_dirs = {str(t.get("run_dir")) for t in trials if t.get("run_dir")}
    added = 0
    next_index = max((int(t.get("run_index", 0)) for t in trials), default=0)

    for metrics_path in sorted(out_root.rglob("personalization_metrics.json")):
        run_dir = metrics_path.parent
        if str(run_dir) in known_dirs:
            continue
        results = json.loads(metrics_path.read_text(encoding="utf-8"))
        cfg = results.get("config", {})
        if not cfg.get("personalization"):
            continue
        params = {k: cfg.get(k) for k in PARAM_KEYS}
        if params.get("train_window_stride") is None:
            params["train_window_stride"] = 1
        if params.get("eval_zero_shot") is None:
            params["eval_zero_shot"] = True
        ch = combo_hash(params)
        if ch in known_hashes:
            continue
        next_index += 1
        record = trial_record_from_run(
            run_index=next_index,
            params=params,
            run_name=run_dir.name,
        )
        finalize_trial_from_results(record, run_dir=run_dir, results=results)
        trials.append(record)
        known_hashes.add(ch)
        known_dirs.add(str(run_dir))
        added += 1
    return added
