"""Smoke tests for SugarOne random tuner (global mode)."""
from __future__ import annotations

from pathlib import Path

from scripts.sugar_one.tune_sugar_one import (
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_OK,
    STATUS_RUNNING,
    TuneContext,
    build_tune_context,
    claim_next_trial,
    finalize_trial,
    reconcile_trial_state,
    try_claim_resumable_trial,
    tune_loop,
)
from tests.conftest import write_sugar_one_csv

DEFAULTS = {"horizon": 2, "input_steps": 4, "batch_size": 4}
RUNTIME = {
    "n_trials": 5,
    "random_seed": 1,
    "max_random_draws": 20,
    "epochs": 1,
    "patience": 0,
    "log_every": 1,
    "val_every_n_epochs": 1,
    "ckpt_every_n_epochs": 0,
    "precision": "fp32",
    "compile_mode": "none",
    "disable_tf32": False,
    "num_workers": 0,
    "prefetch_factor": 2,
    "resume_from": "",
}


def _tune_context(
    tmp_path: Path,
    *,
    defaults: dict[str, object] | None = None,
    space: dict[str, list[float]] | None = None,
    n_trials: int = 5,
) -> TuneContext:
    config_path = tmp_path / "tune.toml"
    config_path.write_text("", encoding="utf-8")
    tune = {**RUNTIME, "n_trials": n_trials}
    if space is not None:
        tune["space"] = space
    return build_tune_context(
        user_cfg={
            "paths": {
                "csv": str(tmp_path / "data.csv"),
                "output_dir": str(tmp_path / "tune_out"),
            },
            "dataset": {
                "unique_id": "sequence_id",
                "drop_interpolated": False,
                "study_groups": "",
                "split_scheme": "classic",
                "max_train_series": 0,
                "max_eval_series": 0,
            },
            "defaults": defaults or DEFAULTS,
            "tune": tune,
        },
        config_path=config_path,
        device_name="cpu",
        seed_override=None,
    )


def test_reconcile_running_and_resume_priority(tmp_path: Path) -> None:
    ctx = _tune_context(tmp_path, space={"lr": [0.01]})
    out = ctx.out_root
    run_dir = out / "trial_0001_deadbeef"
    run_dir.mkdir(parents=True)
    ctx.state_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    state = {
        "version": 1,
        "trials": [
            {
                "trial_index": 1,
                "combo_hash": "deadbeef",
                "params": {"lr": 0.01},
                "run_name": "trial_0001_deadbeef",
                "status": STATUS_RUNNING,
                "run_dir": str(run_dir),
            },
            {
                "trial_index": 2,
                "combo_hash": "oomhash",
                "params": {"lr": 0.02},
                "run_name": "trial_0002_oomhash",
                "status": STATUS_FAILED,
                "non_retryable": False,
                "error": "OutOfMemoryError: CUDA out of memory",
                "run_dir": str(out / "trial_0002_oomhash"),
            },
        ],
        "next_draw_index": 0,
    }
    ctx.state_path.write_text(json.dumps(state), encoding="utf-8")

    n = reconcile_trial_state(ctx)
    assert n >= 1
    data = json.loads(ctx.state_path.read_text(encoding="utf-8"))
    t1 = data["trials"][0]
    assert t1["status"] == STATUS_INTERRUPTED
    t2 = data["trials"][1]
    assert t2["non_retryable"] is True

    resumed = try_claim_resumable_trial(ctx)
    assert resumed is not None
    assert resumed.trial_index == 1
    assert resumed.is_resume is True

    again = try_claim_resumable_trial(ctx)
    assert again is None or again.trial_index != 2


def test_claim_next_trial_no_duplicate_hash(tmp_path: Path) -> None:
    ctx = _tune_context(
        tmp_path,
        space={"lr": [0.01, 0.02, 0.03, 0.04]},
    )
    out = ctx.out_root
    c0 = claim_next_trial(ctx)
    assert c0 is not None
    finalize_trial(
        ctx,
        {
            "trial_index": c0.trial_index,
            "combo_hash": c0.combo_hash,
            "params": c0.trial_params,
            "run_name": c0.run_name,
            "status": STATUS_OK,
            "non_retryable": False,
            "error": None,
            "duration_seconds": 1.0,
            "val_mae": 1.0,
            "val_rmse": 1.0,
            "val_mard": 1.0,
            "run_dir": str(out / c0.run_name),
        },
    )
    c1 = claim_next_trial(ctx)
    assert c1 is not None
    assert c0.combo_hash != c1.combo_hash
    finalize_trial(
        ctx,
        {
            "trial_index": c1.trial_index,
            "combo_hash": c1.combo_hash,
            "params": c1.trial_params,
            "run_name": c1.run_name,
            "status": STATUS_OK,
            "non_retryable": False,
            "error": None,
            "duration_seconds": 1.0,
            "val_mae": 1.0,
            "val_rmse": 1.0,
            "val_mard": 1.0,
            "run_dir": str(out / c1.run_name),
        },
    )
    c2 = claim_next_trial(ctx)
    assert c2 is not None
    assert c2.combo_hash not in {c0.combo_hash, c1.combo_hash}


def test_build_tune_context_defaults_only_no_space(tmp_path: Path) -> None:
    defaults = {"lr": 0.001, "batch_size": 8, "horizon": 2, "input_steps": 4}
    ctx = _tune_context(
        tmp_path,
        defaults=defaults,
        n_trials=99,
    )
    assert ctx.space == {}
    assert ctx.n_trials_target == 1
    claim = claim_next_trial(ctx)
    assert claim is not None
    assert claim.trial_params == defaults


def test_tune_one_trial_cpu(tmp_path: Path) -> None:
    csv_path = tmp_path / "loop_ic_mini.csv"
    write_sugar_one_csv(
        csv_path,
        series=[
            ("smoke-tr-a", "train", "T1DM", 40, 100.0),
            ("smoke-val-b", "val", "T1DM", 35, 110.0),
            ("smoke-te-c", "test", "T1DM", 25, 105.0),
        ],
    )
    out = tmp_path / "tune_out"

    user_cfg: dict = {
        "paths": {
            "csv": str(csv_path),
            "output_dir": str(out),
        },
        "dataset": {
            "unique_id": "sequence_id",
            "drop_interpolated": False,
            "study_groups": "",
            "split_scheme": "classic",
            "max_train_series": 0,
            "max_eval_series": 0,
        },
        "defaults": {
            "d_model": 16,
            "n_heads": 4,
            "n_blocks": 1,
            "ff_units": 32,
            "dropout": 0.1,
            "input_steps": 8,
            "horizon": 2,
            "weight_decay": 0.0001,
            "batch_size": 8,
        },
        "tune": {
            "n_trials": 1,
            "random_seed": 123,
            "max_random_draws": 50,
            "epochs": 1,
            "patience": 0,
            "log_every": 1,
            "val_every_n_epochs": 1,
            "ckpt_every_n_epochs": 0,
            "precision": "fp32",
            "compile_mode": "none",
            "disable_tf32": False,
            "num_workers": 0,
            "prefetch_factor": 2,
            "resume_from": "",
            "space": {
                "lr": [0.001],
            },
        },
    }

    meta_path = tmp_path / "stub.toml"
    meta_path.write_text("# stub\n", encoding="utf-8")

    tune_loop(
        user_cfg=user_cfg,
        config_path=meta_path,
        device_name="cpu",
        seed_override=None,
    )

    state_path = out / "state.json"
    report_path = out / "tune_report.md"
    assert state_path.exists()
    assert report_path.exists()
    trial_dirs = sorted(out.glob("trial_*"))
    assert len(trial_dirs) == 1
    assert (trial_dirs[0] / "val_metrics_overall.csv").exists()
