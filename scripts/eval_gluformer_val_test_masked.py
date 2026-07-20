#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from transformers import AutoConfig, AutoModel
    from transformers.modeling_utils import PreTrainedModel
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependency: install `transformers` (and torch) to use Gluformer.\n"
        "Example: uv add transformers torch\n"
        f"Original error: {exc}"
    )

# ---- Source CSV columns ----
COL_SEQ = "sequence_id"
COL_USER = "User ID"
COL_TS = "Timestamp (YYYY-MM-DDThh:mm:ss)"
COL_SPLIT = "Recommended Split"
COL_GROUP = "Study Group"
COL_EVENT = "Event Type"
COL_GLU = "Glucose Value (mg/dL)"

TS_FORMAT = "%Y-%m-%dT%H:%M:%S"


def ensure_tied_weights_compat():
    """
    Some custom HF models (e.g., Gluformer) ship without `all_tied_weights_keys`,
    but newer Transformers expects it. Patch in a fallback for compatibility.
    """
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = property(  # type: ignore[attr-defined]
            lambda self: (getattr(self, "_tied_weights_keys", None) or {})
        )

    if hasattr(PreTrainedModel, "mark_tied_weights_as_initialized"):
        orig = PreTrainedModel.mark_tied_weights_as_initialized

        def _patched(self):  # type: ignore[override]
            try:
                return orig(self)
            except AttributeError:
                return None

        PreTrainedModel.mark_tied_weights_as_initialized = _patched  # type: ignore[assignment]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--unique_id", choices=["sequence_id", "user_id"], default="sequence_id")
    ap.add_argument("--model_id", type=str, default="njeffrie/Gluformer")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    ap.add_argument(
        "--splits",
        choices=["val", "test", "both"],
        default="both",
        help="Which split(s) to evaluate.",
    )

    ap.add_argument("--chunk_size", type=int, default=1_000_000)
    ap.add_argument("--max_eval_series", type=int, default=0)
    ap.add_argument("--max_points_per_series", type=int, default=0)

    ap.add_argument("--drop_interpolated", action="store_true",
                    help="Drop rows where Event Type == 'Interpolated' before evaluation.")
    ap.add_argument("--mask_interpolated_targets", action="store_true",
                    help="Exclude interpolated target rows from metrics.")

    ap.add_argument("--out_dir", type=Path, default=Path("data/output/runs"))
    ap.add_argument("--save_predictions", action="store_true")
    return ap.parse_args()


def load_splits_streaming(
    csv_path: Path,
    unique_id_choice: str,
    chunk_size: int,
    drop_interpolated: bool,
):
    uid_col = COL_SEQ if unique_id_choice == "sequence_id" else COL_USER
    usecols = [uid_col, COL_TS, COL_SPLIT, COL_GROUP, COL_EVENT, COL_GLU]

    val_parts, test_parts = [], []
    print("Loading VAL/TEST splits (streaming)...")

    seen = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, low_memory=False):
        seen += len(chunk)
        if seen % 6_000_000 == 0:
            print(f"... loaded {seen:,} rows")

        chunk[COL_TS] = pd.to_datetime(chunk[COL_TS], format=TS_FORMAT, errors="coerce")
        chunk = chunk.rename(
            columns={
                uid_col: "unique_id",
                COL_TS: "ds",
                COL_GLU: "y",
                COL_GROUP: "study_group",
                COL_SPLIT: "split",
                COL_EVENT: "event_type",
            }
        )
        chunk = chunk.dropna(subset=["unique_id", "ds", "split", "study_group"])

        chunk["y"] = pd.to_numeric(chunk["y"], errors="coerce")

        va = chunk[chunk["split"] == "val"]
        te = chunk[chunk["split"] == "test"]

        if drop_interpolated:
            va = va[va["event_type"] != "Interpolated"]
            te = te[te["event_type"] != "Interpolated"]

        if not va.empty:
            val_parts.append(va)
        if not te.empty:
            test_parts.append(te)

    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    return val_df, test_df


def basic_impute_and_types(df: pd.DataFrame, max_points_per_series: int = 0) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)

    if max_points_per_series and max_points_per_series > 0:
        df = df.groupby("unique_id", sort=False, observed=True).tail(max_points_per_series).copy()
        df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)

    # Keep ids as strings so both numeric and non-numeric ids are supported.
    df["unique_id"] = df["unique_id"].astype(str).str.strip()
    df = df[df["unique_id"] != ""].copy()

    df["y"] = df.groupby("unique_id", sort=False)["y"].ffill()
    df["y"] = df.groupby("unique_id", sort=False)["y"].bfill()
    df["y"] = df["y"].fillna(0.0)

    df["y"] = df["y"].astype(np.float32)
    df["study_group"] = df["study_group"].astype("category")
    return df


def maybe_limit_series(df: pd.DataFrame, max_series: int) -> pd.DataFrame:
    if df.empty or not max_series or max_series <= 0:
        return df
    keep = df["unique_id"].drop_duplicates().head(max_series).to_numpy()
    return df[df["unique_id"].isin(keep)].copy()


def filter_min_length(df: pd.DataFrame, min_len: int) -> pd.DataFrame:
    if df.empty:
        return df
    sizes = df.groupby("unique_id", sort=False, observed=True).size()
    keep = sizes[sizes >= min_len].index
    return df[df["unique_id"].isin(keep)].copy()


def mae_rmse_mard(y_true: np.ndarray, y_pred: np.ndarray):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    nonzero = y_true != 0
    if nonzero.any():
        mard = float(np.mean(np.abs(err[nonzero]) / np.abs(y_true[nonzero])) * 100)
    else:
        mard = float("nan")
    return mae, rmse, mard


def get_pred_len(cfg) -> int:
    for attr in ("pred_len", "prediction_length", "forecast_horizon", "horizon", "len_pred"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    return 12  # default: 60min at 5min frequency


def eval_split(
    split_name: str,
    df: pd.DataFrame,
    model,
    subject_map: dict[str, int],
    input_len: int,
    pred_len: int,
    run_dir: Path,
    save_predictions: bool,
    mask_interpolated_targets: bool,
):
    if df.empty:
        print(f"[{split_name}] empty split, skipping.")
        return

    df = df.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    min_len = input_len + pred_len
    sizes = df.groupby("unique_id", sort=False, observed=True).size()
    n_series_before = int(sizes.shape[0])
    keep = sizes[sizes >= min_len].index
    n_series_after = int(len(keep))
    n_series_skipped = n_series_before - n_series_after
    if n_series_skipped > 0:
        print(
            f"[{split_name}] skipped {n_series_skipped:,} series shorter than {min_len} points."
        )
    df = df[df["unique_id"].isin(keep)].copy()
    if df.empty:
        print(f"[{split_name}] no series left after min-length filtering.")
        return

    n_series = df["unique_id"].nunique()
    print(f"Evaluating {split_name}: rows={len(df):,} | series={n_series:,}")

    rows = []
    dropped = 0
    for uid, g in df.groupby("unique_id", sort=False, observed=True):
        uid_key = str(uid)
        g = g.sort_values("ds", kind="mergesort")
        tail = g.tail(input_len + pred_len)
        hist = tail.iloc[:input_len]
        true = tail.iloc[input_len:]

        subject_id = int(subject_map[uid_key])
        timestamps = np.asarray(
            [pd.Timestamp(ts).value for ts in hist["ds"].to_list()],
            dtype=np.int64,
        )[None, :]
        input_glucose = hist["y"].to_numpy(np.float32)[None, :]

        with torch.no_grad():
            pred, _log_var = model(subject_id, timestamps, input_glucose)

        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        pred = np.asarray(pred, dtype=np.float32).reshape(-1)
        if len(pred) < pred_len:
            pad = np.full(pred_len - len(pred), pred[-1] if len(pred) else 0.0, dtype=np.float32)
            pred = np.concatenate([pred, pad], axis=0)
        pred = pred[:pred_len]

        for ts, y_true, y_hat, sg, ev in zip(
            true["ds"].to_list(),
            true["y"].to_numpy(np.float32),
            pred,
            true["study_group"].to_list(),
            true["event_type"].to_list(),
        ):
            if mask_interpolated_targets and ev == "Interpolated":
                dropped += 1
                continue
            rows.append(
                {
                    "unique_id": uid_key,
                    "ds": ts,
                    "y": float(y_true),
                    "yhat": float(y_hat),
                    "study_group": sg,
                    "event_type": ev,
                }
            )

    if dropped:
        print(f"[{split_name}] dropped {dropped:,} interpolated target rows from metrics.")

    eval_df = pd.DataFrame(rows)
    if eval_df.empty:
        print(f"[{split_name}] no evaluation rows after filtering.")
        return

    overall_mae, overall_rmse, overall_mard = mae_rmse_mard(
        eval_df["y"].to_numpy(np.float32), eval_df["yhat"].to_numpy(np.float32)
    )

    print(f"\n=== {split_name.upper()} METRICS (overall) ===")
    print(f"MAE : {overall_mae:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"MARD: {overall_mard:.4f}%")

    group_rows = []
    for g, part in eval_df.groupby("study_group", observed=True):
        m, r, md = mae_rmse_mard(part["y"].to_numpy(np.float32), part["yhat"].to_numpy(np.float32))
        group_rows.append((str(g), len(part), m, r, md))

    by_group = (
        pd.DataFrame(group_rows, columns=["study_group", "n_points", "mae", "rmse", "mard"])
        .sort_values("mae")
        .reset_index(drop=True)
    )

    print(f"\n=== {split_name.upper()} METRICS (by Study Group) ===")
    print(by_group.to_string(index=False))

    by_group.to_csv(run_dir / f"{split_name}_metrics_by_study_group.csv", index=False)
    pd.DataFrame([{"mae": overall_mae, "rmse": overall_rmse, "mard": overall_mard}]).to_csv(
        run_dir / f"{split_name}_metrics_overall.csv", index=False
    )

    if save_predictions:
        eval_df.to_csv(run_dir / f"{split_name}_predictions.csv", index=False)


def main():
    args = parse_args()

    val_df, test_df = load_splits_streaming(
        args.csv, args.unique_id, args.chunk_size, args.drop_interpolated
    )
    val_df = basic_impute_and_types(val_df, args.max_points_per_series)
    test_df = basic_impute_and_types(test_df, args.max_points_per_series)

    if args.max_eval_series and args.max_eval_series > 0:
        val_df = maybe_limit_series(val_df, args.max_eval_series)
        test_df = maybe_limit_series(test_df, args.max_eval_series)
        print(f"SMOKE TEST: limited eval series to {args.max_eval_series}")

    print(f"Val   rows: {len(val_df):,} | series: {val_df['unique_id'].nunique():,}")
    print(f"Test  rows: {len(test_df):,} | series: {test_df['unique_id'].nunique():,}")

    print(f"Loading Gluformer from Hugging Face: {args.model_id}")
    ensure_tied_weights_compat()
    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_id, trust_remote_code=True)
    model.to(args.device)
    model.eval()

    input_len = int(getattr(config, "len_seq", 12))
    pred_len = get_pred_len(config)
    print(f"Model input_len={input_len} | pred_len={pred_len}")

    all_ids = (
        pd.concat([val_df["unique_id"], test_df["unique_id"]], ignore_index=True)
        .dropna()
        .astype(str)
        .unique()
    )
    subject_map = {uid: idx for idx, uid in enumerate(sorted(all_ids))}

    run_dir = args.out_dir / f"gluformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    split_frames = {"val": val_df, "test": test_df}
    splits_to_run = ["val", "test"] if args.splits == "both" else [args.splits]
    for split_name in splits_to_run:
        eval_split(
            split_name=split_name,
            df=split_frames[split_name],
            model=model,
            subject_map=subject_map,
            input_len=input_len,
            pred_len=pred_len,
            run_dir=run_dir,
            save_predictions=args.save_predictions,
            mask_interpolated_targets=args.mask_interpolated_targets,
        )

    print(f"\nSaved run to: {run_dir}")


if __name__ == "__main__":
    main()

# Example:
# uv run python scripts/eval_gluformer_val_test_masked.py --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv --device mps --mask_interpolated_targets --save_predictions
