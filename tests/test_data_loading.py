"""Unit tests for scripts/common/data_loading.py."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from scripts.common.data_loading import (
    apply_split_scheme,
    impute_and_sort,
    limit_series,
    load_splits_streaming,
    normalize_study_group_label,
    normalize_study_groups_column,
)

from tests.conftest import write_glumind_csv


def _load_glumind_splits(
    csv_path: Path,
    *,
    drop_interpolated: bool,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return load_splits_streaming(
        csv_path,
        "sequence_id",
        drop_interpolated=drop_interpolated,
        col_seq="sequence_id",
        col_user="User ID",
        col_ts="Timestamp (YYYY-MM-DDThh:mm:ss)",
        col_split="Recommended Split",
        col_group="Study Group",
        col_event="Event Type",
        value_columns={
            "glucose": "Glucose Value (mg/dL)",
            "hr": "Heart Rate",
            "steps": "Step Count",
        },
        ts_format="%Y-%m-%dT%H:%M:%S",
    )


def test_load_splits_streaming_partitions_and_renames(tmp_path: Path) -> None:
    csv_path = tmp_path / "glumind.csv"
    write_glumind_csv(
        csv_path,
        series=[
            ("a", "train", "T1DM", 5, 100.0),
            ("b", "val", "Healthy", 4, 110.0),
            ("c", "test", "T1DM", 3, 105.0),
        ],
    )

    train_df, val_df, test_df = _load_glumind_splits(
        csv_path,
        drop_interpolated=False,
    )

    assert len(train_df) == 5
    assert len(val_df) == 4
    assert len(test_df) == 3
    # Renamed/canonical columns present.
    for df in (train_df, val_df, test_df):
        assert set(["unique_id", "ds", "study_group", "split", "event_type", "glucose", "hr", "steps"]).issubset(
            set(df.columns)
        )
    assert train_df["glucose"].dtype == pl.Float32


def test_load_splits_streaming_drop_interpolated(tmp_path: Path) -> None:
    csv_path = tmp_path / "glumind.csv"
    write_glumind_csv(csv_path, series=[("a", "train", "T1DM", 6, 100.0)])

    # Manually inject one Interpolated row by rewriting the CSV.
    df = pl.read_csv(csv_path)
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(pl.lit("Interpolated")).otherwise(pl.col("Event Type")).alias("Event Type")
    )
    df.write_csv(csv_path)

    train_df, _, _ = _load_glumind_splits(csv_path, drop_interpolated=True)
    assert len(train_df) == 5  # one row dropped

    train_df_kept, _, _ = _load_glumind_splits(
        csv_path,
        drop_interpolated=False,
    )
    assert len(train_df_kept) == 6


def _splits() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train = pl.DataFrame({"unique_id": ["a"], "ds": [1], "split": ["train"]})
    val = pl.DataFrame({"unique_id": ["b"], "ds": [2], "split": ["val"]})
    test = pl.DataFrame({"unique_id": ["c"], "ds": [3], "split": ["test"]})
    return train, val, test


def test_apply_split_scheme_trainval_test_as_val() -> None:
    train, val, test = _splits()
    out_train, out_val, out_test = apply_split_scheme(train, val, test, "trainval_test_as_val")
    assert len(out_train) == 2  # train + val merged
    assert set(out_train["unique_id"].to_list()) == {"a", "b"}
    assert out_val.equals(test)
    assert out_test.is_empty()


def test_apply_split_scheme_unknown_raises() -> None:
    train, val, test = _splits()
    with pytest.raises(ValueError, match="Unknown split_scheme"):
        apply_split_scheme(train, val, test, "bogus_scheme")


def test_apply_split_scheme_empty_test_raises() -> None:
    train, val, test = _splits()
    with pytest.raises(ValueError, match="non-empty test split"):
        apply_split_scheme(train, val, test.clear(), "trainval_test_as_val")


def test_impute_and_sort_uses_feature_and_series_specific_fill_rules() -> None:
    df = pl.DataFrame(
        {
            "unique_id": ["a", "a", "a", "b", "b", "c", "c"],
            "ds": [1, 2, 3, 1, 2, 1, 2],
            "glucose": [10.0, None, 30.0, None, 50.0, None, None],
            "bolus": [5.0, None, None, None, 2.0, None, None],
        }
    )
    out = impute_and_sort(
        df,
        ffill_bfill_columns=["glucose"],
        zero_fill_columns=["bolus"],
    ).sort(["unique_id", "ds"])

    assert out.filter(pl.col("unique_id") == "a")["glucose"].to_list() == [
        10.0,
        10.0,
        30.0,
    ]
    assert out.filter(pl.col("unique_id") == "b")["glucose"].to_list() == [
        50.0,
        50.0,
    ]
    assert out.filter(pl.col("unique_id") == "c")["glucose"].to_list() == [
        0.0,
        0.0,
    ]
    assert out["bolus"].to_list() == [5.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0]


def test_limit_series_caps_to_first_n() -> None:
    df = pl.DataFrame({"unique_id": ["a", "a", "b", "b", "c", "c"], "val": [1, 2, 3, 4, 5, 6]})
    out = limit_series(df, 2)
    assert set(out["unique_id"].unique().to_list()) == {"a", "b"}
    assert len(out) == 4


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("prediabetes", "Pre-T2DM"),
        ("Pre-Diabetes", "Pre-T2DM"),
        ("oral_medication", "Oral-T2DM"),
        ("insulin_dependent", "Insulin-T2DM"),
        ("healthy", "Healthy"),
    ],
)
def test_normalize_study_group_label_aliases(raw: str, expected: str) -> None:
    assert normalize_study_group_label(raw) == expected


def test_normalize_study_group_label_passthrough_unmapped() -> None:
    assert normalize_study_group_label("T1DM") == "T1DM"
    assert normalize_study_group_label("SomeUnknownLabel") == "SomeUnknownLabel"


def test_normalize_study_groups_column() -> None:
    df = pl.DataFrame({"study_group": ["prediabetes", "T1DM", "healthy"]})
    out = normalize_study_groups_column(df)
    assert out["study_group"].to_list() == ["Pre-T2DM", "T1DM", "Healthy"]