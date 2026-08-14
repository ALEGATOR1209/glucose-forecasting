"""EpochMetricsWriter: the curve must survive a crash, a resume, and a skipped val."""
from __future__ import annotations

import csv

from common.metrics_log import EpochMetricsWriter


def _rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_header_and_rows(tmp_path):
    path = tmp_path / "m.csv"
    with EpochMetricsWriter(path) as w:
        w.log({"epoch": 1, "train_loss": 0.5, "val_loss": 0.4})
        w.log({"epoch": 2, "train_loss": 0.3, "val_loss": 0.35})

    rows = _rows(path)
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert rows[0]["train_loss"] == "0.5"


def test_skipped_validation_is_blank_not_zero(tmp_path):
    """A skipped val epoch is missing data. Writing 0.0 would plot as a perfect
    score and quietly ruin the curve."""
    path = tmp_path / "m.csv"
    with EpochMetricsWriter(path) as w:
        w.log({"epoch": 1, "train_loss": 0.5, "val_loss": 0.4})
        w.log({"epoch": 2, "train_loss": 0.3, "val_loss": ""})

    assert _rows(path)[1]["val_loss"] == ""


def test_rows_are_flushed_immediately(tmp_path):
    """Plottable mid-run, and a killed run keeps the epochs it finished."""
    path = tmp_path / "m.csv"
    w = EpochMetricsWriter(path)
    w.log({"epoch": 1, "train_loss": 0.5})
    assert len(_rows(path)) == 1  # readable without close()
    w.close()


def test_resume_appends_and_keeps_one_header(tmp_path):
    path = tmp_path / "m.csv"
    with EpochMetricsWriter(path) as w:
        w.log({"epoch": 1, "train_loss": 0.5})
    with EpochMetricsWriter(path) as w:  # a --resume-from run
        w.log({"epoch": 2, "train_loss": 0.3})

    rows = _rows(path)
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert path.read_text().count("epoch,train_loss") == 1


def test_unknown_keys_do_not_shift_columns(tmp_path):
    path = tmp_path / "m.csv"
    with EpochMetricsWriter(path) as w:
        w.log({"epoch": 1, "train_loss": 0.5})
        w.log({"epoch": 2, "train_loss": 0.3, "surprise": 9})

    rows = _rows(path)
    assert list(rows[1].keys()) == ["epoch", "train_loss"]
    assert rows[1]["train_loss"] == "0.3"
