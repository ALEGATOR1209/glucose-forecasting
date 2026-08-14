"""Per-epoch metrics -> CSV, for plotting a run after (or during) training.

One row per epoch, flushed immediately, so a curve is plottable while the run is
still going and survives a crash or an early stop. Resuming a run appends to the
existing file rather than truncating it.

Deliberately dependency-free (stdlib csv) and shape-agnostic: the field names are
taken from the first row logged, so each trainer decides what its columns are.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class EpochMetricsWriter:
    """Append-only CSV writer, one row per epoch.

    Field names are fixed by the first `log()` call. Later rows may omit keys
    (they come out blank) but unknown keys are dropped rather than silently
    shifting columns.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._fieldnames: list[str] | None = None
        self._fh = None
        self._writer: csv.DictWriter | None = None

    def _open(self, fieldnames: list[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        resuming = self.path.exists() and self.path.stat().st_size > 0
        if resuming:
            with self.path.open("r", newline="") as fh:
                header = next(csv.reader(fh), None)
            if header:
                fieldnames = header  # keep the original column order on resume

        self._fh = self.path.open("a" if resuming else "w", newline="")
        self._fieldnames = fieldnames
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames, extrasaction="ignore")
        if not resuming:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            self._open(list(row.keys()))
        assert self._writer is not None and self._fh is not None
        self._writer.writerow(row)
        self._fh.flush()  # plot mid-run; survive a kill -9

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None

    def __enter__(self) -> EpochMetricsWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
