#!/usr/bin/env python3
"""Aggregate personalization sweep and holdout summaries into one artifact."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import polars as pl
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _read_summary_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pl.read_csv(path)
    return df.to_dicts()


@app.command()
def main(
    root: Path = typer.Option(
        Path("data/output/runs/personalization"),
        "--root",
        help="Root directory containing sweep/holdout artifacts.",
    ),
    out: Path = typer.Option(
        Path("docs/reports/milestone8_personalization_summary.json"),
        "--out",
    ),
    out_csv: Optional[Path] = typer.Option(
        Path("docs/reports/milestone8_personalization_summary.csv"),
        "--out-csv",
    ),
) -> None:
    """Collect summary.csv files under root and write a merged JSON/CSV report."""
    sections: dict[str, Any] = {"root": str(root), "sections": {}}
    flat_rows: list[dict[str, Any]] = []

    patterns = [
        ("hyperparams", "**/sweeps/hyperparams/summary.csv"),
        ("data_size", "**/sweeps/data_size/summary.csv"),
        ("holdout_params", "**/holdout_validation/params/summary.csv"),
        ("holdout_data_size", "**/holdout_validation/data_size/*/summary.csv"),
    ]

    for name, pattern in patterns:
        matches = sorted(root.glob(pattern))
        section_rows: list[dict[str, Any]] = []
        for match in matches:
            rows = _read_summary_csv(match)
            for row in rows:
                enriched = {"section": name, "summary_path": str(match), **row}
                section_rows.append(enriched)
                flat_rows.append(enriched)
        sections["sections"][name] = {
            "n_files": len(matches),
            "n_rows": len(section_rows),
            "files": [str(m) for m in matches],
            "rows": section_rows,
        }

    # Best recipes if present.
    recipes: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/best_recipe.json")):
        with path.open(encoding="utf-8") as f:
            recipes.append({"path": str(path), "recipe": json.load(f)})
    sections["best_recipes"] = recipes

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)
    typer.echo(f"Wrote {out}")

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if flat_rows:
            pl.DataFrame(flat_rows).write_csv(out_csv)
        else:
            pl.DataFrame({"note": ["no summary rows found"]}).write_csv(out_csv)
        typer.echo(f"Wrote {out_csv}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    app()
