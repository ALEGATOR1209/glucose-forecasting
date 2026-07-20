# Load pretrained weights and run on a CSV

## What you need

From the repository root:

```bash
uv sync
```

| Path | What it is |
|------|------------|
| `test_model_sugar_one/` | SugarOne weights (glucose + basal/bolus/carbs) |
| `test_model_glumind/` | GluMind weights (glucose + HR + steps) |
| `test_data/livia_sugar_one_ready.csv` | Small demo CSV with pump columns |
| `test_data/livia_glumind_ready.csv` | Small demo CSV in GluMind shape |

For real benchmarks, prepare data with [glucose_data_processing](https://github.com/GlucoseDAO/glucose_data_processing) and put CSVs in `data/input/` — see [docs/DATA.md](docs/DATA.md).

`--run-dir` = folder with `best_model.pt` and `tuning_meta.json` / `config.json`.  
`--train-csv` = file used to fit scalers (use your full training CSV when you have it; for the demo, use the same file as `--test-csv`).

## SugarOne on your data

```bash
uv run evaluate-model \
  --run-dir test_model_sugar_one \
  --model-type sugar_one \
  --test-csv data/input/loop_ai_ready_joined2.csv \
  --train-csv data/input/loop_ai_ready_joined2.csv \
  --batch-size 256
```

## SugarOne on the demo CSV

The demo has no train/val/test labels, so pass `--test-split ''`:

```bash
uv run evaluate-model \
  --run-dir test_model_sugar_one \
  --model-type sugar_one \
  --test-csv test_data/livia_sugar_one_ready.csv \
  --train-csv test_data/livia_sugar_one_ready.csv \
  --test-split '' \
  --batch-size 256
```

Glucose-only ablation (ignore pump covariates):

```bash
uv run evaluate-model \
  --run-dir test_model_sugar_one \
  --model-type sugar_one \
  --test-csv test_data/livia_sugar_one_ready.csv \
  --train-csv test_data/livia_sugar_one_ready.csv \
  --zero-cov \
  --test-split '' \
  --batch-size 256
```

## GluMind

```bash
uv run evaluate-model \
  --run-dir test_model_glumind \
  --model-type glumind \
  --test-csv test_data/livia_glumind_ready.csv \
  --train-csv test_data/livia_glumind_ready.csv \
  --test-split '' \
  --batch-size 4096
```

## Download GluMind weights from Hugging Face (optional)

Only if `test_model_glumind/` is empty:

```bash
uv run download-glumind-hf \
  --repo-id GlucoseDao/glumind-global-h12 \
  --output-dir test_model_glumind
```

Then run the GluMind command above.

Livia is personal type‑1 data — use those numbers only to confirm the pipeline runs, not as a headline score.
