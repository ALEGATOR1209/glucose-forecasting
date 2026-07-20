# GluMind Command Examples

## Notes

- Prefer `uv run train-glumind ...` or `uv run python scripts/glumind/train_glumind.py ...`.
- `classic` split: uses train/val/test as provided.
- `trainval_test_as_val` split: train <- train+val, val <- test, test disabled.
- Replace output paths as needed for your experiments.
- Examples may use the historical `data/actual/with_complex_steps_processing/` layout. Prefer `data/input/` CSVs from [glucose_data_processing](https://github.com/GlucoseDAO/glucose_data_processing); see [docs/DATA.md](../../docs/DATA.md).

## 1) Smoke Tests

### Global smoke test (tiny run)

```bash
uv run python scripts/glumind/train_glumind.py \
  --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
  --mode global \
  --epochs 1 \
  --patience 0 \
  --max_train_series 12 \
  --max_eval_series 12 \
  --device cuda \
  --batch_size 1024 \
  --precision bf16 \
  --compile_mode none \
  --num_workers 0 \
  --val_every_n_epochs 1 \
  --ckpt_every_n_epochs 0 \
  --log_every 1 \
  --seed 42 \
  --out_dir data/output/runs/_tmp_glumind_smoke/global
```

### Continual smoke test (2 groups)

```bash
uv run python scripts/glumind/train_glumind.py \
  --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
  --mode continual \
  --study_groups Healthy,Pre-T2DM \
  --lwf_lambda 0.2 \
  --epochs 1 \
  --patience 0 \
  --max_train_series 12 \
  --max_eval_series 12 \
  --device cuda \
  --batch_size 1024 \
  --precision bf16 \
  --compile_mode none \
  --num_workers 0 \
  --val_every_n_epochs 1 \
  --ckpt_every_n_epochs 0 \
  --log_every 1 \
  --seed 42 \
  --out_dir data/output/runs/_tmp_glumind_smoke/continual
```

## 2) AI-READI (classic split)

### Global

```bash
uv run python scripts/glumind/train_glumind.py \
  --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
  --split_scheme classic \
  --mode global \
  --epochs 120 \
  --patience 20 \
  --device cuda \
  --batch_size 4096 \
  --precision bf16 \
  --compile_mode reduce-overhead \
  --num_workers -1 \
  --prefetch_factor 4 \
  --val_every_n_epochs 2 \
  --ckpt_every_n_epochs 10 \
  --log_every 1 \
  --seed 42 \
  --out_dir data/output/runs/glumind/ai_ready
```

### Continual

```bash
uv run python scripts/glumind/train_glumind.py \
  --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
  --split_scheme classic \
  --mode continual \
  --lwf_lambda 0.2 \
  --lr 5e-4 \
  --epochs 80 \
  --patience 6 \
  --device cuda \
  --batch_size 2048 \
  --precision bf16 \
  --compile_mode reduce-overhead \
  --num_workers -1 \
  --prefetch_factor 4 \
  --val_every_n_epochs 2 \
  --ckpt_every_n_epochs 10 \
  --log_every 1 \
  --seed 42 \
  --out_dir data/output/runs/glumind/ai_ready
```

## 3) AI-READI Tuning Mode (train+val -> train, test -> val)

### Global

```bash
uv run python scripts/glumind/train_glumind.py \
  --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
  --split_scheme trainval_test_as_val \
  --mode global \
  --epochs 120 \
  --patience 20 \
  --device cuda \
  --batch_size 4096 \
  --precision bf16 \
  --compile_mode reduce-overhead \
  --num_workers -1 \
  --prefetch_factor 4 \
  --val_every_n_epochs 2 \
  --ckpt_every_n_epochs 10 \
  --log_every 1 \
  --seed 42 \
  --out_dir data/output/runs/glumind/ai_ready
```

### Continual (`all_groups` validation)

```bash
uv run python scripts/glumind/train_glumind.py \
  --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
  --split_scheme trainval_test_as_val \
  --mode continual \
  --continual_val_scope all_groups \
  --lwf_lambda 0.3 \
  --lr 1e-3 \
  --epochs 80 \
  --patience 10 \
  --device cuda \
  --batch_size 2048 \
  --precision bf16 \
  --compile_mode reduce-overhead \
  --num_workers -1 \
  --prefetch_factor 4 \
  --val_every_n_epochs 5 \
  --ckpt_every_n_epochs 10 \
  --log_every 1 \
  --seed 42 \
  --out_dir data/output/runs/glumind/ai_ready
```

## 4) AI-READI + Type1 Combined

### Global (classic, held-out test enabled)

```bash
uv run python scripts/glumind/train_glumind.py \
  --csv data/actual/with_complex_steps_processing/ai_ready_plus_type1_v1_val_in_val_and_test.csv \
  --split_scheme classic \
  --mode global \
  --epochs 120 \
  --patience 20 \
  --device cuda \
  --batch_size 4096 \
  --precision bf16 \
  --compile_mode reduce-overhead \
  --num_workers -1 \
  --prefetch_factor 4 \
  --val_every_n_epochs 2 \
  --ckpt_every_n_epochs 10 \
  --log_every 1 \
  --seed 42 \
  --out_dir data/output/runs/glumind/ai_ready_plus_type1
```

### Continual tuning (`trainval_test_as_val`, reverse/default order optional)

```bash
uv run python scripts/glumind/train_glumind.py \
  --csv data/actual/with_complex_steps_processing/ai_ready_plus_type1_v2_val_only_in_test.csv \
  --split_scheme trainval_test_as_val \
  --mode continual \
  --continual_order default \
  --continual_val_scope all_groups \
  --lwf_lambda 0.3 \
  --lr 7e-4 \
  --epochs 80 \
  --patience 20 \
  --device cuda \
  --batch_size 2048 \
  --precision bf16 \
  --compile_mode reduce-overhead \
  --num_workers -1 \
  --prefetch_factor 4 \
  --val_every_n_epochs 2 \
  --ckpt_every_n_epochs 10 \
  --log_every 1 \
  --seed 42 \
  --out_dir data/output/runs/glumind/ai_ready_plus_type1
```

## 5) Type1-Only Dataset

### Global

```bash
uv run python scripts/glumind/train_glumind.py \
  --csv data/actual/type_1/type1_hupa_uom_glumind_trainval_testmirror.csv \
  --mode global \
  --epochs 120 \
  --patience 20 \
  --device cuda \
  --batch_size 4096 \
  --precision bf16 \
  --compile_mode reduce-overhead \
  --num_workers -1 \
  --prefetch_factor 4 \
  --val_every_n_epochs 2 \
  --ckpt_every_n_epochs 10 \
  --log_every 1 \
  --seed 42 \
  --out_dir data/output/runs/glumind/type1_only
```

## 6) Resume from Checkpoint

```bash
uv run python scripts/glumind/train_glumind.py \
  --csv data/actual/with_complex_steps_processing/ai_ready_processed_dataset.csv \
  --mode global \
  --epochs 250 \
  --resume_from data/output/runs/glumind/glumind_global_h12_<timestamp>/last_checkpoint.pt \
  --ckpt_every_n_epochs 20 \
  --device cuda \
  --batch_size 256 \
  --precision bf16 \
  --num_workers -1 \
  --prefetch_factor 4 \
  --val_every_n_epochs 2
```
