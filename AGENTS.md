# Agent conventions

## CLI naming

For new command-line interfaces, avoid long hyphenated executable names. Prefer a concise root Typer command with noun/verb subcommands:

```text
glucose forecast
glucose train
glucose models
glucose train --model sugarone
glucose train --backend neuralforecast --eval holdout
```

Use kebab-case only for multi-word option names when needed. When a command selects an
ML evaluation protocol, use precise terms such as `--eval holdout` or `--eval
cross-val`; reserve “workflow” for multi-step orchestration. Preserve existing command
names and behavior for compatibility.

## Personalization boundary

Do not edit `scripts/personalization/` unless the user explicitly requests personalization work. It is maintained independently; consume its existing public imports from new package code rather than moving or rewriting it.

## Testing conventions

Prefer a few behavioral tests that exercise real train/eval/CLI paths over piles of micro-unit files. When adding coverage:

- **No thin tests.** Do not add tests whose only job is to assert dict key names, log-string formatting, `is` identity of re-exports, env-var clears, or other one-line helper trivia. If a smoke/CLI test already creates the artifact, assert the unique invariant there instead of a parallel unit file.
- **Parametrize model variants.** GluMind / SugarOne / GluMind-Uni (and similar) almost always differ by a few fields (`n_features`, CSV writer, run-dir glob, checkpoint `config_key`). Put those in a table/`@pytest.mark.parametrize` and share one test body. Do not copy an entire test file per model name. Existing pattern: `tests/test_train_cli_smoke.py`, `tests/test_models_forward.py`, `tests/test_datasets.py`.
- **Construct repeated strings.** Shared dims, flag maps, artifact names, and globs belong in constants or helpers (`TINY_*` in `tests/conftest.py`, flag builders, `f"{name}_global_*"`). Do not paste the same `"8"`, `"best_model.pt"`, or nearly identical argv lists in every test.
- **Extend smokes before inventing suites.** Prefer adding a small assert to an existing train/eval/holdout smoke over a new `test_*_common.py` that reimplements save/load in isolation.
- **Keep real edge cases.** Parametrized math edges, CSV remaps, covariate conflicts, and protocol geometry are fine when they catch bugs smokes miss — just keep them compact.

## Device selection

**Never hardcode `--device cpu` or `device="cpu"` in commands, scripts, or tests unless the user explicitly asks for CPU-only execution.** Always use `device="auto"` (the default), which detects CUDA → MPS → CPU automatically. This applies to:

- CLI invocations (`glucose evaluate`, `glucose train`, `evaluate-model`, training scripts)
- Test commands and smoke tests
- Code examples in documentation

If a user has a GPU available, forcing CPU wastes their hardware and time. The `auto` default exists precisely to avoid this.
