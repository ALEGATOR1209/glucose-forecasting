# Legacy scripts API

The `scripts` package is a deprecated compatibility layer scheduled for removal
in the next major release. New code should import from `glucose_forecasting`
and use the `glucose` command.

## Supported during the transition

- Existing console commands registered in `pyproject.toml`
- Direct `scripts.*` imports used by existing training, evaluation, tuning, and
  personalization code
- Re-export modules under `scripts/common/`, model directories, and legacy
  trainer modules

These paths retain their current behavior so existing checkpoints, notebooks,
and automation continue to work. Compatibility shims may delegate to
`glucose_forecasting`, but their public names are preserved.

## Migration target

| Legacy use | Replacement |
|---|---|
| `scripts.common.*` | `glucose_forecasting.common.*` |
| `scripts.glumind.glumind_model` | `glucose_forecasting.models.glumind` |
| `scripts.sugar_one.sugar_one_model` | `glucose_forecasting.models.sugar_one` |
| `scripts.glumind_uni.glumind_uni_model` | `glucose_forecasting.models.glumind_uni` |
| Legacy data/training imports | `glucose_forecasting.data.*` and `glucose_forecasting.training.*` |
| Legacy evaluation commands | `glucose evaluate` |
| Legacy Hub scripts | `glucose release` |

## Personalization is frozen

`scripts/personalization/` is independently maintained and remains untouched by
this migration. New package code must preserve its existing import contract;
personalization changes require an explicit, separate request.
