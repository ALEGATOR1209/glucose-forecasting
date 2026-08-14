# JEPA and our glucose pipeline — a getting-started guide

## Why predicting glucose matters (and not only for diabetics)

Continuous glucose monitors (CGMs) record blood sugar every few minutes, and forecasting where glucose is heading — or how it will react to a meal, an insulin dose, or a workout — is valuable well beyond one group of people. For people with diabetes, especially insulin users, short-term forecasts drive hypo- and hyperglycemia alarms and better dosing decisions, and they are the core of automated insulin-delivery ("closed-loop") systems; a model that answers "what happens if I eat or inject this?" is direct decision support. For healthy and longevity-minded people, glucose stability and the size of post-meal spikes are markers of metabolic health, and knowing your own response to particular foods or activities helps tune diet and lifestyle — and can reveal early insulin resistance long before any clinical diagnosis. That broader payoff — better glucose forecasts and better "what-if" predictions for everyone — is what this project ultimately serves.

## What this is about, in one paragraph

We forecast blood glucose from continuous glucose monitor (CGM) data — a regular time series, one reading every 5 minutes. We already have working forecasting models (they take the recent history and predict the next hour). Separately, there's a research model called **CGM-JEPA** that learns to *represent* glucose data without needing any labels. The project is to see whether CGM-JEPA's way of representing glucose can make our forecasts better, and to do it in careful, checkable steps. Think of it as: you already have a classifier that works on spectrograms; now someone hands you a self-supervised feature extractor and asks whether its features help.

## Is this realistic for your team and timeline?

Yes, if you aim at the right rung. Pretraining a self-supervised model from scratch is *not* a 1.5-week job, so treat that as out of scope. What is very achievable, and where we'd like you to start, is the cheapest idea: turn each day of glucose into an "image" (explained below), and use it as an extra feature — or even as a standalone input to a small CNN, which is exactly the kind of model you already built for audio. That single connection (glucose day → image → CNN) is a complete, self-contained project on its own, and everything else here is optional upside. The guide lays out the fuller vision too, but you should scope down to the first achievable win and treat the rest as stretch goals.

## You are not on your own — the team can help

This is a real project with people behind it, not a solo exam. Alex Karmazin and other team members are available to help, and can even take on small pieces of work to unblock you — for example clarifying the data format, helping you get an environment running, pointing you to the right dataset, or arranging access to restricted data. If you get stuck for more than a short while on setup or plumbing, ask rather than burning days on it; your limited time is better spent on the actual experiment. In particular, if you decide you need the access-restricted AI-READI dataset, that's a case to raise with Alex and the team early.

## Working with an AI coding assistant (e.g. Claude Code)

This guide is written to be useful *to the assistant* as much as to you, so it's worth feeding it in as context. A few things are specific to this project and easy for an assistant to get wrong unless told:

- **Point it at the READMEs and `AGENTS.md` first.** The two repos use different tooling — our repos use `uv` and must **never** use `uv pip install` (only `uv sync` / `uv add`), while CGM-JEPA uses plain `pip`. An assistant that doesn't read this will break the environment.
- **Use each step's check as an acceptance criterion**, and have it surface the actual output (a printed shape, the plotted image, the metric) — the failure mode to guard against is code that runs but is subtly wrong, which the per-step checks are designed to catch.
- **The adapter is the main piece of glue to delegate** (read our standard table → cut day-long windows → call JEPA's functions); just make sure it uses our `StandardFieldNames` rather than inventing its own.

## Background you'll need (short, with analogies)

**Our task is forecasting, i.e. regression over time.** Given the last few hours of glucose (plus, in some models, insulin and carbs), predict the next 12 points = 60 minutes. This is like next-step prediction on an audio signal, except the "labels" are just the future values of the same series, so no manual labelling is needed.

**Embeddings / representations.** When your audio CNN processes a clip, the vector it produces just before the final classification layer is a compact summary of the input — that's an *embedding*. Most of this guide is about a model that produces good embeddings of glucose windows, and whether feeding those embeddings into our forecaster helps.

**Self-supervised learning (SSL) and JEPA.** SSL means learning useful representations without labels, by inventing a task out of the data itself. The most famous example is masked language modelling (hide some words, predict them) — you'll meet it when you get to LLMs. CGM-JEPA does the masked-prediction idea on glucose: hide part of the signal and predict the hidden part. The twist that the name "JEPA" (Joint-Embedding Predictive Architecture) refers to is that it predicts the *embedding* of the hidden part, not the raw glucose numbers. The reasoning: reconstructing exact values forces the model to memorise surface detail, whereas predicting the representation pushes it to capture the higher-level "shape" of glucose dynamics, which tends to transfer better across devices and patients. A practical consequence you must remember: **CGM-JEPA does not output glucose values.** To get an actual prediction you have to add and train a small output layer on top.

**Glucodensity — the "image" of a day (this is the part that fits your background best).** Just as a spectrogram turns audio into an image that a CNN can eat, glucodensity turns one day of glucose into an image. It's built entirely from the glucose signal: smooth the day's curve, compute its speed (first derivative) and acceleration (second derivative), then make three smoothed 2-D histograms — glucose-vs-speed, glucose-vs-acceleration, and speed-vs-acceleration. Stacked together these form a 3-channel image (like RGB) that describes how the day's glucose *behaves*, not just its values. Because it's an image, you can hand it straight to a CNN.

## The three repositories and how they relate

Our work lives in two repositories, and the research model in a third:

- **`glucose_data_processing`** (https://github.com/GlucoseDAO/glucose_data_processing) takes raw data from many different CGM devices and studies and converts it all into one standard, clean table. This is where the data comes from.
- **`glucose-forecasting`** (https://github.com/GlucoseDAO/glucose-forecasting) contains our forecasting models — `GluMind` (uses glucose + heart rate + steps) and `SugarOne` (uses glucose + insulin + carbs) — plus some standard baselines.
- **`CGM-JEPA`** (https://github.com/cruiseresearchgroup/CGM-JEPA, branch `master`) is the external research model that produces glucodensity images and self-supervised embeddings.

The two papers behind them are GluMind (https://arxiv.org/abs/2509.18457, our `SugarOne` is based on it) and CGM-JEPA (https://arxiv.org/abs/2605.00933).

## How to run things (the two repos use different tools — don't mix them)

Our two repositories use **`uv`**, a fast Python environment manager. You set up once with `uv sync`, then run commands through it:

```bash
uv sync                                    # install everything
uv run glucose-process <input_folder> -o <output>          # preprocessing
uv run python scripts/sugar_one/train_sugar_one.py --help  # a forecasting model
uv run evaluate-model --help
```

One firm rule from our `AGENTS.md`: **never use `uv pip install`** — only `uv sync` or `uv add`. The external CGM-JEPA repo, by contrast, follows its own README and uses ordinary `pip install -r requirements.txt` with `python -m ...`. Keep the two worlds separate.

## The words we use (please reuse them, don't invent your own)

The GluMind paper calls the thing we predict the **blood glucose level (BGL)**, and it calls the extra input signals (heart rate, steps, activity, stress) **physiological and behavioral variables**, or *predictive features*. In our code these extra signals are called **covariates**, with fixed names: `glucose`, `hr`, `steps` for GluMind, and `glucose`, `basal`, `bolus`, `carbs` for SugarOne. If you read about "known-future covariates" that is terminology from a different library (NeuralForecast/TFT), not GluMind — in our models the covariates currently come only from the past window.

## Data: what we have and which set to use

Our catalog of datasets is in `docs/datasets.csv` — around fifty CGM datasets ranging from single-subject studies to a thousand participants. The important practical points:

The large, ready-to-use table lives in a folder (`DATA/`) that is deliberately not stored in git, so a fresh clone won't contain it; you regenerate it from raw sources with `uv run glucose-process`. For a very quick first test there's a tiny bundled sample in `test_data/dexcom_small/` (3 patients, about two weeks each) — enough to check that things run, but it only yields a few dozen day-long windows, so for a real experiment you'll process a larger set.

For which larger set to use: **start with the Loop dataset** (entry #35 in the catalog). It is public, has no access restrictions, contains a thousand participants with CGM plus insulin-pump data, and is actually larger than the alternative — so it's both convenient and a good fit for `SugarOne`, which uses insulin and carbs. **Do not build your first experiments around AI-READI**: it is not public, has controlled access with restrictions on who may use it. If you do end up needing it (for example for the clinical grouping labels discussed later), raise it early with Alex Karmazin and the team, who can help arrange access — don't let it block you in the meantime. Many other public sets exist too, but Loop is the natural default.

## Use our clean data format, not JEPA's rough loaders

One thing that will save you pain: we already standardise every dataset into one clean schema (the field names are defined in `processing/core/fields.py` as `StandardFieldNames`, and the per-device converters live in `formats/`). CGM-JEPA's own data-loading code is lower-level and makes its own assumptions (a configurable glucose column, manual padding, a hard-coded glucose range and a value-to-token conversion). Those are their conventions and they can be messy.

So the guiding principle is: **reuse JEPA's math, not its input/output code.** Write a small adapter that reads our standard table, cuts it into day-long (288-point) glucose windows — carefully, never crossing a data gap or mixing two patients in one window — and hands JEPA just the plain array of numbers it needs. Then call JEPA's glucodensity function (or its encoder) on that array. In effect you reduce JEPA to a simple function: *glucose window in → image (or embedding) out*. This matters most when you generate the images in our pipeline (Step 2 below) and when you feed embeddings back to our models (Step 4).

## What you could actually build (from cheapest to most ambitious)

There are three ways this can help our models, and you should think of them as a ladder:

The cheapest is to compute the glucodensity images and use them as extra features — or, most naturally for you, feed them to a small CNN and see whether they predict anything useful. No pretraining and no labels are involved, so this is the realistic target for your timeline.

More ambitious is to pretrain the JEPA encoder on our unlabelled glucose and then attach a small output layer to forecast — this is the "proper" self-supervised route but it needs more compute and time than you likely have.

In between, you can take the encoder with its released pretrained weights (no training of your own), extract embeddings, and use them as an extra input to our existing forecasting models.

## The bigger vision (a stretch goal, not for week one)

Ultimately we'd love a model that answers "what happens to my glucose if I inject insulin, eat, or exercise now?" This is harder than plain forecasting because it needs the model to take *planned future actions* as input, which our current models don't do (they only see past insulin/carbs). JEPA would fit here as a "state summariser" feeding such a model. Mention it in your write-up as future work, but don't try to build it in 1.5 weeks — and be aware there's a genuine scientific catch: models trained on observational data learn correlations (people inject *because* glucose is high), so a good error score does not by itself prove the "what-if" answers are causally correct.

## A step-by-step plan, with a check after every step

The golden rule is to move in small steps and verify each one before continuing, because the most common failure mode is that something *looks* like it worked while being subtly wrong.

**Step 0 — just get it running.** Clone CGM-JEPA, install it their way, and run their glucodensity generation on a small CSV:

```bash
python -m utils.precompute_glucodensity \
  --data_path <small.csv> --output_path out/gluco_cache.pt \
  --patch_size 12 --series_split_size 288 \
  --gluco_spatial_patch_size 8 --gluco_gridsize 32
```

The only goal here is that the environment works and the script finishes without errors. (For reference, a day is 288 readings at 5-minute spacing.)

**Step 1 — look at the images and confirm they're sensible.** Load a few of the generated samples and display the three channels with `matplotlib`. Check the obvious things: the array has the expected shape, the values sit between 0 and 1, there are no missing (`NaN`) values, different days produce visibly different images, and each image looks like a smooth density blob rather than random noise. The point of this step is that *you personally trust* the images before building anything on them.

**Step 2 — generate the images inside our pipeline, for about 2000 samples.** Because we have far more data than the research set, generate the images on our side rather than theirs: read our standard table, cut ~2000 day-long windows, and call JEPA's `compute_glucodensity_patches_from_cgm` on each. Save them in the same format the research code expects. This step is only about generating and saving — no model training yet.

**Step 3 — re-run the same checks on our output.** Repeat the Step-1 checks on the images you produced, and additionally confirm the format and shapes match the reference from Step 0 (so the research code would accept them unchanged), and that each window belongs to a single patient and doesn't span a gap in the data. The goal is that your images are interchangeable with theirs.

**Step 4 — only now bring in the encoder, and do this in small sub-steps too:**

First, run the pretrained encoder on a handful of our windows and confirm the output embeddings have the expected shape, contain no `NaN`, and differ from window to window (checking embedding quality properly is covered in the next section). Then decide to keep the encoder *frozen* (its weights fixed) to start with — it's the simplest option; leave fine-tuning for later. Next, make the smallest possible change to attach the embedding as an extra input to one of our models, and confirm it trains for a few steps without shape errors. Then run a small, fair comparison — the same data, the same random seed — of the model with and without the embedding, and compare the error metrics (MAE, RMSE, and MARD, the last being a glucose-specific percentage error). Finally, only scale up (more data, tuning) if that small comparison actually showed an improvement.

## How to check whether the embeddings are any good

An embedding that runs without crashing can still be useless, so it's worth knowing how to judge quality. There are three levels, and you should do the cheap ones first.

**Basic health checks (no labels needed).** Confirm the shape and absence of `NaN`s, but the most important check is for *representation collapse* — a well-known failure of self-supervised models where the encoder cheats by outputting almost the same vector for every input. You detect it by checking that each embedding dimension actually varies across samples and that different windows aren't all nearly identical (e.g. via cosine similarity). Reassuringly, JEPA's training loop already tracks the variance of its representations for exactly this reason. It's also worth reducing the embeddings to two dimensions with PCA (a standard way to compress many numbers into a few) and plotting them coloured by something simple like average glucose, just to see whether the structure makes sense.

**Reuse JEPA's own quality tools.** The research repo already includes an evaluation file (`eval/class_reg.py`) that measures embedding quality with clustering scores (silhouette, and agreement-with-labels scores called ARI and NMI) and a simple "linear probe" (train a plain logistic-regression classifier on the frozen embeddings and see how well it does). The catch is that all of these need *labels* to cluster around, which brings us to the next point.

**What to cluster the embeddings around, since we lack the paper's labels.** The paper measured quality against clinical labels (types of metabolic dysfunction) that we don't have, so you'll substitute something we do have — but *what* you choose changes what the result means. The strongest choice is a real clinical grouping: AI-READI provides a `study_group` field (healthy / prediabetes / type-2) and age, which is the closest match to the paper — but remember AI-READI is access-restricted, and Loop can't supply this because it's a single-disease-type cohort with no such label. A weaker but always-available choice is a quantity derived from glucose itself (time-in-range, variability, average) — useful as a sanity check, but slightly circular, because an embedding built from glucose should trivially recover these, so success here only proves the embedding didn't collapse. A third choice is behaviour-related covariates from Loop (total daily insulin, how often boluses happen, carb intake); since the encoder only ever saw glucose, weak agreement here is expected and actually informative about what the embedding does and doesn't capture. Two cautions: these clinical labels are one-per-patient while embeddings are one-per-window, so if you spread a patient's label across all their windows you must split train/test *by patient* to avoid cheating; and remember Loop and AI-READI describe different populations. For a public, unblocked start, use the Loop-based and glucose-derived groupings now, and treat the AI-READI clinical grouping as something to request from the team later.

**The check that really matters for us** is simpler than all of the above: does adding the embedding improve the forecast? As a quick preview you can train a tiny model from the embedding to something you already have (like the next glucose value); if even that shows no signal, the full model is unlikely to benefit.

## A note on embedding size

The JEPA embedding can be fairly large (e.g. 128 numbers), while our forecasting models are small internally (their hidden size is 32). If you simply glue a 128-long vector onto a 32-wide model, the embedding can swamp the model's own signal. So treat the size as something to experiment with rather than accept: try shrinking the embedding with PCA to, say, 8, 16, or 32 numbers, or let the model learn its own small projection of it; decide how to pool the sequence of patch-embeddings into one vector (averaging is the simple default); and always rescale the embedding to a similar numeric range as the other inputs so one doesn't dominate. Change one thing at a time and watch the validation error, since a bigger input on a small dataset overfits easily.

## How to actually feed the embedding into our models (the fiddly part)

This is genuinely the trickiest piece, because of a shape mismatch. Our models expect a *sequence over time* — one value per 5-minute step for each channel — whereas a JEPA embedding is a single summary vector for the whole window. On top of that, our data pipeline currently hands the model a `(window, target)` pair, so you'll first need to carry the embedding along as a third item through the data loader and into the model's forward pass. Given that, here are the options from simplest to most faithful:

The simplest is to attach the embedding at the very end: our model already compresses the whole window into one 32-number vector just before its final layer, so you concatenate the (shrunk, rescaled) embedding there and widen that final layer accordingly. This treats the embedding as overall context for the prediction and is the least invasive change — start here.

A slightly more elegant option, called FiLM, is to let the embedding produce a small scale-and-shift that modulates the model's internal features, which injects the "state" information without widening the input at all.

The most faithful option, but the most work, uses the fact that attention (which you're learning now) lets one sequence attend to another: you feed JEPA's sequence of patch-embeddings as an extra stream that the glucose attends to, mirroring how SugarOne already lets glucose attend to insulin and carbs.

For the standard baseline models (NHITS, TFT, NBEATSx) the story is easier, because those libraries have explicit slots for extra input variables; there you'd add the (dimension-reduced) embedding as extra columns, held constant across the window. Whichever route you take, reduce and rescale the embedding first, and change one thing at a time with a fixed data split so any change in error is clearly attributable to the embedding.

## If you run out of time: the work still pays off

Wiring the embedding into our forecasters is the fiddliest part, so if it doesn't fit your timeline, the embeddings are still valuable in ways that don't touch the forecaster at all — and these make a perfectly good project on their own.

Using the embeddings by themselves, you can cluster patients or days to discover natural glucose "types" and see how our many datasets relate; you can spot data-quality problems (unusual days and sensor glitches show up as outliers, and near-duplicate windows across datasets warn you of leakage); and you can do similarity search ("find days like this one").

Using the embeddings alongside an existing model's predictions — without changing that model — you can analyse *where* it makes its biggest errors by seeing which embedding clusters the errors fall into; you can train a small, separate correction model that takes the base prediction plus the embedding and nudges the prediction (a light-touch way to get a benefit without surgery on our models); and you can flag predictions the model is likely unsure about, because windows unlike anything in training stand out in embedding space — which is exactly the kind of safety signal that matters in health data.

In short, even if the deepest integration doesn't happen, there's a ladder of useful outcomes, and the lower rungs are both easier and genuinely worthwhile.

## Why this is worth doing

It's a reasonable bet for a few concrete reasons. We have a great deal of *unlabelled* glucose data and comparatively little labelled data, and self-supervision is precisely the tool that turns the abundant, cheap resource into something useful. Our real, recurring headache is that models trained on one device or patient group often generalise poorly to another, and the CGM-JEPA paper's central claim is that its representations stay more consistent across exactly those shifts. The glucodensity view adds information about glucose *dynamics* that our current inputs may miss, and it costs almost nothing because it's computed from glucose we already have. And a single pretrained encoder could serve several of our future tasks at once. The honest caveat is that the paper only demonstrated these benefits for a classification task, not for forecasting — so this is a promising, cheap-to-test idea rather than a sure thing, which is exactly why the plan is staged with a check at every step.

## What to read first (about 30 minutes)

On the CGM-JEPA side (branch `master`), skim `models/predictor.py` and `pretrain/pretrain_cgm_jepa.py` to see for yourself that the model predicts embeddings rather than glucose values, read `utils/glucodensity_utils.py` to see how the images are built, and glance at `eval/class_reg.py` for the embedding-quality measures. On our side (branch `main`), read `scripts/glumind/glumind_model.py` and `scripts/sugar_one/sugar_one_model.py` to understand the model inputs and outputs, look at `scripts/sugar_one/evaluate_model.py` for the covariate names, and open `docs/datasets.csv` in `glucose_data_processing` to see the available datasets. Both pipelines work on regular 5-minute glucose, so the formats are compatible.

Links:

- CGM-JEPA glucodensity: https://github.com/cruiseresearchgroup/CGM-JEPA/blob/master/utils/glucodensity_utils.py
- CGM-JEPA predictor: https://github.com/cruiseresearchgroup/CGM-JEPA/blob/master/models/predictor.py
- CGM-JEPA pretraining: https://github.com/cruiseresearchgroup/CGM-JEPA/blob/master/pretrain/pretrain_cgm_jepa.py
- CGM-JEPA evaluation: https://github.com/cruiseresearchgroup/CGM-JEPA/blob/master/eval/class_reg.py
- GluMind model: https://github.com/GlucoseDAO/glucose-forecasting/blob/main/scripts/glumind/glumind_model.py
- SugarOne model: https://github.com/GlucoseDAO/glucose-forecasting/blob/main/scripts/sugar_one/sugar_one_model.py
- Covariates in code: https://github.com/GlucoseDAO/glucose-forecasting/blob/main/scripts/sugar_one/evaluate_model.py
- Dataset catalog: https://github.com/GlucoseDAO/glucose_data_processing/blob/main/docs/datasets.csv
