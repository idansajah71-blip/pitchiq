# PitchIQ source audit — 29 October 2026

## Verdict

**Source code improved; do not submit the bundled build as-is yet.** The existing `models/` and `reports/` files are pre-audit artifacts. This environment cannot regenerate them because `pyarrow`, `statsbombpy`, and `streamlit` are not installed and package downloads fail without network access. The source changes compile, and the tests that do not require Streamlit pass. Run the complete pipeline and app smoke tests locally before quoting numbers or treating the project as submission-ready.

## Findings and changes

### P0 — Historical test-set use made the old score unsuitable as a final estimate

**Evidence:** the original `models/metadata.json` records `best_raw_by_test_brier: XGBoost` and computes the decision from test AUC/Brier gaps. The old `src/train.py` also selected calibration by checking the test-set calibration summary. That means the reported test set helped inform model/calibration choices.

**Changed:** deployment model family is now fixed in advance to Logistic Regression. The test set and post-tuning CV values do not choose the deployed model. Calibration selection uses grouped out-of-fold predictions from the training partition. The holdout split uses a new match-grouped seed (`TEST_SPLIT_SEED = 20261029`) and the app/training/insight split helpers use the same constant. New model metadata is tagged `audit_protocol_version: 2026-10-29-v1`.

**Still required:** the changed seed creates a different grouped split, but it does not make the holdout pristine: the same dataset and earlier evaluation were already examined during development. Treat refreshed metrics as development evaluation, not a confirmatory benchmark. Lock the code before reading them and do not tune to improve this split. For a stronger independent estimate, evaluate on later matches or a genuinely untouched dataset. Nested CV would also be needed to make a fair model-family claim when selecting between tuned XGBoost/Random Forest and Logistic Regression.

### P1 — Tuned-tree CV scores were not a fair selection signal

**Evidence:** XGBoost and Random Forest hyperparameters were searched on the same `GroupKFold` folds later used to report their CV metrics. Reusing these post-tuning CV results to pick the winner is optimistically biased unless the procedure is nested.

**Changed:** Logistic Regression is the pre-declared, interpretable deployment baseline. Tree models remain comparisons, and their reported CV metrics are labelled exploratory in the UI/docs. No claim is made that PitchIQ outperforms StatsBomb xG.

### P1 — Player leaderboard mixed club/national-team and competition samples

**Evidence:** the old `player_table()` grouped only by player name, then reported the most common team and first competition for an aggregate that could include shots from other teams/competitions. The generated Messi line (217 shots) was one example of the scope mismatch risk.

**Changed:** player metrics are grouped by `player`, `team`, and `competition`. Creator statistics join within the same team/competition scope, and an outer join preserves a key-pass creator even if they did not take a shot in this dataset slice. The UI clarifies that this is key-pass-linked shots, not an official assist total. The leaderboard, markdown summaries, and finishing-chart labels include competition context. Reports must be regenerated for the correction to appear in the app.

### P1 — Some coaching numbers were overconfident or built on hard-coded assumptions

**Evidence:** the old recommendation used a fixed `0.0267` xG uplift per “low-quality” shot, labelled counter-attacks as a tactical causal lever, and called the `110+ / y20–40` band a central corridor even though the opposite-side band `y40–60` is its mirror.

**Changed:** the replacement scenario computes the mean xG of actual attempts below the threshold and clearly labels the result as an assumption-based scenario, not a causal forecast. Counter-attack comparisons are described as observational associations. Zone recommendations combine mirrored lateral bands before comparing value. These reports must be regenerated; the archive's old `reports/coach_recommendations.md` and `reports/insights_summary.md` are not final evidence.

### P1 — Data downloads could silently produce incomplete analyses

**Evidence:** the old match-fetch workflow skipped failures and continued with whatever matches had succeeded. The creator-fetch workflow could write an empty cache after a failed API request, making an outage look like a match with no resolved assists.

**Changed:** shot fetching now fails loudly if any expected match download fails or if a listed match has no shot rows. Creator fetching now raises after its retries rather than caching failure as an empty result. This prioritizes reproducibility over quietly producing partial reports. If the data API rate-limits, rerun with fewer workers (for example `python -m src.data --workers 4`).

### P1 — The shot-zone usage heatmap rendered blank

**Evidence:** `plot_zone_map()` normalized the sparse shot-count pivot with `matrix_n.values.sum()`. Because empty zones are represented by `NaN`, the NumPy sum became `NaN` and every share cell became blank. The existing figure shows an entirely empty “Share of all shots taken” panel.

**Changed:** `zone_shot_share_matrix()` uses a NaN-safe total and has regression tests for sparse and empty grids. The zone figure must be regenerated to show the correction.

### P2 — Displayed metrics could mismatch the actual prediction pipeline

**Evidence:** the evaluation page did not apply an optional saved calibrator before scoring probabilities for its calibration plot; the shot-map average could mix StatsBomb penalty xG with PitchIQ non-penalty xG under a generic label.

**Changed:** the evaluation page now applies the saved calibrator, and the shot map explicitly labels the penalty-inclusive metric as a mixed-source display average. Permutation importance also uses the deployed probability path, including the optional calibrator.

### P2 — Some coach-facing descriptions overreached the observational data

**Evidence:** a few factor descriptions and recommendations said pressure cuts shot quality by a fixed amount, or that an extra pass is usually better, without a causal design. The long-shot threshold was also named `LONG_SHOT_METERS` even though the source coordinates are described elsewhere as pitch-coordinate units.

**Changed:** these descriptions now frame pressure and play-pattern differences as observational associations; long-shot threshold names/text say pitch-coordinate units, and finisher recommendations include team/competition scope. The scenario and recommendations still need regeneration to update the bundled report files.

### P2 — A README figure caption overstated what the static graphic showed

**Evidence:** `fig1_shot_map.png` colours points by outcome (goal/no goal), while its README caption said the static graphic was coloured/sized by PitchIQ xG.

**Changed:** the caption now correctly distinguishes the static exploratory figure from the interactive app map, which uses xG for size/colour.

### P2 — The estimator's x-coordinate help text was inaccurate

**Evidence:** the slider label said `x: 0 = midfield`, but StatsBomb pitch coordinates place the halfway line around `x = 60` on a 120-unit pitch.

**Changed:** the label now states `x = 60` for halfway and `x = 120` for the opponent's goal. The estimator smoke test now checks that moving the slider updates the displayed distance rather than only checking for an exception.

### P2 — Submission copy and live-demo state were misleading or stale

**Evidence:** README said a live demo was deployed but supplied no URL; Devpost copy had stale metrics and unsupported tactical uplift claims; README mentioned “shot-clock tendencies” as a football-rule drift example.

**Changed:** README now states that no verified deployment URL is bundled, removes that unrelated reference, and warns against quoting old outputs. `DEVPOST.md` is rewritten as a submission draft without stale numerical claims. `PITCH_SCRIPT.md` avoids repeating stale model metrics and lists the pre-submit checks. The app shows a warning while it detects the pre-audit model metadata.

## Checks performed

- `python -m compileall -q src app tests` — passed after the audit updates.
- `python -m pytest tests/ -q` — **19 passed, 1 skipped**. The Streamlit smoke-test module was skipped because `streamlit.testing.v1` is unavailable in this environment.
- A real app smoke test still requires installing `requirements.txt`; it has not run in this environment.
- Full data/training/report pipeline — **not executed**: `pyarrow`, `statsbombpy`, and `streamlit` are missing, and package installation could not reach the package index. Existing model/report artifacts were deliberately not represented as refreshed.

## Required before the Devpost submission

1. On the development machine, create/activate a virtual environment and install `requirements.txt`.
2. Run `python -m src.data --workers 4` first; resolve every fetch error rather than accepting a partial dataset. Then run `python -m src.features`, `python -m src.eda`, `python -m src.train`, `python -m src.creators`, and `python -m src.insights` (or `./scripts/run_pipeline.ps1` from PowerShell after setting the worker count to 4 in the first command).
3. Run `python -m pytest tests/ -q`. All 22 tests, including both Streamlit smoke tests, must pass before treating the app as tested.
4. Restart Streamlit. Confirm the pre-audit warning disappears, all pages render, the evaluation and recommendations match regenerated files, the shot-zone usage panel is populated, and penalty display is clearly labelled.
5. Inspect `models/metadata.json` for `audit_protocol_version: 2026-10-29-v1` and `split.test_split_seed: 20261029`. Refresh the Devpost text using only regenerated numbers. Deploy the app, test its public URL in a private/incognito session, and replace the live-demo placeholder.
6. Publish the audited source to a public GitHub repository and include that repository URL and the working demo URL in Devpost. Check all hackathon eligibility rules separately before submitting.

## Files changed in this audit

- `src/train.py`, `src/config.py`, `src/data.py`, `src/creators.py`
- `src/insights.py`, `app/streamlit_app.py`
- `tests/test_train_selection.py`, `tests/test_insights_audit.py`, `tests/test_app_smoke.py`
- `README.md`, `DEVPOST.md`, `PITCH_SCRIPT.md`, `PRE_SUBMISSION_CHECKLIST.md`

**Important:** the packaged parquet data and existing report/model files remain the original bundled data/artifacts because this environment could not regenerate them. The code now warns about that stale state; follow the required steps above before submitting.
