# ShotWorth — pre-submission checklist

The source code in this archive has been audited, but `models/` and `reports/` still contain pre-audit artifacts. **Do not submit the existing numbers or treat the app as final until these steps succeed on a machine with package access.**

## 1. Set up on Windows / PowerShell

Use Python 3.11. From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use a permitted activation method or call `.venv\Scripts\python.exe` directly. Do not work around your machine's execution policy without understanding its security impact.

## 2. Regenerate the model and all reports

```powershell
python -m src.data --workers 4
python -m src.features
python -m src.eda
python -m src.train
python -m src.creators --workers 4
python -m src.insights
```

Stop if data fetching fails. Do not use partial-match outputs as if they were a complete dataset. Re-run with fewer workers or retry after the source is available. After this succeeds, run:

```powershell
python -m pytest tests/ -q
```

There are 22 tests when both Streamlit app smoke tests are available. Require all 22 to pass before claiming the app was tested. The app smoke tests must not be silently skipped in the final environment.

## 3. Check the app and evidence

```powershell
streamlit run app/streamlit_app.py
```

Confirm:

- The orange pre-audit warning is gone.
- All four pages render; changing the estimator location updates displayed distance/probability.
- The shot-zone usage heatmap has percentages, not a blank left panel.
- Player rows are separated by team/competition; creators who did not shoot can still appear in the creator table.
- Penalty xG is labelled as StatsBomb benchmark data, not ShotWorth's non-penalty model.
- `models/metadata.json` contains `audit_protocol_version: 2026-10-29-v1` and `split.test_split_seed: 20261029`.
- Refresh `DEVPOST.md` and `PITCH_SCRIPT.md` only with regenerated metrics. The changed-seed holdout is still a development evaluation because the dataset and earlier results were already seen; do not describe it as a pristine independent benchmark.

## 4. Publish

- Deploy the app and verify the live URL in an incognito/private browser.
- Push the audited source to a **public** GitHub repository.
- Replace the live-demo placeholder in `README.md` and add both the source and demo URL to Devpost.
- Verify the hackathon's eligibility rules before submitting.
