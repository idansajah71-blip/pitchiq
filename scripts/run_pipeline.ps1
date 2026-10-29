# Reproduce the whole PitchIQ pipeline end-to-end, in order.
# Usage (from the project root):  ./scripts/run_pipeline.ps1

$steps = @(
    @{ Name = "data      (download + cache shots)"; Command = "python -m src.data --workers 4" },
    @{ Name = "features  (engineered feature table)"; Command = "python -m src.features" },
    @{ Name = "eda       (figures + tables)"; Command = "python -m src.eda" },
    @{ Name = "train     (LogReg + XGBoost + RF + calibration)"; Command = "python -m src.train" },
    @{ Name = "creators  (assist linkages)"; Command = "python -m src.creators --workers 4" },
    @{ Name = "insights  (SHAP + players + teams + renders)"; Command = "python -m src.insights" },
    @{ Name = "tests     (unit + app smoke)"; Command = "python -m pytest tests/ -q" }
)

foreach ($step in $steps) {
    Write-Host ">> $($step.Name)"
    Invoke-Expression $step.Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED at: $($step.Name)"
        exit 1
    }
}
Write-Host "All pipeline steps finished."
