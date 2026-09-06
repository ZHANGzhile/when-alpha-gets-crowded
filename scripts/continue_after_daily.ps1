$ErrorActionPreference = "Stop"
$workspacePath = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "python_runtime.ps1")
$pythonPath = Resolve-AlphaCrowdingPython $workspacePath
$manifestDirectory = Join-Path $workspacePath "data\raw\manifests"
$dailyPidPath = Join-Path $manifestDirectory "daily_download.pid"
$statusPath = Join-Path $manifestDirectory "production_supervisor.json"
$env:PYTHONPATH = "$workspacePath\.python-packages;$workspacePath\src"

function Write-Status([string]$status, [string]$stage, [string]$message) {
    $payload = [ordered]@{
        schema_version = 1
        status = $status
        stage = $stage
        message = $message
        updated_at = [DateTimeOffset]::UtcNow.ToString("o")
    }
    $temporary = "$statusPath.tmp"
    $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

function Invoke-PythonStage([string]$name, [string]$script, [string[]]$arguments) {
    Write-Status "RUNNING" $name "starting $script"
    & $pythonPath (Join-Path $PSScriptRoot $script) @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "stage $name failed with exit code $LASTEXITCODE"
    }
    Write-Status "RUNNING" $name "completed $script"
}

try {
    Write-Status "WAITING" "daily" "waiting for complete universe daily download"
    if (Test-Path -LiteralPath $dailyPidPath) {
        $dailyPid = [int](Get-Content -LiteralPath $dailyPidPath -Raw)
        $dailyProcess = Get-Process -Id $dailyPid -ErrorAction SilentlyContinue
        if ($null -ne $dailyProcess) {
            Wait-Process -Id $dailyPid
        }
    }
    $dailyManifest = Get-Content -LiteralPath (Join-Path $manifestDirectory "daily_download.json") -Raw | ConvertFrom-Json
    if ($dailyManifest.status -ne "COMPLETE") {
        throw "daily download ended with status $($dailyManifest.status)"
    }

    Invoke-PythonStage "daily_acceptance" "15_summarize_daily_history.py" @()
    Invoke-PythonStage "benchmark" "17_download_csi800_benchmark.py" @()
    Invoke-PythonStage "industry" "12_download_monthly_industry.py" @("--workers", "2")
    Invoke-PythonStage "signals" "11_build_weekly_security_signals.py" @()
    Invoke-PythonStage "daily_panel" "18_build_daily_research_panel.py" @("--batch-size", "100")
    Invoke-PythonStage "portfolios" "13_build_factor_memberships.py" @()
    Invoke-PythonStage "convergence" "20_build_strategy_convergence.py" @()
    Invoke-PythonStage "portfolio_risk" "22_build_portfolio_risk_features.py" @()
    Invoke-PythonStage "structural_core" "19_build_structural_core.py" @()
    Invoke-PythonStage "crowding_state" "21_build_crowding_state.py" @()
    Invoke-PythonStage "returns" "16_build_factor_returns.py" @()
    Invoke-PythonStage "market_factor_state" "27_build_market_factor_state.py" @()
    Invoke-PythonStage "factor_return_stress" "23_build_factor_return_stress.py" @()
    Invoke-PythonStage "state_panel" "24_build_state_panel.py" @()
    Invoke-PythonStage "model_feature_panel" "28_build_model_feature_panel.py" @()
    Invoke-PythonStage "dynamic_outcomes" "25_build_dynamic_outcomes.py" @()
    Invoke-PythonStage "fixed_outcomes" "26_build_fixed_outcomes.py" @()
    Invoke-PythonStage "primary_walk_forward" "29_run_primary_walk_forward.py" @()
    Invoke-PythonStage "lead_time_walk_forward" "30_run_lead_time_walk_forward.py" @()
    Invoke-PythonStage "crash_event_study" "31_build_crash_event_study.py" @()
    Invoke-PythonStage "continuous_pseudo_returns" "33_build_continuous_pseudo_returns.py" @()
    Invoke-PythonStage "falsification_diagnostics" "32_run_falsification_diagnostics.py" @()
    Write-Status "COMPLETE" "falsification_diagnostics" "production P5 and P6 diagnostics completed"
}
catch {
    Write-Status "FAILED" "pipeline" $_.Exception.Message
    throw
}
