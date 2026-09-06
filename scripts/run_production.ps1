param(
    [ValidateSet("membership", "daily", "benchmark", "industry", "signals", "daily_panel", "portfolios", "measurements", "returns", "states", "outcomes", "models", "all")]
    [string]$Stage = "all",
    [int]$MembershipWorkers = 4,
    [int]$DailyWorkers = 4,
    [int]$IndustryWorkers = 2
)

$ErrorActionPreference = "Stop"
$workspacePath = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "python_runtime.ps1")
$pythonPath = Resolve-AlphaCrowdingPython $workspacePath
$env:PYTHONPATH = "$workspacePath\.python-packages;$workspacePath\src"

function Invoke-Stage([string]$script, [string[]]$arguments) {
    & $pythonPath (Join-Path $PSScriptRoot $script) @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Production stage failed: $script (exit $LASTEXITCODE)"
    }
}

if ($Stage -in @("membership", "all")) {
    Invoke-Stage "09_download_membership_history.py" @(
        "--start", "2012-01-01", "--end", "2026-08-31",
        "--workers", $MembershipWorkers.ToString()
    )
    Invoke-Stage "14_summarize_membership_history.py" @()
}
if ($Stage -in @("daily", "all")) {
    Invoke-Stage "10_download_universe_daily.py" @("--workers", $DailyWorkers.ToString())
    Invoke-Stage "15_summarize_daily_history.py" @()
}
if ($Stage -in @("benchmark", "all")) {
    Invoke-Stage "17_download_csi800_benchmark.py" @()
}
if ($Stage -in @("industry", "all")) {
    Invoke-Stage "12_download_monthly_industry.py" @("--workers", $IndustryWorkers.ToString())
}
if ($Stage -in @("signals", "all")) {
    Invoke-Stage "11_build_weekly_security_signals.py" @()
}
if ($Stage -in @("daily_panel", "all")) {
    Invoke-Stage "18_build_daily_research_panel.py" @("--batch-size", "100")
}
if ($Stage -in @("portfolios", "all")) {
    # VALUE remains excluded until the PB publication-time audit passes.
    Invoke-Stage "13_build_factor_memberships.py" @()
}
if ($Stage -in @("measurements", "all")) {
    Invoke-Stage "20_build_strategy_convergence.py" @()
    Invoke-Stage "22_build_portfolio_risk_features.py" @()
    Invoke-Stage "19_build_structural_core.py" @()
    Invoke-Stage "21_build_crowding_state.py" @()
}
if ($Stage -in @("returns", "all")) {
    Invoke-Stage "16_build_factor_returns.py" @()
}
if ($Stage -in @("states", "all")) {
    Invoke-Stage "27_build_market_factor_state.py" @()
    Invoke-Stage "23_build_factor_return_stress.py" @()
    Invoke-Stage "24_build_state_panel.py" @()
    Invoke-Stage "28_build_model_feature_panel.py" @()
}
if ($Stage -in @("outcomes", "all")) {
    Invoke-Stage "25_build_dynamic_outcomes.py" @()
    Invoke-Stage "26_build_fixed_outcomes.py" @()
}
if ($Stage -in @("models", "all")) {
    Invoke-Stage "29_run_primary_walk_forward.py" @()
}
