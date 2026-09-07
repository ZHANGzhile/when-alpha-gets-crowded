$ErrorActionPreference = "Stop"
$workspacePath = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "python_runtime.ps1")
$pythonPath = Resolve-AlphaCrowdingPython $workspacePath
$manifestDirectory = Join-Path $workspacePath "data\raw\manifests"
$dailyPidPath = Join-Path $manifestDirectory "daily_download.pid"
$statusPath = Join-Path $manifestDirectory "production_supervisor.json"
$dailyManifestPath = Join-Path $manifestDirectory "daily_download.json"
$env:PYTHONPATH = "$workspacePath\.python-packages;$workspacePath\src"
$staleThreshold = [TimeSpan]::FromMinutes(20)
$maximumStaleRestarts = 3
$pollSeconds = 30

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

function Start-DailyRecovery {
    $runStamp = [DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $stdoutPath = Join-Path $manifestDirectory "daily_download.$runStamp.stdout.log"
    $stderrPath = Join-Path $manifestDirectory "daily_download.$runStamp.stderr.log"
    $process = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList @(
            ".\scripts\10_download_universe_daily.py",
            "--workers",
            "1"
        ) `
        -WorkingDirectory $workspacePath `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
    $process.Id | Set-Content -LiteralPath $dailyPidPath -Encoding ascii
    return $process
}

try {
    Write-Status "WAITING" "daily" "waiting for complete universe daily download"
    $staleRestarts = 0
    while ($true) {
        $dailyPid = $null
        $dailyProcess = $null
        if (Test-Path -LiteralPath $dailyPidPath) {
            $dailyPid = [int](Get-Content -LiteralPath $dailyPidPath -Raw)
            $dailyProcess = Get-Process -Id $dailyPid -ErrorAction SilentlyContinue
        }
        if ($null -eq $dailyProcess) {
            $dailyStatus = $null
            if (Test-Path -LiteralPath $dailyManifestPath) {
                $dailyStatus = (
                    Get-Content `
                        -LiteralPath $dailyManifestPath `
                        -Raw `
                        -Encoding utf8 |
                        ConvertFrom-Json
                ).status
            }
            if ($dailyStatus -eq "COMPLETE") {
                break
            }
            if ($staleRestarts -ge $maximumStaleRestarts) {
                throw "daily download exited after $staleRestarts recovery attempts"
            }
            $staleRestarts += 1
            Write-Status `
                "RUNNING" `
                "daily_recovery" `
                "worker exited with status=$dailyStatus; restart $staleRestarts"
            $dailyProcess = Start-DailyRecovery
            Start-Sleep -Seconds 5
            continue
        }
        $lastProgress = $dailyProcess.StartTime.ToUniversalTime()
        if (Test-Path -LiteralPath $dailyManifestPath) {
            $manifestUpdated = (
                Get-Item -LiteralPath $dailyManifestPath
            ).LastWriteTimeUtc
            if ($manifestUpdated -gt $lastProgress) {
                $lastProgress = $manifestUpdated
            }
        }
        $progressAge = [DateTime]::UtcNow - $lastProgress
        if ($progressAge -ge $staleThreshold) {
            if ($staleRestarts -ge $maximumStaleRestarts) {
                throw "daily download remained stale after $staleRestarts restarts"
            }
            $staleRestarts += 1
            Write-Status `
                "RUNNING" `
                "daily_recovery" `
                "stale worker pid=$dailyPid; restart $staleRestarts"
            Stop-Process -Id $dailyPid -Force
            Start-Sleep -Seconds 2
            $dailyProcess = Start-DailyRecovery
            Start-Sleep -Seconds 5
            continue
        }
        Start-Sleep -Seconds $pollSeconds
    }
    $dailyManifest = Get-Content `
        -LiteralPath $dailyManifestPath `
        -Raw `
        -Encoding utf8 |
        ConvertFrom-Json
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
    Invoke-PythonStage "continuous_pseudo_returns" "33_build_continuous_pseudo_returns.py" @()
    Invoke-PythonStage "continuous_pseudo_crowding" "34_build_continuous_pseudo_crowding.py" @()
    Invoke-PythonStage "continuous_pseudo_state" "35_build_continuous_pseudo_state.py" @()
    Invoke-PythonStage "continuous_pseudo_model_features" "37_build_continuous_pseudo_model_features.py" @()
    while ($true) {
        Write-Status `
            "WAITING_PROTOCOL_FREEZE" `
            "protocol_freeze" `
            "pre-outcome artifacts complete; waiting for an intact protocol freeze"
        & $pythonPath (Join-Path $PSScriptRoot "43_validate_protocol_freeze.py")
        if ($LASTEXITCODE -eq 0) {
            break
        }
        if ($LASTEXITCODE -ne 3) {
            throw "protocol freeze validation failed with exit code $LASTEXITCODE"
        }
        Start-Sleep -Seconds 60
    }
    Invoke-PythonStage "dynamic_outcomes" "25_build_dynamic_outcomes.py" @()
    Invoke-PythonStage "fixed_outcomes" "26_build_fixed_outcomes.py" @()
    Invoke-PythonStage "continuous_pseudo_outcomes" "36_build_continuous_pseudo_outcomes.py" @()
    Invoke-PythonStage "primary_walk_forward" "29_run_primary_walk_forward.py" @()
    Invoke-PythonStage "continuous_pseudo_walk_forward" "38_run_continuous_pseudo_walk_forward.py" @()
    Invoke-PythonStage "lead_time_walk_forward" "30_run_lead_time_walk_forward.py" @()
    Invoke-PythonStage "crash_event_study" "31_build_crash_event_study.py" @()
    Invoke-PythonStage "falsification_diagnostics" "32_run_falsification_diagnostics.py" @()
    Invoke-PythonStage "controller_exposure_schedule" "39_build_controller_exposure_schedule.py" @()
    Invoke-PythonStage "controller_reference_data" "42_fetch_controller_reference_data.py" @()
    Invoke-PythonStage "benchmark_replication_gate" "40_validate_benchmark_replication.py" @()
    Invoke-PythonStage "stock_level_controller" "41_run_stock_level_controller.py" @()
    Write-Status "COMPLETE" "stock_level_controller" "stock-level controller paths completed"
}
catch {
    Write-Status "FAILED" "pipeline" $_.Exception.Message
    throw
}
