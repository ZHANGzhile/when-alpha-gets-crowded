$ErrorActionPreference = "Stop"
$workspacePath = Split-Path -Parent $PSScriptRoot
$manifestDirectory = Join-Path $workspacePath "data\raw\manifests"
$pidPath = Join-Path $manifestDirectory "production_supervisor.pid"
$stdoutPath = Join-Path $manifestDirectory "production_supervisor.stdout.log"
$stderrPath = Join-Path $manifestDirectory "production_supervisor.stderr.log"
$workerScript = Join-Path $PSScriptRoot "continue_after_daily.ps1"

if (Test-Path -LiteralPath $pidPath) {
    $existingPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $existing = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Write-Output "production supervisor already running pid=$existingPid"
        exit 0
    }
}

New-Item -ItemType Directory -Path $manifestDirectory -Force | Out-Null
$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$workerScript`"") `
    -WorkingDirectory $workspacePath `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
Write-Output "production supervisor started pid=$($process.Id)"
