param(
    [ValidateRange(1, 4)]
    [int]$Workers = 1
)

$ErrorActionPreference = "Stop"
$workspacePath = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "python_runtime.ps1")
$pythonPath = Resolve-AlphaCrowdingPython $workspacePath
$manifestDirectory = Join-Path $workspacePath "data\raw\manifests"
$pidPath = Join-Path $manifestDirectory "daily_download.pid"
$runStamp = [DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssZ")
$stdoutPath = Join-Path $manifestDirectory "daily_download.$runStamp.stdout.log"
$stderrPath = Join-Path $manifestDirectory "daily_download.$runStamp.stderr.log"

if (Test-Path -LiteralPath $pidPath) {
    $existingPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $existing = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Write-Output "daily download already running pid=$existingPid"
        exit 0
    }
}

New-Item -ItemType Directory -Path $manifestDirectory -Force | Out-Null
$env:PYTHONPATH = "$workspacePath\.python-packages;$workspacePath\src"
$process = Start-Process `
    -FilePath $pythonPath `
    -ArgumentList @(
        ".\scripts\10_download_universe_daily.py",
        "--workers",
        $Workers.ToString()
    ) `
    -WorkingDirectory $workspacePath `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
Write-Output "daily download resumed pid=$($process.Id) workers=$Workers log=$stdoutPath"
