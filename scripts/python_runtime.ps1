function Resolve-AlphaCrowdingPython([string]$WorkspacePath) {
    if ($env:ALPHA_CROWDING_PYTHON) {
        if (-not (Test-Path -LiteralPath $env:ALPHA_CROWDING_PYTHON)) {
            throw "ALPHA_CROWDING_PYTHON does not exist: $env:ALPHA_CROWDING_PYTHON"
        }
        return (Resolve-Path -LiteralPath $env:ALPHA_CROWDING_PYTHON).Path
    }
    $candidates = @(
        (Join-Path $WorkspacePath ".venv\Scripts\python.exe"),
        (Join-Path $WorkspacePath "venv\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        throw "Python was not found. Create .venv or set ALPHA_CROWDING_PYTHON."
    }
    return $command.Source
}
