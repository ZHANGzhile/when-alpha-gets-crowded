$ErrorActionPreference = "Stop"
$workspacePath = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "python_runtime.ps1")
$pythonPath = Resolve-AlphaCrowdingPython $workspacePath
$env:PYTHONPATH = "$workspacePath\.python-packages;$workspacePath\src"

& $pythonPath -m unittest discover -s "$workspacePath\tests" -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonPath -m compileall -q "$workspacePath\src" "$workspacePath\scripts"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonPath -c "from pathlib import Path; import yaml; [yaml.safe_load(p.read_text(encoding='utf-8')) for p in Path(r'$workspacePath\config').glob('*.yaml')]; print('YAML configuration parse: OK')"
exit $LASTEXITCODE
