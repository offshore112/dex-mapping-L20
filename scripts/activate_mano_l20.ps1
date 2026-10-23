# Dot-source in PowerShell: . .\scripts\activate_mano_l20.ps1
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentPrefix = Join-Path $projectRoot '.conda\envs\mano-l20'
$condaExecutable = 'D:\conda\Scripts\conda.exe'
if (-not (Test-Path -LiteralPath (Join-Path $environmentPrefix 'python.exe'))) {
    throw "Environment is not ready: $environmentPrefix"
}
if (-not (Test-Path -LiteralPath $condaExecutable)) {
    throw "Conda executable not found: $condaExecutable"
}
# Initialize this PowerShell session only; do not modify the user's profile.
if (Test-Path Variable:PSNativeCommandArgumentPassing) {
    # Conda 22.9's module passes empty _CE_M/_CE_CONDA arguments. PowerShell
    # 7.3+ otherwise preserves them and breaks its native argument handling.
    $PSNativeCommandArgumentPassing = 'Legacy'
}
$condaSessionHook = (& $condaExecutable shell.powershell hook) | Out-String
if ($LASTEXITCODE -ne 0) { throw 'Could not get Conda PowerShell hook.' }
Invoke-Expression $condaSessionHook
conda activate $environmentPrefix
if ($LASTEXITCODE -ne 0) { throw 'Conda activation failed.' }
$env:PYTHONNOUSERSITE = '1'
python -c "import sys; print(sys.executable); print(sys.version.split()[0])"
