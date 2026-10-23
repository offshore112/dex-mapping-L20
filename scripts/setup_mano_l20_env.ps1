param(
    [string]$CondaExe = 'D:\conda\Scripts\conda.exe'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentPrefix = Join-Path $projectRoot '.conda\envs\mano-l20'
$environmentPython = Join-Path $environmentPrefix 'python.exe'
$env:CONDA_PKGS_DIRS = Join-Path $projectRoot '.conda\pkgs'
$env:CONDA_ENVS_PATH = Join-Path $projectRoot '.conda\envs'
$env:PYTHONNOUSERSITE = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$pipCache = Join-Path $projectRoot '.conda\pip-cache'

if (-not (Test-Path -LiteralPath $CondaExe)) {
    throw "Conda executable not found: $CondaExe"
}
if (-not (Test-Path -LiteralPath $environmentPython)) {
    $condaLock = Join-Path $projectRoot 'environment\conda-win-64-explicit.txt'
    if (Test-Path -LiteralPath $condaLock) {
        & $CondaExe create --prefix $environmentPrefix --file $condaLock --no-default-packages -y
    } else {
        & $CondaExe create --prefix $environmentPrefix --override-channels --channel conda-forge --no-default-packages python=3.11 pip -y
    }
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed.' }
}

$torchLock = Join-Path $projectRoot 'environment\requirements-torch-cpu.txt'
if (Test-Path -LiteralPath $torchLock) {
    & $environmentPython -m pip install --requirement $torchLock --index-url https://download.pytorch.org/whl/cpu --cache-dir $pipCache
} else {
    & $environmentPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu --cache-dir $pipCache
}
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }

$packageLock = Join-Path $projectRoot 'environment\requirements-runtime-lock.txt'
if (-not (Test-Path -LiteralPath $packageLock)) {
    $packageLock = Join-Path $projectRoot 'environment\requirements-mano-l20.txt'
}
& $environmentPython -m pip install --requirement $packageLock --index-url https://pypi.org/simple --cache-dir $pipCache
if ($LASTEXITCODE -ne 0) { throw 'MANO dependencies installation failed.' }
& $environmentPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Package consistency check failed.' }
Write-Output "Environment ready: $environmentPrefix"
Write-Output "Activate with: conda activate `"$environmentPrefix`""
