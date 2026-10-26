param([ValidateSet('env', 'test')][string]$Action = 'env')
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot '.venv-e1/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Local environment missing; read MIGRATION.md.' }
$env:HF_HOME = Join-Path $ProjectRoot '.hf_cache'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
Push-Location -LiteralPath $ProjectRoot
try {
    if ($Action -eq 'env') {
        & $PythonPath -c "import sys, pathlib, torch, torchvision, torchhd, transformers, sklearn, common.thesis_eval, ssl_hdc_pilot; print('Python:',sys.executable); print('Base:',sys.base_prefix); print('Torch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('Metrics:',common.thesis_eval.__file__); print('HDC:',ssl_hdc_pilot.__file__)"
    } else {
        $PreviousCuda = $env:CUDA_VISIBLE_DEVICES
        try {
            $env:CUDA_VISIBLE_DEVICES = '-1'
            $TestRoot = Join-Path $ProjectRoot ('.cache/test-' + [guid]::NewGuid().ToString('N'))
            & $PythonPath -m pytest ssl_hdc_pilot/tests -q -p no:cacheprovider --basetemp $TestRoot
        } finally { $env:CUDA_VISIBLE_DEVICES = $PreviousCuda }
    }
    if ($LASTEXITCODE -ne 0) { throw "$Action failed with exit $LASTEXITCODE" }
} finally { Pop-Location }
