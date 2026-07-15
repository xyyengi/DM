param(
    [string]$Archive = "shandong_v3_vmix_server.tar.gz"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$paths = @(
    "diffusion_npy_normalized",
    "configs/v3_actual_forecast_time_encoding_168h.yaml",
    "configs/v_mix_residual_forecast_concat_guidance.yaml",
    "src",
    "dataset_multivariate.py",
    "diff_models_multivariate.py",
    "evaluation.py",
    "train.py",
    "generate.py",
    "requirements.txt",
    "run_shandong_v3_vmix.sh"
)

& tar.exe -czf $Archive --exclude="__pycache__" --exclude="*.pyc" @paths
if ($LASTEXITCODE -ne 0) {
    throw "tar failed with exit code $LASTEXITCODE"
}

Get-Item -LiteralPath $Archive | Select-Object FullName, Length, LastWriteTime
