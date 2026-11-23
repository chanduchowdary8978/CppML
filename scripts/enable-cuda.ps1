
$ErrorActionPreference = "Stop"

Write-Host "`n========== CppML CUDA ENVIRONMENT CHECK ==========" -ForegroundColor Cyan

$NVCC = Get-Command nvcc -ErrorAction SilentlyContinue

if (-not $NVCC) {
    throw "CUDA compiler nvcc not found in PATH."
}

Write-Host "`n========== CUDA COMPILER ==========" -ForegroundColor Green
nvcc --version

if ($LASTEXITCODE -ne 0) {
    throw "Could not execute nvcc."
}

Write-Host "`n========== NVIDIA GPU ==========" -ForegroundColor Green
nvidia-smi

if ($LASTEXITCODE -ne 0) {
    throw "NVIDIA GPU verification failed."
}

Write-Host "`n========== PROJECT CUDA FILES ==========" -ForegroundColor Green

$ROOT = "D:\AI_Projects\CppML"

$RequiredFiles = @(
    "$ROOT\src\math\cuda_operations.cu",
    "$ROOT\src\models\cuda_linear_regression.cu",
    "$ROOT\examples\cuda\CMakeLists.txt",
    "$ROOT\examples\cuda\vector_add.cu"
)

foreach ($File in $RequiredFiles) {
    if (-not (Test-Path $File)) {
        throw "Required CUDA file missing: $File"
    }

    Write-Host "[OK] $File" -ForegroundColor Green
}

Write-Host "`nCUDA environment verification successful." -ForegroundColor Green
