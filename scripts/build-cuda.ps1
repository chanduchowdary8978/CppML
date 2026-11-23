
$ErrorActionPreference = "Stop"

$ROOT = "D:\AI_Projects\CppML"
$CUDA_DIR = "$ROOT\examples\cuda"
$BUILD_DIR = "$ROOT\build\cuda-vector-add"

$VSDEV = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"

if (-not (Test-Path "$CUDA_DIR\CMakeLists.txt")) {
    throw "CUDA example CMakeLists.txt not found."
}

if (-not (Test-Path "$CUDA_DIR\vector_add.cu")) {
    throw "CUDA vector_add.cu not found."
}

if (-not (Test-Path $VSDEV)) {
    throw "Visual Studio developer environment not found."
}

$COMMAND = "call `"$VSDEV`" -arch=x64 -host_arch=x64 && cmake -S `"$CUDA_DIR`" -B `"$BUILD_DIR`" -G Ninja -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_HOST_COMPILER=cl -DCMAKE_CUDA_COMPILER=nvcc -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build `"$BUILD_DIR`" --parallel"

cmd.exe /c $COMMAND

if ($LASTEXITCODE -ne 0) {
    throw "CUDA compilation failed."
}

Write-Host "`n========== RUNNING GPU KERNEL ==========" -ForegroundColor Green

& "$BUILD_DIR\cuda_vector_add.exe"

if ($LASTEXITCODE -ne 0) {
    throw "GPU execution or validation failed."
}

Write-Host "`nCUDA GPU TEST SUCCESSFUL!" -ForegroundColor Green
