@echo off
setlocal

where nvcc >nul 2>nul
if errorlevel 1 (
  echo nvcc was not found in PATH.
  exit /b 1
)

nvcc -O2 -std=c++17 src\heat1d_cuda.cu -o heat1d_cuda.exe
if errorlevel 1 exit /b 1

heat1d_cuda.exe
