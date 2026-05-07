# HPC Homework CUDA C++ Version

This project rewrites the original PETSc/MPI homework as a local CUDA C++
program that can be verified on a single NVIDIA GPU. It solves the 1D heat
equation with an explicit Euler update and compares the GPU result against a
CPU reference implementation.

## Requirements

- NVIDIA GPU with a CUDA-capable driver
- CUDA Toolkit, including `nvcc`
- A supported host C++ compiler for `nvcc`

The program was prepared for local validation on Windows with CUDA 12.9.

## Build

From this directory:

```powershell
nvcc -O2 -std=c++17 src\heat1d_cuda.cu -o heat1d_cuda.exe
```

If `nvcc` cannot find a host compiler, open the "x64 Native Tools Command
Prompt for VS" and run the same command there.

On Windows, you can also run:

```bat
build_windows.bat
```

## Run

```powershell
.\heat1d_cuda.exe
```

Optional arguments:

```powershell
.\heat1d_cuda.exe --points 257 --steps 20000 --alpha 1.0
```

## Expected Output

The program prints GPU information, simulation parameters, CPU/GPU runtime,
and the maximum absolute error between the CPU and GPU arrays. A successful
run ends with:

```text
Verification: PASS
```

## Model

The solved equation is:

```text
u_t = alpha * u_xx + sin(pi * x),  0 <= x <= 1
u(0,t) = u(1,t) = 0
u(x,0) = exp(x)
```

Only the interior points are updated on the GPU. Boundary points remain fixed
at zero.
