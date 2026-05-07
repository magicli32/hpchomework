# HPC Homework CUDA C++ Version

This project rewrites the original PETSc/MPI homework as a local CUDA C++
program that can be verified on a single NVIDIA GPU. It solves the 1D heat
equation with both explicit Euler and implicit Euler updates, then compares
the GPU result against a CPU reference implementation.

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
.\heat1d_cuda.exe --method explicit
.\heat1d_cuda.exe --method implicit --points 129 --steps 1000 --jacobi-iters 800 --tol 1e-6
```

Optional arguments:

```powershell
.\heat1d_cuda.exe --method explicit --points 257 --steps 20000 --alpha 1.0
.\heat1d_cuda.exe --method implicit --points 129 --steps 1000 --alpha 1.0 --jacobi-iters 800
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

For the explicit method, the GPU directly applies the finite-difference update.
For the implicit method, the CPU reference uses the Thomas tridiagonal solver,
and the GPU path solves the same tridiagonal system with Jacobi iterations so
it can run locally without PETSc, MPI, cuSPARSE, or a cluster environment.

Only the interior points are updated or solved. Boundary points remain fixed at
zero.
