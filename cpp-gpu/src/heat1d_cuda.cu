#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status = (call);                                                \
    if (status != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) + " - " +              \
                               cudaGetErrorString(status));                     \
    }                                                                           \
  } while (0)

struct Options {
  int points = 257;
  int steps = 20000;
  double alpha = 1.0;
  double t_final = 0.1;
  double tolerance = 1.0e-10;
};

__global__ void heat_step_kernel(const double *current, double *next, int points,
                                 double r, double dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i < points - 1) {
    double x = static_cast<double>(i) / static_cast<double>(points - 1);
    double forcing = sin(3.14159265358979323846 * x);
    next[i] = current[i] + r * (current[i - 1] - 2.0 * current[i] + current[i + 1]) +
              dt * forcing;
  }
}

void print_usage(const char *program) {
  std::cout << "Usage: " << program
            << " [--points N] [--steps N] [--alpha A] [--time T] [--tol E]\n";
}

Options parse_args(int argc, char **argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    auto require_value = [&](const char *name) -> char * {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("Missing value for ") + name);
      }
      return argv[++i];
    };

    if (std::strcmp(argv[i], "--points") == 0) {
      opts.points = std::atoi(require_value("--points"));
    } else if (std::strcmp(argv[i], "--steps") == 0) {
      opts.steps = std::atoi(require_value("--steps"));
    } else if (std::strcmp(argv[i], "--alpha") == 0) {
      opts.alpha = std::atof(require_value("--alpha"));
    } else if (std::strcmp(argv[i], "--time") == 0) {
      opts.t_final = std::atof(require_value("--time"));
    } else if (std::strcmp(argv[i], "--tol") == 0) {
      opts.tolerance = std::atof(require_value("--tol"));
    } else if (std::strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument(std::string("Unknown argument: ") + argv[i]);
    }
  }

  if (opts.points < 3) {
    throw std::invalid_argument("--points must be at least 3");
  }
  if (opts.steps < 1) {
    throw std::invalid_argument("--steps must be positive");
  }
  if (opts.alpha <= 0.0) {
    throw std::invalid_argument("--alpha must be positive");
  }
  if (opts.t_final <= 0.0) {
    throw std::invalid_argument("--time must be positive");
  }
  return opts;
}

std::vector<double> initial_state(int points) {
  std::vector<double> u(points, 0.0);
  double dx = 1.0 / static_cast<double>(points - 1);
  for (int i = 1; i < points - 1; ++i) {
    u[i] = std::exp(i * dx);
  }
  return u;
}

void run_cpu(std::vector<double> &u, int steps, double alpha, double dt) {
  int points = static_cast<int>(u.size());
  double dx = 1.0 / static_cast<double>(points - 1);
  double r = alpha * dt / (dx * dx);
  std::vector<double> next(points, 0.0);

  for (int step = 0; step < steps; ++step) {
    for (int i = 1; i < points - 1; ++i) {
      double x = i * dx;
      double forcing = std::sin(3.14159265358979323846 * x);
      next[i] = u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1]) + dt * forcing;
    }
    std::swap(u, next);
    next[0] = 0.0;
    next[points - 1] = 0.0;
  }
}

std::vector<double> run_gpu(const std::vector<double> &initial, int steps,
                            double alpha, double dt, float &elapsed_ms) {
  int points = static_cast<int>(initial.size());
  double dx = 1.0 / static_cast<double>(points - 1);
  double r = alpha * dt / (dx * dx);
  std::vector<double> result(points, 0.0);

  double *d_current = nullptr;
  double *d_next = nullptr;
  std::size_t bytes = static_cast<std::size_t>(points) * sizeof(double);

  CUDA_CHECK(cudaMalloc(&d_current, bytes));
  CUDA_CHECK(cudaMalloc(&d_next, bytes));
  CUDA_CHECK(cudaMemcpy(d_current, initial.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_next, 0, bytes));

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  int threads = 256;
  int blocks = (points - 2 + threads - 1) / threads;

  CUDA_CHECK(cudaEventRecord(start));
  for (int step = 0; step < steps; ++step) {
    heat_step_kernel<<<blocks, threads>>>(d_current, d_next, points, r, dt);
    CUDA_CHECK(cudaGetLastError());
    std::swap(d_current, d_next);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  CUDA_CHECK(cudaMemcpy(result.data(), d_current, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_current));
  CUDA_CHECK(cudaFree(d_next));

  return result;
}

double max_abs_error(const std::vector<double> &a, const std::vector<double> &b) {
  double err = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    err = std::max(err, std::abs(a[i] - b[i]));
  }
  return err;
}

int main(int argc, char **argv) {
  try {
    Options opts = parse_args(argc, argv);
    double dx = 1.0 / static_cast<double>(opts.points - 1);
    double dt = opts.t_final / static_cast<double>(opts.steps);
    double r = opts.alpha * dt / (dx * dx);

    if (r > 0.5) {
      std::cerr << "Unstable explicit Euler parameters: alpha * dt / dx^2 = "
                << r << " > 0.5\n";
      std::cerr << "Increase --steps or decrease --points/--alpha/--time.\n";
      return 2;
    }

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Points: " << opts.points << ", steps: " << opts.steps
              << ", alpha: " << opts.alpha << ", final time: " << opts.t_final
              << "\n";
    std::cout << "dx: " << dx << ", dt: " << dt << ", r: " << r << "\n";

    std::vector<double> cpu = initial_state(opts.points);
    std::vector<double> initial = cpu;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    run_cpu(cpu, opts.steps, opts.alpha, dt);
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

    float gpu_ms = 0.0f;
    std::vector<double> gpu = run_gpu(initial, opts.steps, opts.alpha, dt, gpu_ms);
    double err = max_abs_error(cpu, gpu);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CPU time: " << cpu_ms << " ms\n";
    std::cout << "GPU time: " << gpu_ms << " ms\n";
    std::cout << std::scientific << "Max absolute error: " << err << "\n";
    std::cout << "Verification: " << (err <= opts.tolerance ? "PASS" : "FAIL")
              << "\n";

    return err <= opts.tolerance ? 0 : 1;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
