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

enum class Method {
  explicit_euler,
  implicit_euler,
};

struct Options {
  Method method = Method::explicit_euler;
  int points = 257;
  int steps = 20000;
  int jacobi_iters = 400;
  double alpha = 1.0;
  double t_final = 0.1;
  double tolerance = 1.0e-8;
};

__global__ void explicit_step_kernel(const double *current, double *next,
                                     int points, double r, double dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i < points - 1) {
    double x = static_cast<double>(i) / static_cast<double>(points - 1);
    double forcing = sin(3.14159265358979323846 * x);
    next[i] = current[i] + r * (current[i - 1] - 2.0 * current[i] + current[i + 1]) +
              dt * forcing;
  }
}

__global__ void build_implicit_rhs_kernel(const double *current, double *rhs,
                                          int points, double dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i < points - 1) {
    double x = static_cast<double>(i) / static_cast<double>(points - 1);
    rhs[i] = current[i] + dt * sin(3.14159265358979323846 * x);
  }
}

__global__ void implicit_jacobi_kernel(const double *old_guess,
                                       const double *rhs, double *new_guess,
                                       int points, double r) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i < points - 1) {
    new_guess[i] = (rhs[i] + r * (old_guess[i - 1] + old_guess[i + 1])) /
                   (1.0 + 2.0 * r);
  }
}

void print_usage(const char *program) {
  std::cout << "Usage: " << program
            << " [--method explicit|implicit] [--points N] [--steps N]\n"
            << "       [--alpha A] [--time T] [--tol E] [--jacobi-iters N]\n";
}

Method parse_method(const char *value) {
  if (std::strcmp(value, "explicit") == 0) {
    return Method::explicit_euler;
  }
  if (std::strcmp(value, "implicit") == 0) {
    return Method::implicit_euler;
  }
  throw std::invalid_argument("--method must be explicit or implicit");
}

const char *method_name(Method method) {
  return method == Method::explicit_euler ? "explicit Euler" : "implicit Euler";
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

    if (std::strcmp(argv[i], "--method") == 0) {
      opts.method = parse_method(require_value("--method"));
    } else if (std::strcmp(argv[i], "--points") == 0) {
      opts.points = std::atoi(require_value("--points"));
    } else if (std::strcmp(argv[i], "--steps") == 0) {
      opts.steps = std::atoi(require_value("--steps"));
    } else if (std::strcmp(argv[i], "--jacobi-iters") == 0) {
      opts.jacobi_iters = std::atoi(require_value("--jacobi-iters"));
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
  if (opts.jacobi_iters < 1) {
    throw std::invalid_argument("--jacobi-iters must be positive");
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

void run_explicit_cpu(std::vector<double> &u, int steps, double alpha,
                      double dt) {
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

void thomas_solve(std::vector<double> &rhs, double r) {
  int points = static_cast<int>(rhs.size());
  int interior = points - 2;
  std::vector<double> c_prime(interior, 0.0);
  std::vector<double> d_prime(interior, 0.0);
  double diag = 1.0 + 2.0 * r;
  double off = -r;

  c_prime[0] = off / diag;
  d_prime[0] = rhs[1] / diag;
  for (int j = 1; j < interior; ++j) {
    double denom = diag - off * c_prime[j - 1];
    c_prime[j] = j == interior - 1 ? 0.0 : off / denom;
    d_prime[j] = (rhs[j + 1] - off * d_prime[j - 1]) / denom;
  }

  rhs[points - 2] = d_prime[interior - 1];
  for (int j = interior - 2; j >= 0; --j) {
    rhs[j + 1] = d_prime[j] - c_prime[j] * rhs[j + 2];
  }
  rhs[0] = 0.0;
  rhs[points - 1] = 0.0;
}

void run_implicit_cpu(std::vector<double> &u, int steps, double alpha,
                      double dt) {
  int points = static_cast<int>(u.size());
  double dx = 1.0 / static_cast<double>(points - 1);
  double r = alpha * dt / (dx * dx);
  std::vector<double> rhs(points, 0.0);

  for (int step = 0; step < steps; ++step) {
    for (int i = 1; i < points - 1; ++i) {
      double x = i * dx;
      rhs[i] = u[i] + dt * std::sin(3.14159265358979323846 * x);
    }
    thomas_solve(rhs, r);
    std::swap(u, rhs);
    rhs[0] = 0.0;
    rhs[points - 1] = 0.0;
  }
}

std::vector<double> run_explicit_gpu(const std::vector<double> &initial,
                                     int steps, double alpha, double dt,
                                     float &elapsed_ms) {
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
    explicit_step_kernel<<<blocks, threads>>>(d_current, d_next, points, r, dt);
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

std::vector<double> run_implicit_gpu(const std::vector<double> &initial,
                                     int steps, int jacobi_iters, double alpha,
                                     double dt, float &elapsed_ms) {
  int points = static_cast<int>(initial.size());
  double dx = 1.0 / static_cast<double>(points - 1);
  double r = alpha * dt / (dx * dx);
  std::vector<double> result(points, 0.0);

  double *d_current = nullptr;
  double *d_rhs = nullptr;
  double *d_next = nullptr;
  std::size_t bytes = static_cast<std::size_t>(points) * sizeof(double);

  CUDA_CHECK(cudaMalloc(&d_current, bytes));
  CUDA_CHECK(cudaMalloc(&d_rhs, bytes));
  CUDA_CHECK(cudaMalloc(&d_next, bytes));
  CUDA_CHECK(cudaMemcpy(d_current, initial.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_rhs, 0, bytes));
  CUDA_CHECK(cudaMemset(d_next, 0, bytes));

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  int threads = 256;
  int blocks = (points - 2 + threads - 1) / threads;

  CUDA_CHECK(cudaEventRecord(start));
  for (int step = 0; step < steps; ++step) {
    build_implicit_rhs_kernel<<<blocks, threads>>>(d_current, d_rhs, points, dt);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(d_next, d_current, bytes, cudaMemcpyDeviceToDevice));
    for (int iter = 0; iter < jacobi_iters; ++iter) {
      implicit_jacobi_kernel<<<blocks, threads>>>(d_current, d_rhs, d_next, points, r);
      CUDA_CHECK(cudaGetLastError());
      std::swap(d_current, d_next);
    }
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  CUDA_CHECK(cudaMemcpy(result.data(), d_current, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_current));
  CUDA_CHECK(cudaFree(d_rhs));
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

    if (opts.method == Method::explicit_euler && r > 0.5) {
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
    std::cout << "Method: " << method_name(opts.method) << "\n";
    std::cout << "Points: " << opts.points << ", steps: " << opts.steps
              << ", alpha: " << opts.alpha << ", final time: " << opts.t_final
              << "\n";
    std::cout << "dx: " << dx << ", dt: " << dt << ", r: " << r << "\n";

    std::vector<double> cpu = initial_state(opts.points);
    std::vector<double> initial = cpu;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    if (opts.method == Method::explicit_euler) {
      run_explicit_cpu(cpu, opts.steps, opts.alpha, dt);
    } else {
      run_implicit_cpu(cpu, opts.steps, opts.alpha, dt);
    }
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

    float gpu_ms = 0.0f;
    std::vector<double> gpu =
        opts.method == Method::explicit_euler
            ? run_explicit_gpu(initial, opts.steps, opts.alpha, dt, gpu_ms)
            : run_implicit_gpu(initial, opts.steps, opts.jacobi_iters, opts.alpha,
                               dt, gpu_ms);

    double err = max_abs_error(cpu, gpu);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CPU time: " << cpu_ms << " ms\n";
    std::cout << "GPU time: " << gpu_ms << " ms\n";
    if (opts.method == Method::implicit_euler) {
      std::cout << "GPU Jacobi iterations per step: " << opts.jacobi_iters << "\n";
    }
    std::cout << std::scientific << "Max absolute error: " << err << "\n";
    std::cout << "Verification: " << (err <= opts.tolerance ? "PASS" : "FAIL")
              << "\n";

    return err <= opts.tolerance ? 0 : 1;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
