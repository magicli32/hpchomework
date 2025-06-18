cmake_minimum_required(VERSION 3.10)
project(HeatEquationSolver)

# ======================
# 基础编译设置
# ======================
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -O3 -g")

# ======================
# MPI 配置
# ======================
set(MPI_DIR "/home/lis/petsc/arch-linux-c-debug" CACHE PATH "MPI installation directory")
find_package(MPI REQUIRED)
message(STATUS "MPI库路径: ${MPI_C_LIBRARIES}")

# ======================
# PETSc 配置
# ======================
set(PETSC_DIR "/home/lis/petsc")
set(PETSC_ARCH "arch-linux-c-debug")

# ======================
# HDF5 配置
# ======================
set(HDF5_DIR "/usr/lib/x86_64-linux-gnu/hdf5/openmpi/cmake" CACHE PATH "HDF5 CMake directory")
set(HDF5_PREFER_PARALLEL TRUE CACHE BOOL "Prefer parallel HDF5")
find_package(HDF5 COMPONENTS C HL REQUIRED)

# ======================
# 头文件和源文件设置
# ======================
include_directories(
    ${PETSC_DIR}/include
    ${PETSC_DIR}/${PETSC_ARCH}/include
    ${CMAKE_SOURCE_DIR}/src
    ${HDF5_INCLUDE_DIRS}
    ${MPI_C_INCLUDE_DIRS}
)

# 确保包含所有源文件
set(SOURCES
    src/main.c
    src/initialization.c
    src/hdf5_restart.c
    src/time_integration.c
    # 添加其他源文件...
)

# ======================
# 定义可执行文件（关键修改点）
# ======================
add_executable(heat_solver ${SOURCES})

# ======================
# 链接设置（紧跟在add_executable之后）
# ======================
target_link_libraries(heat_solver 
    ${HDF5_C_LIBRARIES}
    ${HDF5_C_HL_LIBRARIES}
    ${MPI_C_LIBRARIES}
    ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so
)

# ======================
# 输出目录
# ======================
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)


