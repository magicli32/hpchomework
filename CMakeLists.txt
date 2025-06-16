cmake_minimum_required(VERSION 3.10)
project(HeatEquationSolver)

# 设置编译器和编译选项
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -O3 -g")

# 直接指定PETSc路径
set(PETSC_DIR "/home/lis/petsc")
set(PETSC_ARCH "arch-linux-gnu")  # 根据您的实际架构调整

# 包含PETSc头文件
include_directories(${PETSC_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)  # 指向/home/lis/rechuandao/src

# 先查找 MPI
find_package(MPI REQUIRED)
include_directories(${MPI_C_INCLUDE_DIRS})

# 查找HDF5库
find_package(HDF5 COMPONENTS C REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})
add_definitions(-DH5_HAVE_PARALLEL) 

# 包含头文件目录
include_directories(${PETSC_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# 源文件目录
file(GLOB_RECURSE SOURCES "src/*.c")

# 添加可执行文件
add_executable(heat_solver ${SOURCES})

# 链接库
target_link_libraries(heat_solver 
    ${HDF5_C_LIBRARIES}
    ${MPI_C_LIBRARIES}  # 显式链接 MPI 库
    ${PETSC_DIR}/arch-linux-c-debug/lib/libpetsc.so
    
)

# 设置输出目录
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
# 打印配置信息（便于调试）
message(STATUS "Project path: ${CMAKE_SOURCE_DIR}")
message(STATUS "PETSc path: ${PETSC_DIR}")
message(STATUS "HDF5 path: ${HDF5_INCLUDE_DIRS}")
