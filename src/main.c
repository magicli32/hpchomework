#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "petsc.h"
#include "heat_problem.h"

int main(int argc, char **argv) {
    HeatProblem prob;
    PetscInt step = 0;
    PetscReal current_time = 0.0;
    PetscReal dt = 0.001;
    PetscInt max_steps = 1000;
    char restart_file[100] = "";
    
    // 初始化PETSc环境
    PetscCall(PetscInitialize(&argc, &argv, NULL, "Heat Equation Solver"));
    
    // 处理命令行参数
    PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Heat Solver Options", "");
    PetscCall(PetscOptionsInt("-nx", "Number of grid points in x-direction", "", 100, &prob.Nx, NULL));
    PetscCall(PetscOptionsReal("-dt", "Time step", "", 0.001, &prob.dt, NULL));
    PetscCall(PetscOptionsReal("-tfinal", "Final time", "", 1.0, &prob.Tfinal, NULL));
    PetscCall(PetscOptionsString("-restart_file", "Restart file", "", restart_file, restart_file, 100, NULL));
    PetscOptionsEnd();
    
    // 设置物理参数
    prob.rho = 1.0;    // 密度
    prob.c = 1.0;      // 热容量
    prob.kappa = 1.0;  // 热导率
    prob.Lx = 1.0;     // 区域x方向长度
    prob.dx = prob.Lx / prob.Nx;
    
    // 初始化问题
    PetscCall(InitializeProblem(&prob));
    
    // 若提供重启文件，则加载；否则设置初始条件
    if (restart_file[0] != '\0') {
        PetscCall(LoadFromHDF5(&prob, restart_file));
        sscanf(restart_file, "solution_step_%d.h5", &step);
        current_time = step * prob.dt;
        PetscPrintf(PETSC_COMM_WORLD, "Restart from step %d, time = %f\n", step, current_time);
    } else {
        // 设置初始条件
        PetscCall(SetInitialConditions(&prob));
        PetscCall(VecDuplicate(prob.u, &prob.u_prev));
        PetscCall(VecCopy(prob.u, prob.u_prev));
    }
    
    // 时间积分求解
    PetscCall(TimeIntegration(&prob, max_steps, dt));
    
    // 保存最终结果
    PetscCall(SaveToHDF5(&prob, step));
    
    // 释放资源
    PetscCall(VecDestroy(&prob.u));
    PetscCall(VecDestroy(&prob.u_prev));
    PetscCall(VecDestroy(&prob.f));
    PetscCall(MatDestroy(&prob.A));
    PetscCall(PetscFinalize());
    return 0;
}

