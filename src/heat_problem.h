#ifndef HEAT_PROBLEM_H
#define HEAT_PROBLEM_H

#include "petsc.h"
#include "hdf5.h"

typedef struct {
    PetscReal rho, c, kappa;  // 物理参数
    PetscInt Nx;         // 网格点数
    PetscReal Lx;        // 区域尺寸
    PetscReal dx, dt;    // 网格步长和时间步长
    PetscReal Tfinal;    // 终止时间
    
    // PETSc数据结构
    Mat A;                    // 系数矩阵
    Vec u, u_prev, f;        // 当前解、上一时刻解、源项
} HeatProblem;

// 函数声明
PetscErrorCode InitializeProblem(HeatProblem *prob);
PetscErrorCode SetInitialConditions(HeatProblem *prob);
PetscErrorCode SetBoundaryConditions(HeatProblem *prob, PetscReal time);
PetscErrorCode DefineSourceTerm(HeatProblem *prob, PetscReal time);
PetscErrorCode TimeIntegration(HeatProblem *prob, PetscInt max_steps, PetscReal dt);
PetscErrorCode SaveToHDF5(HeatProblem *prob, PetscInt step);
PetscErrorCode LoadFromHDF5(HeatProblem *prob, char *filename);

#endif /* HEAT_PROBLEM_H */



