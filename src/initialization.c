#include "heat_problem.h"

PetscErrorCode InitializeProblem(HeatProblem *prob) {
    PetscFunctionBeginUser;
    PetscInt start, end;  // 移除 nlocal
    PetscReal dx, dy;
    PetscInt i, j, idx;
    PetscScalar *values;

    // 计算网格间距
    dx = 1.0 / (prob->Nx - 1);
    dy = 1.0 / (prob->Ny - 1);
    prob->dx = dx;
    prob->dy = dy;

    // 创建向量
    PetscCall(VecCreate(PETSC_COMM_WORLD, &prob->u));
    PetscCall(VecSetSizes(prob->u, PETSC_DECIDE, prob->Nx * prob->Ny));
    PetscCall(VecSetFromOptions(prob->u));
    PetscCall(VecDuplicate(prob->u, &prob->u_prev));

    // 初始化向量
    PetscCall(VecGetOwnershipRange(prob->u, &start, &end));
    PetscCall(VecGetArray(prob->u, &values));

    for (idx = start; idx < end; idx++) {
        i = idx % prob->Nx;
        j = idx / prob->Nx;
        PetscReal x = i * dx;
        PetscReal y = j * dy;

        // 初始条件：中心热源
        if ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) < 0.1 * 0.1) {
            values[idx - start] = 1.0;
        } else {
            values[idx - start] = 0.0;
        }
    }

    PetscCall(VecRestoreArray(prob->u, &values));
    PetscCall(VecCopy(prob->u, prob->u_prev));

    PetscFunctionReturn(0);
}

