#include "heat_problem.h"

PetscErrorCode InitializeProblem(HeatProblem *prob) {
    PetscFunctionBeginUser;
    PetscInt start, end;
    PetscReal dx;
    PetscInt i, idx;
    PetscScalar *values;

    // 计算网格间距
    dx = prob->Lx / (prob->Nx - 1);
    prob->dx = dx;

    // 创建向量
    PetscCall(VecCreate(PETSC_COMM_WORLD, &prob->u));
    PetscCall(VecSetSizes(prob->u, PETSC_DECIDE, prob->Nx));
    PetscCall(VecSetFromOptions(prob->u));
    PetscCall(VecDuplicate(prob->u, &prob->u_prev));

    // 初始化向量
    PetscCall(VecGetOwnershipRange(prob->u, &start, &end));
    PetscCall(VecGetArray(prob->u, &values));

    for (idx = start; idx < end; idx++) {
        i = idx;
        PetscReal x = i * dx;

        // 初始条件：中心热源
        if ((x - 0.5 * prob->Lx) * (x - 0.5 * prob->Lx) < 0.1 * 0.1) {
            values[idx - start] = 1.0;
        } else {
            values[idx - start] = 0.0;
        }
    }

    PetscCall(VecRestoreArray(prob->u, &values));
    PetscCall(VecCopy(prob->u, prob->u_prev));

    PetscFunctionReturn(0);
}

PetscErrorCode SetInitialConditions(HeatProblem *prob) {
    PetscFunctionBeginUser;
    PetscInt start, end;
    PetscReal dx;
    PetscInt i, idx;
    PetscScalar *values;

    // 计算网格间距
    dx = prob->Lx / (prob->Nx - 1);

    // 获取向量数据
    PetscCall(VecGetOwnershipRange(prob->u, &start, &end));
    PetscCall(VecGetArray(prob->u, &values));

    for (idx = start; idx < end; idx++) {
        i = idx;
        PetscReal x = i * dx;

        // 初始条件：中心热源
        if ((x - 0.5 * prob->Lx) * (x - 0.5 * prob->Lx) < 0.1 * 0.1) {
            values[idx - start] = 1.0;
        } else {
            values[idx - start] = 0.0;
        }
    }

    PetscCall(VecRestoreArray(prob->u, &values));

    PetscFunctionReturn(0);
}

