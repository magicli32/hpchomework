#include "heat_problem.h"

PetscErrorCode SetBoundaryConditions(HeatProblem *prob, PetscReal time) {
    PetscFunctionBeginUser;
    (void)time;  // 添加此行标记参数未使用
    PetscInt i, idx;
    PetscScalar *values;
    PetscInt start, end;

    PetscCall(VecGetOwnershipRange(prob->u, &start, &end));
    PetscCall(VecGetArray(prob->u, &values));

    for (idx = start; idx < end; idx++) {
        i = idx;

        // 边界条件：Dirichlet 边界条件
        if (i == 0 || i == prob->Nx - 1) {
            values[idx - start] = 0.0;
        }
    }

    PetscCall(VecRestoreArray(prob->u, &values));
    PetscFunctionReturn(0);
}

PetscErrorCode TimeIntegration(HeatProblem *prob, PetscInt max_steps, PetscReal dt) {
    PetscFunctionBeginUser;
    PetscInt step;
    PetscReal t = 0.0;
    PetscScalar *u_values, *u_prev_values;
    PetscInt i, idx;
    PetscInt start, end;
    PetscReal dx = prob->dx;
    PetscReal dx2 = dx * dx;
    PetscReal coef = dt / dx2;

    // 设置时间步长
    prob->dt = dt;

    // 时间迭代
    for (step = 0; step < max_steps; step++) {
        // 保存当前解
        PetscCall(VecCopy(prob->u, prob->u_prev));

        // 应用边界条件
        PetscCall(SetBoundaryConditions(prob, t));

        // 获取数组
        PetscCall(VecGetArray(prob->u, &u_values));
        PetscCall(VecGetArray(prob->u_prev, &u_prev_values));
        PetscCall(VecGetOwnershipRange(prob->u, &start, &end));

        // 内部点的显式时间积分
        for (idx = start; idx < end; idx++) {
            i = idx;

            // 跳过边界点
            if (i == 0 || i == prob->Nx - 1) {
                continue;
            }

            // 计算拉普拉斯算子
            PetscReal laplacian = (u_prev_values[idx - 1 - start] + u_prev_values[idx + 1 - start] - 2.0 * u_prev_values[idx - start]) / dx2;

            // 时间积分
            u_values[idx - start] = u_prev_values[idx - start] + coef * laplacian;
        }

        // 恢复数组
        PetscCall(VecRestoreArray(prob->u, &u_values));
        PetscCall(VecRestoreArray(prob->u_prev, &u_prev_values));

        // 更新时间
        t += dt;

        // 每100步保存一次
        if (step % 100 == 0) {
            PetscCall(SaveToHDF5(prob, step));
            PetscPrintf(PETSC_COMM_WORLD, "Step %d, Time %f\n", step, t);
        }
    }

    // 保存最终结果
    PetscCall(SaveToHDF5(prob, step));
    PetscPrintf(PETSC_COMM_WORLD, "Final step %d, Time %f\n", step, t);

    PetscFunctionReturn(0);
}

