#include <petscsys.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>

int main(int argc, char **argv) {
    PetscErrorCode ierr;
    PetscInt       n = 3;      // 矩阵维度
    Mat            A;          // 系数矩阵
    Vec            b, x;       // 右端向量和解向量
    KSP            ksp;       // Krylov 子空间求解器
    PC             pc;        // 预处理器

    // 初始化 PETSc 环境
    ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

    // 创建稀疏矩阵 A（3x3）
    ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
    ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n); CHKERRQ(ierr);
    ierr = MatSetType(A, MATMPIAIJ); CHKERRQ(ierr);  // 分布式存储格式
    ierr = MatSetUp(A); CHKERRQ(ierr);

    // 填充矩阵 A：对角为 2，其余为 1
    for (PetscInt i = 0; i < n; i++) {
        for (PetscInt j = 0; j < n; j++) {
            PetscScalar val = (i == j) ? 2.0 : 1.0;
            ierr = MatSetValue(A, i, j, val, INSERT_VALUES); CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    // 创建右端向量 b（所有元素为 1）
    ierr = VecCreate(PETSC_COMM_WORLD, &b); CHKERRQ(ierr);
    ierr = VecSetSizes(b, PETSC_DECIDE, n); CHKERRQ(ierr);
    ierr = VecSetFromOptions(b); CHKERRQ(ierr);
    ierr = VecSet(b, 1.0); CHKERRQ(ierr);  // 全部赋值为 1
    ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

    // 创建解向量 x
    ierr = VecDuplicate(b, &x); CHKERRQ(ierr);

    // 设置线性求解器（KSP）
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);  // 传入系数矩阵
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);     // Jacobi 预处理
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);      // 共轭梯度法
    ierr = KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);      // 支持命令行参数

    // 求解 Ax = b
    ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

    // 输出解向量 x
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\nSolution vector x:\n"); CHKERRQ(ierr);
    ierr = VecView(x, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    // 释放资源
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    ierr = MatDestroy(&A); CHKERRQ(ierr);
    ierr = VecDestroy(&b); CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}

