#include "heat_problem.h"
#include "H5FDmpio.h"  // 添加 HDF5 并行 IO 头文件

PetscErrorCode SaveToHDF5(HeatProblem *prob, PetscInt step) {
    PetscFunctionBeginUser;
    hid_t file_id, dataset_id, dataspace_id, global_dataspace_id, plist_id;
    char filename[100];
    PetscScalar *values;
    PetscInt my_rank;
    hsize_t dims[1], global_dims[1];
    PetscInt offset_int[1], dims_int[1];
    herr_t status;
    
    // 获取进程ID
    MPI_Comm_rank(PETSC_COMM_WORLD, &my_rank);
    
    // 创建文件名
    sprintf(filename, "solution_step_%04d.h5", step);
    
    // 创建并行文件访问属性列表
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, PETSC_COMM_WORLD, MPI_INFO_NULL);
    
    // 创建HDF5文件（使用并行IO）
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);
    
    // 写入基本参数（只需要主进程写入）
    if (my_rank == 0) {
        hsize_t scalar_dims[1] = {1};
        dataspace_id = H5Screate_simple(1, scalar_dims, NULL);
        
        // 写入网格参数
        dataset_id = H5Dcreate(file_id, "/Nx", H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &prob->Nx);
        H5Dclose(dataset_id);
        
        dataset_id = H5Dcreate(file_id, "/dx", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &prob->dx);
        H5Dclose(dataset_id);
        
        dataset_id = H5Dcreate(file_id, "/dt", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &prob->dt);
        H5Dclose(dataset_id);
        
        dataset_id = H5Dcreate(file_id, "/step", H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &step);
        H5Dclose(dataset_id);
        
        H5Sclose(dataspace_id);
    }
    
    // 同步所有进程
    MPI_Barrier(PETSC_COMM_WORLD);
    
    // 获取本地向量大小和索引
    PetscCall(VecGetOwnershipRange(prob->u, &offset_int[0], &dims_int[0]));
    dims[0] = dims_int[0];
    
    // 创建本地数据空间
    dataspace_id = H5Screate_simple(1, dims, NULL);
    
    // 创建全局数据空间
    global_dims[0] = prob->Nx;
    global_dataspace_id = H5Screate_simple(1, global_dims, NULL);
    
    // 创建数据集传输属性列表（使用集体IO）
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    
    // 创建数据集
    dataset_id = H5Dcreate(file_id, "/solution", H5T_NATIVE_DOUBLE, global_dataspace_id, 
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    // 获取向量数据
    PetscCall(VecGetArray(prob->u, &values));
    
    // 设置超体元选择（指定本进程负责的数据区域）
    hsize_t count[1] = {dims[0]};
    H5Sselect_hyperslab(global_dataspace_id, H5S_SELECT_SET, (const hsize_t*)offset_int, NULL, count, NULL);
    
    // 集体写入数据
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, dataspace_id, global_dataspace_id, 
                  plist_id, values);
    if (status < 0) {
        PetscPrintf(PETSC_COMM_WORLD, "Error writing dataset\n");
    }
    
    // 恢复向量数据
    PetscCall(VecRestoreArray(prob->u, &values));
    
    // 关闭资源
    H5Pclose(plist_id);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Sclose(global_dataspace_id);
    H5Fclose(file_id);
    
    PetscFunctionReturn(0);
}

PetscErrorCode LoadFromHDF5(HeatProblem *prob, char *filename) {
    PetscFunctionBeginUser;
    hid_t file_id, dataset_id, dataspace_id, global_dataspace_id, plist_id;
    PetscScalar *values;
    hsize_t dims[1];
    PetscInt offset_int[1], dims_int[1];
    herr_t status;
    int mpi_err;
    
    // 创建并行文件访问属性列表
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, PETSC_COMM_WORLD, MPI_INFO_NULL);
    
    // 打开HDF5文件（使用并行IO）
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, plist_id);
    H5Pclose(plist_id);
    
    // 读取基本参数（只需要主进程读取，然后广播到所有进程）
    {
        PetscInt Nx, step;
        PetscReal dx, dt;
        hid_t dataset_id;
        
        if (PetscGlobalRank == 0) {
            dataset_id = H5Dopen(file_id, "/Nx", H5P_DEFAULT);
            H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Nx);
            H5Dclose(dataset_id);
            
            dataset_id = H5Dopen(file_id, "/dx", H5P_DEFAULT);
            H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dx);
            H5Dclose(dataset_id);
            
            dataset_id = H5Dopen(file_id, "/dt", H5P_DEFAULT);
            H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dt);
            H5Dclose(dataset_id);
            
            dataset_id = H5Dopen(file_id, "/step", H5P_DEFAULT);
            H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &step);
            H5Dclose(dataset_id);
        }
        
        // 广播参数到所有进程并检查返回值
        mpi_err = MPI_Bcast(&Nx, 1, MPI_INT, 0, PETSC_COMM_WORLD);
        if (mpi_err != MPI_SUCCESS) PetscPrintf(PETSC_COMM_WORLD, "MPI_Bcast error\n");
        
        mpi_err = MPI_Bcast(&dx, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
        if (mpi_err != MPI_SUCCESS) PetscPrintf(PETSC_COMM_WORLD, "MPI_Bcast error\n");
        
        mpi_err = MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
        if (mpi_err != MPI_SUCCESS) PetscPrintf(PETSC_COMM_WORLD, "MPI_Bcast error\n");
        
        mpi_err = MPI_Bcast(&step, 1, MPI_INT, 0, PETSC_COMM_WORLD);
        if (mpi_err != MPI_SUCCESS) PetscPrintf(PETSC_COMM_WORLD, "MPI_Bcast error\n");
        
        // 检查参数是否匹配
        if (Nx != prob->Nx || dx != prob->dx || dt != prob->dt) {
            PetscPrintf(PETSC_COMM_WORLD, "Warning: Grid parameters from restart file do not match current settings!\n");
        }
    }
    
    // 获取本地向量大小和索引
    PetscCall(VecGetOwnershipRange(prob->u, &offset_int[0], &dims_int[0]));
    dims[0] = dims_int[0];
    
    // 创建本地数据空间
    dataspace_id = H5Screate_simple(1, dims, NULL);
    
    // 打开数据集
    dataset_id = H5Dopen(file_id, "/solution", H5P_DEFAULT);
    
    // 获取全局数据空间
    global_dataspace_id = H5Dget_space(dataset_id);
    
    // 创建数据集传输属性列表（使用集体IO）
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
    
    // 获取向量数据
    PetscCall(VecGetArray(prob->u, &values));
    
    // 设置超体元选择（指定本进程负责读取的数据区域）
    hsize_t count[1] = {dims[0]};
    H5Sselect_hyperslab(global_dataspace_id, H5S_SELECT_SET, (const hsize_t*)offset_int, NULL, count, NULL);
    
    // 集体读取数据
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, dataspace_id, global_dataspace_id, 
                   plist_id, values);
    if (status < 0) {
        PetscPrintf(PETSC_COMM_WORLD, "Error reading dataset\n");
    }
    
    // 恢复向量数据
    PetscCall(VecRestoreArray(prob->u, &values));
    
    // 关闭资源
    H5Pclose(plist_id);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Sclose(global_dataspace_id);
    H5Fclose(file_id);
    
    // 复制当前解到上一时刻解
    PetscCall(VecDuplicate(prob->u, &prob->u_prev));
    PetscCall(VecCopy(prob->u, prob->u_prev));
    
    PetscFunctionReturn(0);
}


