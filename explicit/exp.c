static char help[] = "Solves a tridiagonal linear system.\n\n";

#include <petscksp.h>
#include <math.h>
#include <assert.h>
//#include <mpi.h>

int main(int argc,char **args)
{ 
  //MPI_Init(&argc, &args);
  Vec            us,u,f,ft;
  /* us-temperature at the next moment, u-current temperature
  f-heat supply per unit volume, ft-copy of f */
  Mat            A;                /* linear system matrix */
  PetscInt       i,n=100000,m=101,nt=0,col[3],rank,rstart,rend,nlocal; 
  /* n-number of time steps, m-number of spatial mesh poits, nt-current time step*/
  PetscReal      dx=0.01, t=1.0, dt=t/n, k=1.0, r=k*dt/(dx*dx);
  /* dx-spatial mesh size, t-, dt-temporal mesh size, k-conductivity */
  PetscErrorCode ierr;
  PetscScalar    value[3], u0,f0, zero=0.0;
  /* u0-initial value of u, f0-intial value of f */

  assert(dx*(m-1) == 1);
  assert(dt*n == t);
  assert(dt <= 0.5*dx*dx/k);

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, us = Au + dt*f.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed. For this simple case let PETSc decide how
     many elements of the vector are stored on each processor. The second
     argument to VecSetSizes() below causes PETSc to decide.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);  
  ierr = VecSetSizes(u,PETSC_DECIDE,m);CHKERRQ(ierr); 
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&us);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&f);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&ft);CHKERRQ(ierr);

  /* Identify the starting and ending mesh points on each
     processor for the interior part of the mesh. We let PETSc decide
     above. */
  ierr = VecGetOwnershipRange(u,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetLocalSize(u,&nlocal);CHKERRQ(ierr);
  
  /*
     Set the vector elements.
  */
  if( rank == 0 ) {
    for (i=0; i<m; i++) {
      if (i == 0 || i == m - 1) {
	ierr = VecSetValues(u,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
      }
      else {
	u0 = exp(i*dx);
	ierr = VecSetValues(u,1,&i,&u0,INSERT_VALUES);CHKERRQ(ierr);
      }
      f0 = sin(3.14*i*dx); 
      ierr = VecSetValues(f,1,&i,&f0,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);

//  ierr = VecView(f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     We pass in nlocal as the "local" size of the matrix to force it
     to have the same parallel layout as the vector created above.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,nlocal,nlocal,m,m);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  /*
     Assemble matrix.

     The linear system is distributed across the processors by
     chunks of contiguous rows, which correspond to contiguous
     sections of the mesh on which the problem is discretized.
     For matrix assembly, each processor contributes entries for
     the part that it owns locally.
  */

  if (!rstart) {
    rstart = 1;
    i      = 0; col[0] = 0; col[1] = 1; value[0] = 1.0-2*r; value[1] = r;
    ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  if (rend == m) {
    rend = m-1;
    i    = m-1; col[0] = m-2; col[1] = m-1; value[0] = r; value[1] = 1.0-2*r;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Set entries corresponding to the mesh interior */
  value[0] = r; value[1] = 1.0-2*r; value[2] = r;
  for (i=rstart; i<rend; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

//  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  while (nt < n){
    ierr = VecCopy(f,ft);CHKERRQ(ierr);
    ierr = VecScale(ft,(PetscScalar)dt);CHKERRQ(ierr);
    ierr = MatMultAdd(A,u,ft,us);CHKERRQ(ierr);
    i = 0;
    ierr = VecSetValues(us,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
    i = m-1;
    ierr = VecSetValues(us,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecCopy(us,u);CHKERRQ(ierr);
    nt++;
    if (nt == n){
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Temperature in %D iteration is \n", nt);CHKERRQ(ierr);
      ierr = VecView(us,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&u);CHKERRQ(ierr); ierr = VecDestroy(&us);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr); ierr = VecDestroy(&ft);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  //MPI_Finalize();
  return ierr;
}

// EOF`
