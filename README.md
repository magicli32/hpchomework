# HPC final project

There are 4 folders, alpha_beta, explicit, implicit and verification.  

Folder alpha_beta:   
    - alpha_beta.py is used to calculate alpha and beta;   
    - the numerical solutions at different dx are stacked vertically in alpha.txt.  

Folder explicit:   
    - ref.c is an explicit Euler alporithm written only in C language, used to understand the alporithm;  
    - exp.c is the explicit Euler alporithm;   
    - exp_hdf5.c is the explicit Euler alporithm with HDF5 restart function, but it has warnings at compile time due to problems linking the HDF5 libraries;  
    - Makefile is used to compile exp.c, and can be used to compile exp_hdf5.c after simple modification;   
    - ty_script is used to run exp.out on TaiYi, and can be used to run exp_hdf5.out after simple modification.  
    

Folder implicit:  
    - imp.c is the implicii Euler alporithm;  
    - Makefile is used to compile imp.c;  
    - ty_script is used to run imp.out on TaiYi.  
    
Folder verification:  
    - figure_error.py is used to plot figure and calculate error for verification;  
    - the numerical solutions of verification case are stored in num_solution.txt.  
    
