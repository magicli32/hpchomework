#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main()
{
	float dx, dt, t, L = 1.0, a[100000], b[100000], k = 1.0;
	int i, n1, n2, n = 0, total = 1;
	FILE *F;	
	F = fopen("data.txt", "w");
	dx = 0.01;
	dt = 0.00001;
	t = 1;
	n1 = (int)(L / dx) + 1;		
	n2 = (int)(t / dt);		

	/*The initial temperature at t=0*/
	for (i = 0; i < n1; i++) {   
		if (i == 0 || i == n1 - 1) 
			a[i] = 0;
		else a[i] = exp(i*dx);
		fprintf(F, "%8.4f", a[i]);
	}
	fprintf(F, "\n"); 

	/*Iterative computing*/
	while (n < n2) {    
		for (i = 0; i < n1; i++) {
			/*The temperature on the left and right boundaries is always 0*/
			if (i == 0 || i == n1-1) {
				b[i] = 0;
				a[i] = b[i];
			}     
			else {
				/*difference equation*/
				b[i] = a[i] + k * dt*(a[i + 1] - 2 * a[i] + a[i - 1]) / (dx*dx) + dt*sin(3.14*i*dx);  
		ierr = VecSetValues(us,1,&i,0,INSERT_VALUES);CHKERRQ(ierr);		a[i] = b[i];
			}
			n++;
			fprintf(F, "%8.4f ", a[i]);
			if (total%n1 == 0) {             /*Output data file  */
				fprintf(F, "\n");
			}
			total++;
		}
	}
	fclose(F);
	return 0;
}
