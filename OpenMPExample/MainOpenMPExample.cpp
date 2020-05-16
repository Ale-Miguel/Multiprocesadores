/*#include <stdio.h>
#include <omp.h>

#define NUM_OF_THREADS 9000

int main() {
	/*
#pragma omp parallel
	{//Parallel region begins
		printf("Hello OMP World!... From thread %d\n", omp_get_thread_num());
	}//Parallel regon ends

	double A[16], B[16], C[16];

	for (int i = 0; i < 16; i++) {
		A[i] = i;
		B[i] = i;
		C[i] = 0;

	}

	int i;
#pragma omp parallel default(none) shared(A,B,C) private(i)
	{
#pragma omp for
		
		for (i= 0; i < 16; i++) {
			C[i] = A[i] + B[i];
			printf("C[%d] = %f from thread %d\n", i, C[i], omp_get_thread_num());
		}
		
		 
	}*/
/*
	omp_set_num_threads(NUM_OF_THREADS);
	int a;
#pragma omp parallel
	{
		if(omp_get_thread_num() == 0)
			printf("Number of threads %d out of %d\n", omp_get_num_threads(), NUM_OF_THREADS);
	}

	//return 0;
}

*/

#include <stdio.h>
#include <omp.h>
#include <Windows.h>

int main() {
	/*int rT = 8185;   //Requested Threads
	int aT = 0;   //Actual Threads

	do {
		rT += 1;
		omp_set_num_threads(rT);
#pragma omp parallel
		{
			if (omp_get_thread_num() == 0) {
				aT = omp_get_num_threads();
				printf("Master Control Program gave me %d threads. He's a nice program.\n", aT);
			}
			int a = 50 + 100; //Just some work...
		}
	} while (rT == aT);

	printf("\nI take it back! Master Control Program gave me %d thread. He's EVIL!\n", aT);*/
	int sharedVar = 6;
#pragma omp parallel num_threads(10)                   
	{
		int privateVar = omp_get_thread_num(); //This data is mine!
		sharedVar = omp_get_thread_num(); //I´ll leave my mark!
		//Add here a pause.
		Sleep(1);
		printf("Thread (%d), privateVar = %d, sharedVar= %d \n", omp_get_thread_num(), privateVar, sharedVar);
	}


	return 0;
}

