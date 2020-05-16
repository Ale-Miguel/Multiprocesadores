/*#include <stdio.h>
#include <time.h>
#include <omp.h>
#define MAXTHREADS 8
long cantidadIntervalos = 10000000; //10 Million
double baseIntervalo;
//double acum = 0; //No puede ser una variable global
double acumG = 0; //No puede ser una variable global
clock_t start, end;

void main() {
	int THREADS = MAXTHREADS;
	baseIntervalo = 1.0 / (double)cantidadIntervalos;

	double x, partialSum[MAXTHREADS], totalSum = 0;
	omp_set_num_threads(THREADS);
	start = clock();
#pragma omp parallel
	{
		int numThread = omp_get_thread_num();
		double acum = 0; //No puede ser una variable global. Es una variable privada al thread.
		double fdx = 0; //No puede ser una variable global. Es una variable privada al thread.
		for (long i = numThread; i < cantidadIntervalos; i += THREADS) {
			x = i * baseIntervalo;
			fdx = 4 / (1 + x * x);
			acum += fdx;
//#pragma omp atomic
	//		acumG += fdx * baseIntervalo;
		}
		acum *= baseIntervalo; //Multiplico todas las alturas de los rectangulos acumuladas por el tamaño de la base.
		partialSum[numThread] = acum;
//#pragma omp atomic
		//acumG += acum;
		printf("Resultado parcial (Thread %d)\nacum = %lf\n", numThread, acum);
	}
	end = clock();

	for (int c = 0; c < THREADS; c++)
		totalSum += partialSum[c];
	printf("\nResultado (%d threads) = %20.18lf (%ld)\n", THREADS, totalSum, end - start);

}
*/
/*
#include <stdio.h>
#include <time.h>

long cantidadIntervalos = 1000000000;
double baseIntervalo;

double acum = 0;
clock_t start, end;

void main() {

	long i;
	baseIntervalo = 1.0 / cantidadIntervalos;
	start = clock();
#pragma omp parallel
	{
		double fdx;
		double x = 0;
		#pragma omp for reduction(+:acum)
		for (i = 0; i < cantidadIntervalos; i++) {
			x = (i + 0.5) * baseIntervalo;
			fdx = 4 / (1 + x * x);
			acum += fdx;
		}
	}
	
	acum *= baseIntervalo;
	end = clock();
	printf("Result = %20.18lf (%ld)\n", acum, end - start);
}
*/

#include <stdio.h>
#include <time.h>

long cantidadIntervalos = 100000000;
double baseIntervalo;
double fdx;
double acum = 0;
clock_t start, end;

void main() {
	double x = 0;
	long i;
	baseIntervalo = 1.0 / cantidadIntervalos;
	start = clock();

#pragma omp parallel for  reduction(+:acum) private(x,fdx)
	for (i = 0; i < cantidadIntervalos; i++) {
		x = (i + 0.5) * baseIntervalo;
		fdx = 4 / (1 + x * x);
		acum += fdx;
	}
	acum *= baseIntervalo;
	end = clock();
	printf("Result = %20.18lf (%ld)\n", acum, end - start);
}
