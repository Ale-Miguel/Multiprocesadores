/*
Alejandro Miguel Sánchez Mora
A01272352

OMP Code to calculate Pi

Code adapted from the multithreaded version 
*/

#include <stdio.h>
#include <omp.h>
#include <Windows.h>
#include <time.h>

#define NUMBER_OF_THREADS 32
#define NUMBER_OF_INTERVALS 1000000000

double resultados[NUMBER_OF_THREADS];

long double interval = 1.0 / NUMBER_OF_INTERVALS;  

int main() {
	//int sharedVar = 6;
    clock_t start, endT;

    start = clock();

#pragma omp parallel num_threads(NUMBER_OF_THREADS)                   
	{
        int threadNumber = omp_get_thread_num();
        long double acum = 0;
        long double baseIntervalo;
        long double x = interval * (NUMBER_OF_INTERVALS / NUMBER_OF_THREADS) * threadNumber;
        long double fdx;

        baseIntervalo = 1.0 / NUMBER_OF_INTERVALS;

        for (long i = 0; i < (NUMBER_OF_INTERVALS / NUMBER_OF_THREADS); i++) {

            fdx = 4.0 / (1.0 + x * x);

            acum += (fdx * baseIntervalo);

            x += baseIntervalo;
        }

        resultados[threadNumber] = acum;
	}

    endT = clock();
    double resultado = 0;

    for (int i = 0; i < NUMBER_OF_THREADS; i++) {
        resultado += resultados[i];
    }

    printf("Resultado = %20.18lf (%ld)\n", resultado, endT - start);
	return 0;
}