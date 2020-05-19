#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

#define NUMBER_OF_THREADS 1024
#define NUMBER_OF_INTERVALS 1000000000


//long double interval = 1.0 / NUMBER_OF_INTERVALS;


__global__ void calcular(double* resultados) {

    int threadNumber = threadIdx.x;

    double acum = 1;
    double baseIntervalo = 1.0 / NUMBER_OF_INTERVALS;
    int timesExec = NUMBER_OF_INTERVALS / NUMBER_OF_THREADS;

    //Mitades de rectángulos
    double x = baseIntervalo * (NUMBER_OF_INTERVALS / NUMBER_OF_THREADS) * threadNumber + (baseIntervalo / 2);
    double fdx;

    printf("%d\n", timesExec);

    for (int i = 0; i < 500000; i++) {

        fdx = 4.0 / (1.0 + x * x);

        acum += fdx;

        x += baseIntervalo;
        /*x = (i + 0.5) * baseIntervalo;
        fdx = 4 / (1 + x * x);
        acum += fdx;*/

    }
    /*for (int i = 0; i < NUMBER_OF_THREADS; i++) {
        acum += i;
    }*/
    acum *= baseIntervalo;
   
    //printf("THREAD\n");

    resultados[threadNumber] = acum;
   
}

int main() {
    //int sharedVar = 6;
    clock_t start, endT;
     double* h_resultados;
     double* d_resultados;

    int size = NUMBER_OF_THREADS * sizeof( double);

    h_resultados = ( double*)malloc(size);

    cudaMalloc((void**)&d_resultados, size);

    for (int i = 0; i < NUMBER_OF_THREADS; i++) {
        h_resultados[i] = 0;
    }

    cudaMemcpy(d_resultados, h_resultados, size, cudaMemcpyDeviceToHost);

    start = clock();

    calcular << < 1, NUMBER_OF_THREADS >> > (d_resultados);
    cudaDeviceSynchronize();
    endT = clock();



    cudaMemcpy(h_resultados, d_resultados, size, cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        //Print the CUDA error message
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
     double resultado = 0;

    for (int i = 0; i < NUMBER_OF_THREADS; i++) {
        //cout << resultado << endl;
        resultado += h_resultados[i];
    }

    cout << "resultado = " << resultado << " (" << endT - start << ")" << endl;

    //return 0;
}