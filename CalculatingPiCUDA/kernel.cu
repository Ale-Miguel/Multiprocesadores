#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

#define MAX_ITERATIONS_PER_THREAD 500000    //Cantidad máxima de iteraciones que corre cada thread
#define MAX_THREADS_PER_BLOCK 1024
#define NUMBER_OF_INTERVALS 1000000000




__global__ void calcular(double* resultados) {

    int threadNumber = threadIdx.x;
    int blockNumber = blockIdx.x;

    //Se verifica si este thread se debe de ejecutar
    if (NUMBER_OF_INTERVALS / MAX_ITERATIONS_PER_THREAD < MAX_THREADS_PER_BLOCK * blockNumber  + threadNumber) {
        return;
    }
    double acum = 1;
    double baseIntervalo = 1.0 / NUMBER_OF_INTERVALS;
    

    //Mitades de rectángulos
    double x = baseIntervalo * (NUMBER_OF_INTERVALS / MAX_THREADS_PER_BLOCK) * threadNumber + (baseIntervalo / 2);
    double fdx;

    

    for (int i = 0; i < MAX_ITERATIONS_PER_THREAD; i++) {
        
        fdx = 4.0 / (1.0 + x * x);

        acum += fdx;

        x += baseIntervalo;

    }

    acum *= baseIntervalo;

    resultados[MAX_THREADS_PER_BLOCK * blockNumber + threadNumber] = acum;
   
}

int main() {
    //int sharedVar = 6;
    clock_t start, endT;
     double* h_resultados;
     double* d_resultados;

     //Se calcula la cantidad de threads necesarios para calcular la aproximación con 
     //la cantidad de intervalos dados
     int numberOfThreads = NUMBER_OF_INTERVALS / MAX_ITERATIONS_PER_THREAD;

     int numberOfBlocks = numberOfThreads / MAX_THREADS_PER_BLOCK  + 1;

    int size = numberOfThreads * sizeof( double);
    printf("SIZE %d\n", numberOfBlocks);

    h_resultados = ( double*)malloc(size);

    cudaMalloc((void**)&d_resultados, size);

    for (int i = 0; i < numberOfThreads; i++) {
        h_resultados[i] = 0;
    }


    //Copiar de host to device del arreglo de resultados
    cudaMemcpy(d_resultados, h_resultados, size, cudaMemcpyHostToDevice);

    start = clock();

    calcular << < numberOfBlocks, MAX_THREADS_PER_BLOCK >> > (d_resultados);
    cudaDeviceSynchronize();

    endT = clock();

    cudaMemcpy(h_resultados, d_resultados, size, cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        //Print the CUDA error message
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
     double resultado = 0;
    
    for (int i = 0; i < numberOfThreads; i++) {
        //cout << resultado << endl;
        resultado += h_resultados[i];
    }

    //cout << "resultado = " << resultado << " (" << endT - start << ")" << endl;
    printf("Result = %20.18lf (%ld)\n", resultado, endT - start);
    free(h_resultados);
    cudaFree(d_resultados);

    //return 0;
}