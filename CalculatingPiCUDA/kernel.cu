/*
Alejandro Miguel Sánchez Mora
A01272385

Jessica Tovar Saucedo
A00818101
*/
#include <stdio.h>
#include <iostream>
#include <time.h>

using namespace std;

#define MAX_ITERATIONS_PER_THREAD 100000    //Cantidad óptima de iteraciones que corre cada thread (máximo es 500000)
#define MAX_THREADS_PER_BLOCK 1024          //Cantidad máxima de threads que se ejecutan por bloque (máximo 1024)
#define NUMBER_OF_INTERVALS 1000000000


__global__ void calcular(double* resultados) {

    int threadNumber = threadIdx.x;
    int blockNumber = blockIdx.x;

    //Variable que guarda el número de thread respecto a 0 - CantidadTotalDeThreads (sin importar en qué bloque está)
    int trueThreadNumber = MAX_THREADS_PER_BLOCK * blockNumber + threadNumber;

    //Se verifica si este thread se debe de ejecutar, por si la división de intervalos / iteraciones no es exacta
    if (NUMBER_OF_INTERVALS / MAX_ITERATIONS_PER_THREAD < trueThreadNumber) {
        
        return;
    }

    double acum = 0;
    double baseIntervalo = 1.0 / NUMBER_OF_INTERVALS;
    
    //Mitades de rectángulos
    double x = baseIntervalo * MAX_ITERATIONS_PER_THREAD * trueThreadNumber + (baseIntervalo / 2);  //Se establece desde dónde se inicia x en el algoritmo
    double fdx;

    for (int i = 0; i < MAX_ITERATIONS_PER_THREAD; i++) {
        
        fdx = 4.0 / (1.0 + x * x);

        acum += fdx;

        x += baseIntervalo;

    }

    //acum *= baseIntervalo;

    resultados[trueThreadNumber] = acum;
}

int main() {

    clock_t start, endT;

    double* h_resultados;
    double* d_resultados;

    //Se calcula la cantidad de threads necesarios para calcular la aproximación con 
    //la cantidad de intervalos dados
    int numberOfThreads = NUMBER_OF_INTERVALS / MAX_ITERATIONS_PER_THREAD;

    //+1 por que con que la fracción no sea exacta, se necesita de otro bloque
    int numberOfBlocks = numberOfThreads / MAX_THREADS_PER_BLOCK  + 1;

    int size = numberOfThreads * sizeof( double);

    h_resultados = ( double*)malloc(size);

    cudaMalloc((void**)&d_resultados, size);

    //Se inicializa el arreglo de resultados para evitar datos erróneos en caso de alguna falla
    for (int i = 0; i < numberOfThreads; i++) {
        h_resultados[i] = 0;
    }

    //Copiar de host to device del arreglo de resultados
    cudaMemcpy(d_resultados, h_resultados, size, cudaMemcpyHostToDevice);

    start = clock();

    calcular << < numberOfBlocks, MAX_THREADS_PER_BLOCK >> > (d_resultados);
    cudaDeviceSynchronize();

    endT = clock();

    //Copiar del device al host
    cudaMemcpy(h_resultados, d_resultados, size, cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        //Print the CUDA error message
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    
    double resultado = 0;
    
    for (int i = 0; i < numberOfThreads; i++) {

        resultado += h_resultados[i];
    }

    resultado *= (1.0 / NUMBER_OF_INTERVALS);

    printf("Result using CUDA = %20.18lf (%ld)\n", resultado, endT - start);

    free(h_resultados);

    cudaFree(d_resultados);

}

/*
//Código secuencial
#include <stdio.h>
#include <time.h>

long cantidadIntervalos = 1000000000;
double baseIntervalo;
double fdx;
double acum = 0;
clock_t start, end;

void main() {
   double x=0;
   long i;
   baseIntervalo = 1.0 / cantidadIntervalos;
   start = clock();
   for (i = 0; i < cantidadIntervalos; i++) {
      x = (i+0.5)*baseIntervalo;
      fdx = 4 / (1 + x * x);
      acum += fdx;
   }
   acum *= baseIntervalo;
   end = clock();

   printf("Result = %20.18lf (%ld)\n", acum, end - start);
}
*/