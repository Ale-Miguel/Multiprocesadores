/*
Alejandro Miguel S�nchez Mora
A01272385

Aproxximaci�n de Pi utilizando threads

C�digo fuente de la tarea 4
*/

#include <iostream>
#include <stdio.h>
#include <Windows.h>
#include <time.h>

#define NUMBER_OF_THREADS 32
#define NUMBER_OF_INTERVALS  1000000000


double resultado = 0;

CRITICAL_SECTION cs;
//Funci�n que ejecuta el thread
DWORD WINAPI calcular(LPVOID inicio) {
     
    long double* param = (long double*)inicio;

    long double acum = 0;
    long double baseIntervalo;
    long double x = *param;
    long double fdx;

    baseIntervalo = 1.0 / NUMBER_OF_INTERVALS;

    for (long i = 0; i < (NUMBER_OF_INTERVALS / NUMBER_OF_THREADS); i++) {

        fdx = 4.0 / (1.0 + x * x);

        acum += (fdx * baseIntervalo);

        x += baseIntervalo;
    }

    //Se accede a la secci�n cr�tica
    EnterCriticalSection(&cs);
    //Secci�n cr�tica
    resultado += acum;

    //Se sale de la secci�n cr�tica
    LeaveCriticalSection(&cs);

    return 0;
}


clock_t start, end;



int main() {

    long i;

    start = clock();

    HANDLE threadsArray[NUMBER_OF_THREADS];
    long double startingValues[NUMBER_OF_THREADS];

    long double interval = 1.0 / NUMBER_OF_INTERVALS;
    long double startsIn = 0;

    //Se inicializa la secci�n cr�tica
    InitializeCriticalSection(&cs);

    for (i = 0; i < NUMBER_OF_THREADS; i++) {
        startingValues[i] = startsIn;
        threadsArray[i] = CreateThread(NULL, 0, calcular, &startingValues[i], 0, NULL);
        startsIn += interval * (NUMBER_OF_INTERVALS / NUMBER_OF_THREADS);
    }

    WaitForMultipleObjects(NUMBER_OF_THREADS, threadsArray, true, INFINITE);

    //Se elimina la secci�n cr�tica
    DeleteCriticalSection(&cs);

    end = clock();

    printf("Resultado = %20.18lf (%ld)\n", resultado, end - start);

    return 0;
}