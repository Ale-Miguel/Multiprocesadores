// Aproximacion_Pi.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//

#include <iostream>
#include <stdio.h>
#include <Windows.h>
#include <time.h>

#define NUMBER_OF_THREADS 32
#define NUMBER_OF_INTERVALS  1000000000


double resultado = 0;

CRITICAL_SECTION cs;
//Función que ejecuta el thread
DWORD WINAPI calcular(LPVOID inicio) {

    long double* param = (long double*)inicio;
   
    long double acum = 0;
    long double baseIntervalo;
    long double x = * param;
    long double fdx;
   
    baseIntervalo = 1.0 / NUMBER_OF_INTERVALS;

    for (long i = 0; i < (NUMBER_OF_INTERVALS / NUMBER_OF_THREADS); i++) {
       
        fdx = 4.0 / (1.0 + x * x);
       
        acum += (fdx * baseIntervalo);
       
        x += baseIntervalo;
    }
    
    //Se accede a la sección crítica
    EnterCriticalSection(&cs);
        //Sección crítica
        resultado += acum;
    
    //Se sale de la sección crítica
    LeaveCriticalSection(&cs);

    return 0;
}


clock_t start, end;

void aproxPiSecuencial() {

    start = clock();

    long cantidadIntervalos = 1000000000;
    double baseIntervalo;
    double fdx;
    double acum = 0;
    double x;
    long i;
    baseIntervalo = 1.0 / cantidadIntervalos;

    for (i = 0, x = 0.0; i < cantidadIntervalos; i++) {
        fdx = 4 / (1 + x * x);
        acum = acum + (fdx * baseIntervalo);
        x = x + baseIntervalo;
    }
    end = clock();

    printf("Resultado = %20.18lf (%ld)\n", acum, end - start);
}
 

int main() {

    long i; 
    
    start = clock();

    HANDLE threadsArray[NUMBER_OF_THREADS];
    long double startingValues[NUMBER_OF_THREADS];

    long double interval = 1.0 / NUMBER_OF_INTERVALS;
    long double startsIn = 0;

    //Se inicializa la sección crítica
    InitializeCriticalSection(&cs);

    for (i = 0; i < NUMBER_OF_THREADS; i++) { 
        startingValues[i] = startsIn;
        threadsArray[i] = CreateThread(NULL, 0, calcular, &startingValues[i], 0, NULL);
        startsIn += interval * (NUMBER_OF_INTERVALS / NUMBER_OF_THREADS);
    }
      
    WaitForMultipleObjects(NUMBER_OF_THREADS, threadsArray, true, INFINITE);

    //Se elimina la sección crítica
    DeleteCriticalSection(&cs);

    end = clock();

    printf("Resultado = %20.18lf (%ld)\n", resultado, end - start);

    return 0;
}

// Ejecutar programa: Ctrl + F5 o menú Depurar > Iniciar sin depurar
// Depurar programa: F5 o menú Depurar > Iniciar depuración

// Sugerencias para primeros pasos: 1. Use la ventana del Explorador de soluciones para agregar y administrar archivos
//   2. Use la ventana de Team Explorer para conectar con el control de código fuente
//   3. Use la ventana de salida para ver la salida de compilación y otros mensajes
//   4. Use la ventana Lista de errores para ver los errores
//   5. Vaya a Proyecto > Agregar nuevo elemento para crear nuevos archivos de código, o a Proyecto > Agregar elemento existente para agregar archivos de código existentes al proyecto
//   6. En el futuro, para volver a abrir este proyecto, vaya a Archivo > Abrir > Proyecto y seleccione el archivo .sln
