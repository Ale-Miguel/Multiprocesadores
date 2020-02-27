// Aproximation_Pi_Event.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//

/*
Alejandro Miguel Sánchez Mora
A01272385

Use the multithreading version of the algorithm to get an approximation to constant PI that we complete class.
Apply now two threads (two different routines), remove the CRITICAL SECTION and protect the global accumulator with an EVENT.
*/

#include <iostream>
#include <Windows.h>
#include <time.h>

using namespace std;

#define NUMBER_OF_THREADS 2
#define NUMBER_OF_INTERVALS 1000000000

double resultado = 0;

//Handle del evento
HANDLE threadEvent;

clock_t startClock, endClock;

DWORD WINAPI calcularPrimeraParte(LPVOID inicio) {

    cout << "Empieza primera parte" << endl;
    
    //Se inica el evento en no evento
    ResetEvent(threadEvent);

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

    resultado += acum;

    cout << "Terminando primera parte" << endl;

    //Se manda el evento de que este thread ha terminado
    SetEvent(threadEvent);

    cout << "Se mando evento" << endl;

    return 0;
}

DWORD WINAPI calcularSegundaParte(LPVOID inicio) {

    //Se espera a que el evento sea activado para empezar la ejecución de la función
    WaitForSingleObject(threadEvent, INFINITE);

    cout << "Empezando segunda parte" << endl;

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


    resultado += acum;

    cout << "Termina segunda parte" << endl;

    return 0;
}


int main(){

    HANDLE threadsArray[NUMBER_OF_THREADS];

    double startsIn1 = 0.0;
    double startsIn2 = 0.5;

    //Se crea el evento
    threadEvent = CreateEvent(NULL, false, false, NULL);
    
    //Se inicializa el evento en bajo para asegurar que el evento no esté levantado antes de ejecutar los threads
    ResetEvent(threadEvent);

    startClock = clock();

    threadsArray[0] = CreateThread(NULL, 0, calcularPrimeraParte, &startsIn1, 0, NULL);
  
    threadsArray[1] = CreateThread(NULL, 0, calcularSegundaParte, &startsIn2, 0, NULL);

    WaitForMultipleObjects(NUMBER_OF_THREADS, threadsArray, true, INFINITE);

    endClock = clock();

    cout << "Aprox Pi = " << resultado << " Calculado en " << endClock - startClock << " ciclos de reloj." << endl;

    return 0;
}
