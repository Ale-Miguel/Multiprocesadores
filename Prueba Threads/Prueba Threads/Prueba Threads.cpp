// Prueba Threads.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//

#include <stdio.h>
#include <Windows.h>

#define NUMBER_OF_THREADS 4

//Función que ejecuta el thread
DWORD WINAPI helloFunc(LPVOID pArg) {

    int* param = (int*)pArg;

    printf("HELLO THREAD %d\n", *param);

    return 0;
}


int main(){

    int i = 0;

    //Se crea el handle del thread que guarda la referencia hacia el thread
   //HANDLE hThread;

    //Se crea el thread
   //hThread = CreateThread(NULL, 0, helloFunc, &i, 0, NULL);

    //Se espera  a que el thread termine para seguir con la ejecución del código
   //WaitForSingleObject(hThread, INFINITE);

    //Arreglo de handles para tener varios threads
    HANDLE threadArray[NUMBER_OF_THREADS];

    //Arreglo auxiliar para guardar el número de thread
    int threadNumber[NUMBER_OF_THREADS];

    //Ciclo para lanzar a ejecutar los threads
    for (i = 0; i < NUMBER_OF_THREADS; i++) {

        threadNumber[i] = i;
        threadArray[i] = CreateThread(NULL, 0, helloFunc, &threadNumber[i], 0, NULL);
    }
   
    //Se espera a que todos los threads terminen para continuar con la ejecución del programa
    WaitForMultipleObjects(NUMBER_OF_THREADS, threadArray, true, INFINITE);

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
