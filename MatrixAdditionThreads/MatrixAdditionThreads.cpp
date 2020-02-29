// MatrixAdditionThreads.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//

/*
Alejandro Miguel Sánchez Mora
A01272385

 Consider the 'Matrix Addition' algorithm (c code uploaded in Canvas); modify it for be multithreading and respond to the following questions:

	Report the execution time (in clock cycles) for the baseline (reference) sequential algorithm.
	Report the execution time for a 2, 4, 8 and 32 thread version.
	Report your source code and include a screen shots showing that your runtime adaptable parallel version executed correctly.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

#define MAX_THREADS 8

const int dimension = 10000;

int* A = NULL;
int* B = NULL;
int* C = NULL;

DWORD WINAPI sumar(LPVOID inicio) {

		
	int *inicioSuma = (int*)inicio;
	int limite = *inicioSuma + (dimension / MAX_THREADS) + (dimension % MAX_THREADS);

	for (int row = 0; row < dimension; row++) {
		for (int col = *inicioSuma; col < limite; col++) {
			if (A && B && C)				
				*(C + row * dimension + col) = *(A + row * dimension + col) + *(B + row * dimension + col);
		}
	}

	return 0;
}

int main() {
	int row, col;
		
	int result = 1;
	
	time_t start, end;

	size_t datasize = sizeof(int) * dimension * dimension;

	HANDLE *threadsArray;
	int *startsIn;
	
	threadsArray = (HANDLE*)malloc(sizeof(HANDLE) * MAX_THREADS);
	startsIn = (int*)malloc(sizeof(int) * MAX_THREADS);

	A = (int*)malloc(datasize);
	B = (int*)malloc(datasize);
	C = (int*)malloc(datasize);

	for (row = 0; row < dimension; row++) {
		for (col = 0; col < dimension; col++) {
			if (A)
				*(A + row * dimension + col) = row + col;
			if (B)
				*(B + row * dimension + col) = row + col;
		}
	}

	int range = dimension / MAX_THREADS;

	start = clock();

	for (int i = 0; i < MAX_THREADS; i++) {
		*(startsIn + i) = i * range;
		*(threadsArray + i) = CreateThread(NULL, 0, sumar, (startsIn + i), 0, NULL);

	}

	WaitForMultipleObjects(MAX_THREADS, threadsArray, true, INFINITE);

	end = clock();

	for (row = 0; row < dimension; row++) {
		for (col = 0; col < dimension; col++) {
			if (C) {
				if (*(C + row * dimension + col) != ((row + col) * 2)) {
					result = 0;
					break;
				}
			}
		}
	}

	if (result) {
		printf("Results verified!!! (%ld) with %d threads\n", (long)(end - start), MAX_THREADS);
	}
	else {
		printf("Wrong results!!!\n");
	}

	free(A);
	free(B);
	free(C);

	free(threadsArray);
	free(startsIn);


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
