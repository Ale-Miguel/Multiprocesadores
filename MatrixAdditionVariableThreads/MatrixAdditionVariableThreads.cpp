// MatrixAdditionThreads.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//

/*
Alejandro Miguel Sánchez Mora
A01272385

Modify the MulktiplyMatrixThreads code in order to ask the user how many cores he intended to use for its execution, submit your source 
code, report the execution time and include a screen shot showing that your runtime adaptable parallel version executed
correctly.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

const int dimension = 10000;

int max_threads = 1;

int* A = NULL;
int* B = NULL;
int* C = NULL;

DWORD WINAPI sumar(LPVOID inicio) {


	int* inicioSuma = (int*)inicio;
	int limite = *inicioSuma + (dimension / max_threads) + (dimension % max_threads);

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

	HANDLE* threadsArray;
	int* startsIn;

	printf("How many threads do you want to use?\n");
	scanf_s("%d", &max_threads);

	threadsArray = (HANDLE*)malloc(sizeof(HANDLE) * max_threads);
	startsIn = (int*)malloc(sizeof(int) * max_threads);



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

	int range = dimension / max_threads;

	start = clock();

	for (int i = 0; i < max_threads; i++) {
		*(startsIn + i) = i * range;
		*(threadsArray + i) = CreateThread(NULL, 0, sumar, (startsIn + i), 0, NULL);

	}

	WaitForMultipleObjects(max_threads, threadsArray, true, INFINITE);

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
		printf("Results verified!!! (%ld) with %d threads\n", (long)(end - start), max_threads);
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
