/*
Alejandro Miguel Sánchez Mora
A012722385

1.- Create a program that do the following:

-Define 6 Arrays of FLOATS.
-16 elements each.
-Array A = {10.0, 11.0, 12.0… 25.0}
-Array  B= {1.0, 2.0, 3.0… 16.0}
-Array C, D, E & F are zeroed at the begining.
-Using OMP, do task decomposition:
-C[i] = A[i] + B[i];
-D[i] = A[i] - B[i];
-E[i] = A[i] * B[i];
-F[i] = A[i] / B[i];
-Print the 4 lines with the results. One line per operation (the 16 elements in the same line separated by spaces or commas).

*/
#include <stdio.h>
#include <omp.h>

#define NUMBER_OF_ELEMENTS 16

float A[NUMBER_OF_ELEMENTS];
float B[NUMBER_OF_ELEMENTS];
float C[NUMBER_OF_ELEMENTS];

float D[NUMBER_OF_ELEMENTS];
float E[NUMBER_OF_ELEMENTS];
float F[NUMBER_OF_ELEMENTS];

void addition() {
	for (int i = 0; i < NUMBER_OF_ELEMENTS; i++) {
		C[i] = A[i] + B[i];
	}
}

void substraction() {
	for (int i = 0; i < NUMBER_OF_ELEMENTS; i++) {
		D[i] = A[i] - B[i];
	}
}

void multiplication() {
	for (int i = 0; i < NUMBER_OF_ELEMENTS; i++) {
		E[i] = A[i] * B[i];
	}
}

void division() {
	for (int i = 0; i < NUMBER_OF_ELEMENTS; i++) {
		F[i] = A[i] / B[i];
	}
}

int main() {

	//Inicializando los arreglos
	for (int i = 0; i < NUMBER_OF_ELEMENTS; i++) {
		A[i] = i + 10;
		B[i] = i + 1;
		C[i] = D[i] = E[i] = F[i] = 0;
	}

	#pragma omp parallel sections shared(A,B) private(C,D,E,F)
	{
		#pragma omp section
		addition();

		#pragma omp section
		substraction();

		#pragma omp section
		multiplication();

		#pragma omp section
		division();
	}

	printf("Array C = ");
	for (int i = 0; i < NUMBER_OF_ELEMENTS; i++) {
		printf("%f, ", C[i]);
	}

	printf("\nArray D = ");
	for (int i = 0; i < NUMBER_OF_ELEMENTS; i++) {
		printf("%f, ", D[i]);
	}

	printf("\nArray E = ");
	for (int i = 0; i < NUMBER_OF_ELEMENTS; i++) {
		printf("%f, ", E[i]);
	}

	printf("\nArray F = ");
	for (int i = 0; i < NUMBER_OF_ELEMENTS; i++) {
		printf("%f, ", F[i]);
	}
}