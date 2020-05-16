/*
Alejandro Miguel Sánchez Mora
A01272385

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <intrin.h>
#include <fstream>
#include <string>

#define FILE_NAME "numbers.txt"
#define NUMBER_OF_RUNS 100000
#define ELEMENTS 8


using namespace std;


time_t start, endT;

float* numbers = NULL; //Apuntador a donde se almacena la lista de números

int items = 0;	//Contador que indica la cantidad de números que hay

long scalarTime = 0;
long intrinsicsTime = 0;
long autoVecTime = 0;

float scalarAvg = 0;
float intrinsicsAvg = 0;
float autoVecAvg = 0;

int incrementMult = 1;

void scalarOperations() {
	float avg = 0;
	start = clock();
#pragma loop(no_vector)
	for (int i = 0; i < items; i++) {
		avg += *(numbers + i);
	}

	avg /= items;

	endT = clock();

	scalarAvg = avg;
	scalarTime += (long)(endT - start);

}

void intrinsics() {
	float avg = 0;

	__m128 avgV, a, b;
	__m128 div;

	float* resultado = NULL;
	
	resultado = (float*)malloc(sizeof(float) * 4);

	start = clock();
#pragma loop(no_vector)
	for(int i = 0,  max = incrementMult; i <  max; i++) {
		
		a = _mm_loadu_ps(numbers + i * 8);
		b = _mm_loadu_ps(numbers + i * 8 + 4);

		//Se usan 3 sumas horizontales para sumar todos los números,
		//A la tercera suma, la respuesta se encuentra en cualquier elemento el total de la suma
		avgV = _mm_hadd_ps(a, b);
		avgV = _mm_hadd_ps(avgV, avgV);
		avgV = _mm_hadd_ps(avgV, avgV);
		

		_mm_store_ps(resultado, avgV);

		avg += resultado[0] 
		
	}

	avg /= items;

	endT = clock();

	intrinsicsAvg = avg;
	intrinsicsTime = (long)(endT - start);
}



void autoVectorization() {
	float avg = 0;
	start = clock();

	float aux;

	for (int i = 0; i < items; i++) {
		aux = numbers[i];

		avg = avg + aux;
	}

	avg /= items;

	endT = clock();

	autoVecAvg = avg;
	autoVecTime += (long)(endT - start);

}


int main() {

	float value;
	string aux;

	ifstream inputFile(FILE_NAME);

	size_t alignment = 16;
	size_t datasize = sizeof(float) * ELEMENTS;

	numbers = (float*)_aligned_malloc(datasize, alignment);

	//Si el archivo no se pudo abrir o no se encontró
	if (!inputFile.is_open()) {
		printf("ERROR: File not found\n");
		return 0;
	}

	items = 0;

	while (inputFile >> value) {
		//printf("%f\n", value);

		//Si se llena el espacio reservado, se crea reserva más espacio
		if (items == (incrementMult * ELEMENTS) - 1) {

			incrementMult++;

			float *aux2;	//Apuntador para liberar el espacio que ocupaba numbers

			float *aux = (float*)_aligned_malloc(incrementMult * ELEMENTS * sizeof(float), alignment);

			memcpy(aux, numbers, sizeof(float)*(incrementMult-1) * ELEMENTS);

			aux2 = numbers;

			numbers = aux;

			_aligned_free(aux2);


		}

		*(numbers + items) = value;

		items++;
	}


	//Se rellena con 0 las casillas que no se usaron
	//Padding
	for (int i = items ; i < incrementMult * ELEMENTS; i++) {
		*(numbers + i) = 0;
	}

	for (int i = 0; i < items; i++) {
		printf("%f\n", *(numbers + i));
	}
	
	//Scalar operation loop
	for (int i = 0; i < NUMBER_OF_RUNS; i++) {
		scalarOperations();
	}

	//Intrinsics loop
	for (int i = 0; i < NUMBER_OF_RUNS; i++) {
		intrinsics();
	}

	//Auto-vectorization loop
	for (int i = 0; i < NUMBER_OF_RUNS; i++) {
		autoVectorization();
	}

	printf("The file had %d floatingpoint numbers.\nEach function was called %ld times.\n", items, NUMBER_OF_RUNS);

	printf("\nUsing scalcar operators: \n");
	printf("-The average is: %f\n", scalarAvg);
	printf("-Time used: %ld\n", scalarTime);

	printf("\nUsing SSE/AVX Intrinsics: \n");
	printf("-The average is: %f\n", intrinsicsAvg);
	printf("-Time used: %ld\n", intrinsicsTime);

	printf("\nUsing Auto-vectorization: \n");
	printf("-The average is: %f\n", autoVecAvg);
	printf("-Time used: %ld\n", autoVecTime);

	_aligned_free(numbers);

	return 0;
}

/*
CONCLUSIONS
La autovectorización si ayuda a mejorar la eficiencia del código. En esta tarea, se utilizó para hacer más eficiente 
la versión escalar. Si bien no lo mejora mucho, si es de gran ayuda para hacer más eficiente el programa que puede
escribir cualquier persona con conocimientos básicos de programación. Usando intrínsicas a comparación de la escalar
si fue una mejora significativa, ya que haciendo 3 operaciones de suma con intrínsicas, obtenía el resultado de hacer
8 sumas de manera escalar, lo que se traduce en una gran reducción en tiempo de ejecución. La autovectorización no es
muy inteligente por ahora, ya que obtuvo un rendimiento peor que las intrínsicas al momento de calcular el promedio,
sin embargo, en ocaciones si nos puede ser útil en el caso en que no se pueda encontrar una manera de eficientar el código.
*/