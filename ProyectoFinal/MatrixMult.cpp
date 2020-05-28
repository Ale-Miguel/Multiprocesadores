/*
Programa Final - Multiplicación de Matrices

Por:
Alejandro Miguel Sánchez Mora - A01272385
Jessica Tovar Saucedo - A00818101

Procesamiento:
	• El programa hará 15 veces el cálculo de la multiplicación de la matriz A x matriz B.
	• Manejar todas las operaciones en DOBLE precisión.
		o Serial
			- Las primeras 5 veces se usará un código puramente serial (sin optimizaciones).
			- Se guardará el tiempo que tarde en ejecutar cada ejecución. Sólo considerar el
			  tiempo que tarde en ejecutar la multiplicación. El tiempo no deberá incluir ni la
			  carga de los datos ni la escritura de la matriz resultante.
			- Se escribirá la matriz resultante a un archivo llamado matrizC.txt
		o Paralelo 1
			- Usar código paralelizado de entre los siguientes: Ensamblador, Intrínsecas,
			  Auto-Vectorización, OpenMP u OpenCL.
			- Ejecutar 5 veces el código guardando el tiempo que tarda cada ejecución.
			- Comparar la matriz resultante vs. la obtenida en la parte serial.
			- Imprimir en consola si son iguales.
		o Paralelo 2.
			- Lo mismo que en paralelo 1, pero escoger otra herramienta para paralelizar.
Validaciones:
• El programa validará:
	o Que la cantidad de elementos leídos de los archivos permita construir la matriz del
	  tamaño solicitado.
	o Que la multiplicación de ambas matrices se pueda realizar.
	o Que el operativo haya entregado suficiente memoria para guardar los arreglos.
*/

#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <fstream>

using namespace std;

#define MATRIX_A_FILE_NAME "matrixA.txt"
#define MATRIX_B_FILE_NAME "matrixB.txt"
#define MATRIX_C_FILE_NAME "matrixC.txt"

double **matrixA;
double **matrixB;
double **matrixCseq;		//Apuntador que guarda la matriz resultante del código secuencial
double **matrixCparalel;	//Apuntador que guarda la matriz resultante del código en paralelo

int columnsA, rowsA;
int columnsB, rowsB;
int columnsC, rowsC;

//Función para estandarizar el formato de mensaje de error
void sendErrorMessage(string message) {
	cout << "ERROR: " << message << endl;
	exit(0);
}

void sequentialCode(double**& matrix) {

	matrix = (double**)malloc(rowsC * sizeof(double*));
	if (!matrix) {
		sendErrorMessage("No se pudo asignar memoria para matriz C en código secuencial");
	}

	for (int i = 0; i < rowsC; i++) {
		matrix[i] = (double*)malloc(columnsC * sizeof(double));

		if (!matrix[i]) {
			sendErrorMessage("No se pudo asignar memoria para matriz C en código secuencial");
		}
		for (int j = 0; j < columnsC; j++) {
			matrix[i][j] = 0;
		}
	}

	for (int i = 0; i < rowsC; i++) {
		for (int j = 0; j < columnsC; j++) {
			for (int k = 0; k < rowsB; k++) {
				matrix[i][j] += matrixA[i][k] * matrixB[k][j];
			}
		}
	}
}

//Función para llenar las matices cuya información se obtiene de un archivo
void fillMatrix(double** &matrix, ifstream &matrixFile, int rows, int columns) {
	matrix = (double**)malloc(rows * sizeof(double*));

	if (!matrix) {
		sendErrorMessage("No se pudo asignar la memoria suficiente para la matriz");
	}

	double line;	//Variable que va guardando el valor de cada línea del archivo

	for (int i = 0; i < rows; i++) {

		matrix[i] = (double*)malloc(columns * sizeof(double));

		if (!matrix[i]) {
			sendErrorMessage("No se pudo asignar la memoria suficiente para la matriz");
		}

		for (int j = 0; j < columns; j++) {

			if (matrixFile >> line) {
				matrix[i][j] = line;
			}
			else {
				sendErrorMessage("Elementos en archivo no suficientes para llenar la matriz");
			}
		}
	}
}


int main() {

	ifstream matrixAFile(MATRIX_A_FILE_NAME);	//Archivo de matriz A
	ifstream matrixBFile(MATRIX_B_FILE_NAME);	//Archivo de matriz B

	ofstream matrixCFile(MATRIX_C_FILE_NAME);	//Archivo de matriz resultante C

	//Se verifica que los archivos se puedan abrir
	if (!matrixAFile.is_open() || !matrixBFile.is_open() || !matrixCFile.is_open()) {
		sendErrorMessage("No se pudo abrir el archivo");
	}

	cout << "Cantidad de filas de Matriz A: ";
	cin >> rowsA;

	cout << "Cantidad de columnas de Matriz A: ";
	cin >> columnsA;

	cout << "Cantidad de filas de Matriz B: ";
	cin >> rowsB;

	cout << "Cantidad de columnas de Matriz B: ";
	cin >> columnsB;

	//Se valida que la multiplicación se pueda hacer
	if (rowsA != columnsB) {
		sendErrorMessage("No se puede hacer la multiplicacion de matrices");
	}
	
	rowsC = rowsA;
	columnsC = columnsB;

	fillMatrix(matrixA, matrixAFile, rowsA, columnsA);
	fillMatrix(matrixB, matrixBFile, rowsB, columnsB);

	sequentialCode(matrixCseq);

	cout << "Matrix A" << endl;
	for (int i = 0; i < rowsA; i++) {
		for (int j = 0; j < columnsA; j++) {
			cout << matrixA[i][j] << "\t";
		}

		cout << endl;
	}

	cout << "Matrix B" << endl;
	for (int i = 0; i < rowsB; i++) {
		for (int j = 0; j < columnsB; j++) {
			cout << matrixB[i][j] << "\t";
		}

		cout << endl;
	}

	cout << "Matrix C sequential" << endl;
	for (int i = 0; i < rowsC; i++) {
		for (int j = 0; j < columnsC; j++) {
			cout << matrixCseq[i][j] << "\t";
		}

		cout << endl;
	}

	matrixAFile.close();
	matrixBFile.close();
	matrixCFile.close();

	free(matrixA);
	free(matrixB);
	free(matrixCseq);
	
	return 0;
}