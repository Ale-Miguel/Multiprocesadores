/*
Programa Final - Multiplicaci�n de Matrices

Por:
Alejandro Miguel S�nchez Mora - A01272385
Jessica Tovar

Procesamiento:
	� El programa har� 15 veces el c�lculo de la multiplicaci�n de la matriz A x matriz B.
	� Manejar todas las operaciones en DOBLE precisi�n.
		o Serial
			- Las primeras 5 veces se usar� un c�digo puramente serial (sin optimizaciones).
			- Se guardar� el tiempo que tarde en ejecutar cada ejecuci�n. S�lo considerar el
			  tiempo que tarde en ejecutar la multiplicaci�n. El tiempo no deber� incluir ni la
			  carga de los datos ni la escritura de la matriz resultante.
			- Se escribir� la matriz resultante a un archivo llamado matrizC.txt
		o Paralelo 1
			- Usar c�digo paralelizado de entre los siguientes: Ensamblador, Intr�nsecas,
			  Auto-Vectorizaci�n, OpenMP u OpenCL.
			- Ejecutar 5 veces el c�digo guardando el tiempo que tarda cada ejecuci�n.
			- Comparar la matriz resultante vs. la obtenida en la parte serial.
			- Imprimir en consola si son iguales.
		o Paralelo 2.
			- Lo mismo que en paralelo 1, pero escoger otra herramienta para paralelizar.
Validaciones:
� El programa validar�:
	o Que la cantidad de elementos le�dos de los archivos permita construir la matriz del
	  tama�o solicitado.
	o Que la multiplicaci�n de ambas matrices se pueda realizar.
	o Que el operativo haya entregado suficiente memoria para guardar los arreglos.
*/

#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <fstream>

using namespace std;

#define MATRIX_A_FILE_NAME "test.txt"
#define MATRIX_B_FILE_NAME "test.txt"
#define MATRIX_C_FILE_NAME "test.txt"

double **matrixA;
double **matrixB;
double **matrixC;

int columnsA, rowsA;
int columnsB, rowsB;
int columnsC, rowsC;


int main() {

	ifstream matrixAFile(MATRIX_A_FILE_NAME);	//Archivo de matriz A
	ifstream matrixBFile(MATRIX_B_FILE_NAME);	//Archivo de matriz B

	ofstream matrixCFile(MATRIX_C_FILE_NAME);	//Archivo de matriz resultante C

	//Se verifica que los archivos se puedan abrir
	if (!matrixAFile.is_open() || !matrixBFile.is_open() || !matrixCFile.is_open()) {
		cout << "ERROR: Could not open file" << endl;
		return 0;
	}

	cout << "Cantidad de filas de Matriz A: ";
	cin >> rowsA;

	cout << "Cantidad de columnas de Matriz A: ";
	cin >> columnsA;

	cout << "Cantidad de filas de Matriz B: ";
	cin >> rowsB;

	cout << "Cantidad de columnas de Matriz B: ";
	cin >> columnsB;

	//Se valida que la multiplicaci�n se pueda hacer
	if (rowsA != columnsB) {
		cout << "ERROR: can't do the matrix multiplications (rows_matrix_A != columns_matrix_B)" << endl;
		return 0;
	}




	
	return 0;
}
