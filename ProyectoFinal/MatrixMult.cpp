/*
Programa Final - Multiplicación de Matrices

Por:
Alejandro Miguel Sánchez Mora - A01272385
Jessica Tovar

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

	//Se valida que la multiplicación se pueda hacer
	if (rowsA != columnsB) {
		cout << "ERROR: can't do the matrix multiplications (rows_matrix_A != columns_matrix_B)" << endl;
		return 0;
	}




	
	return 0;
}
