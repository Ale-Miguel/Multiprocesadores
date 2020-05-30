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
#include <intrin.h>
#include <iomanip>

using namespace std;

#define MATRIX_A_FILE_NAME "matrixA.txt"
#define MATRIX_B_FILE_NAME "matrixB.txt"
#define MATRIX_C_FILE_NAME_P1 "matrixCp1.txt"
#define MATRIX_C_FILE_NAME_P2 "matrixCp2.txt"
#define MATRIX_C_FILE_NAME_SEQ "matrixCSeq.txt"

#define NUMBERS_PER_REGISTER 8
#define NUMBER_OF_RUNS 5

double **matrixA;
double **matrixB;
double **matrixCseq;		//Apuntador que guarda la matriz resultante del código secuencial
double **matrixCparallel1;	//Apuntador que guarda la matriz resultante del código en paralelo
double** matrixCparallel2;	//Apuntador que guarda la matriz resultante del código en paralelo

double** matrixT;

int columnsA, rowsA;
int columnsB, rowsB;
int columnsC, rowsC;

long tSequential[NUMBER_OF_RUNS];
long tIntrinsics[NUMBER_OF_RUNS];
long tOmpIntrinsics[NUMBER_OF_RUNS];

time_t start, endT;

//Función para estandarizar el formato de mensaje de error
void sendErrorMessage(string message) {
	cout << "ERROR: " << message << endl;
	exit(0);
}
void createResultMatrix(double**& matrix) {
	matrix = (double**)malloc(rowsC * sizeof(double*));
	if (!matrix) {
		sendErrorMessage("No se pudo asignar memoria para matriz C ");
	}

	for (int i = 0; i < rowsC; i++) {
		matrix[i] = (double*)malloc(columnsC * sizeof(double));

		if (!matrix[i]) {
			sendErrorMessage("No se pudo asignar memoria para matriz C");
		}

		for (int j = 0; j < columnsC; j++) {
			matrix[i][j] = 0;
		}
	}
}
void sequentialCode(double**& matrix) {

	createResultMatrix(matrix);

	for (int run = 0; run < NUMBER_OF_RUNS; run++) {

		start = clock();

		for (int i = 0; i < rowsC; i++) {
			for (int j = 0; j < columnsC; j++) {
				for (int k = 0; k < rowsB; k++) {
					matrix[i][j] += matrixA[i][k] * matrixB[k][j];
				}
			}
		}

		endT = clock();

		tSequential[run] = endT - start;

		printf("Seq time %d = %d\n", run, tSequential[run]);
	}
	
}

void transpose(double**& matrixOrigin, double**& matrixTarget , int rows, int columns) {
	matrixTarget = (double**)malloc((columns + columns % NUMBERS_PER_REGISTER) * sizeof(double*));
	if (!matrixTarget) {
		sendErrorMessage("No se pudo asignar memoria para matriz Aux en código secuencial");
	}

	for (int i = 0; i < columns; i++) {
		matrixTarget[i] = (double*)malloc((rows + rows % NUMBERS_PER_REGISTER) * sizeof(double));

		if (!matrixTarget[i]) {
			sendErrorMessage("No se pudo asignar memoria para matriz Aux en código secuencial");
		}

		for (int j = 0; j < rows; j++) {
			matrixTarget[i][j] = matrixOrigin[j][i];
		}

		for (int j = 0; j < rows % NUMBERS_PER_REGISTER; j++) {
			matrixTarget[i][j + rows] = 0;
		}
	}

	for (int i = 0; i < columns % NUMBERS_PER_REGISTER; i++) {

		matrixTarget[i + columns] = (double*)malloc((columns + columns % NUMBERS_PER_REGISTER) * sizeof(double));

		if (!matrixTarget[i]) {
			sendErrorMessage("No se pudo asignar la memoria suficiente para la matriz");
		}

		for (int j = 0; j < rows + rows % NUMBERS_PER_REGISTER; j++) {
			matrixTarget[i + columns][j] = 0;
		}
	}
}

void intrinsicsCode(double**& matrix) {
	
	__m256d a, b, r, r2;

	createResultMatrix(matrix);

	double* resultado = (double*)malloc(sizeof(double) * 4);

	for (int run = 0; run < NUMBER_OF_RUNS; run++) {

		start = clock();
#pragma omp parallel for 
		for (int i = 0; i < rowsC; i++) {
			for (int j = 0; j < columnsC; j++) {
				//for (int k = 0; k < rowsB / 4 + 1; k++) {
				for (int k = 0; k < rowsB; k++) {
					/*a = _mm256_loadu_pd(matrixA[i] + k * 4);
					b = _mm256_loadu_pd(matrixT[j] + k * 4);

					r = _mm256_mul_pd(a, b);
					r2 = _mm256_hadd_pd(r, r);

					_mm256_storeu_pd(resultado, r2);

					matrix[i][j] += resultado[0] + resultado[2];*/

					matrix[i][j] += matrixA[i][k] * matrixT[j][k];
				}
			}
		}

		endT = clock();

		tIntrinsics[run] = endT - start;

		printf("Intrin time %d = %d\n", run, tIntrinsics[run]);
	}
}

void openMPCode(double**& matrix) {

	__m256d a, b, r, r2;

	createResultMatrix(matrix);

	double* resultado = (double*)malloc(sizeof(double) * 4);

	for (int run = 0; run < NUMBER_OF_RUNS; run++) {

		start = clock();
#pragma omp parallel for 
		for (int i = 0; i < rowsC; i++) {
			for (int j = 0; j < columnsC; j++) {
				for (int k = 0; k < rowsB / 4 + 1; k++) {
					a = _mm256_loadu_pd(matrixA[i] + k * 4);
					b = _mm256_loadu_pd(matrixT[j] + k * 4);

					r = _mm256_mul_pd(a, b);
					r2 = _mm256_hadd_pd(r, r);

					_mm256_storeu_pd(resultado, r2);

					matrix[i][j] += resultado[0] + resultado[2];

					//matrix[i][j] += matrixA[i][k] * matrixT[j][k];
				}
			}
		}

		endT = clock();

		tOmpIntrinsics[run] = endT - start;

		printf("OMP Intrin time %d = %d\n", run, tOmpIntrinsics[run]);
	}
}


//Función para llenar las matrices cuya información se obtiene de un archivo
void fillMatrix(double** &matrix, ifstream &matrixFile, int rows, int columns) {
	matrix = (double**)malloc((rows + rows % NUMBERS_PER_REGISTER) * sizeof(double*));

	if (!matrix) {
		sendErrorMessage("No se pudo asignar la memoria suficiente para la matriz");
	}

	double line;	//Variable que va guardando el valor de cada línea del archivo

	for (int i = 0; i < rows; i++) {

		matrix[i] = (double*)malloc((columns + columns % NUMBERS_PER_REGISTER) * sizeof(double));

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

		for (int j = 0; j < columns % NUMBERS_PER_REGISTER; j++) {
			matrix[i][j + columns] = 0;
		}
	}

	for (int i = 0; i < rows % NUMBERS_PER_REGISTER; i++) {
		matrix[i + rows] = (double*)malloc((columns + columns % NUMBERS_PER_REGISTER) * sizeof(double));

		if (!matrix[i]) {
			sendErrorMessage("No se pudo asignar la memoria suficiente para la matriz");
		}

		for (int j = 0; j < columns + columns % NUMBERS_PER_REGISTER; j++) {
			matrix[i + rows][j] = 0;
		}
	}
	
}

void saveMatrix(double**& matrix, ofstream& matrixFile, int rows, int columns) {
	double line;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			matrixFile << fixed << setprecision(10) << (matrix[i][j]) << endl;
		}

	}
}

void validateMatrix(string matrixFileSeq, string matrixFileP) {
	ifstream seqFile(matrixFileSeq);
	ifstream parFile(matrixFileP);

	double seq, par;

	for (int i = 0; i < rowsC * columnsC; i++) {
		seqFile >> seq;
		parFile >> par;

		//cout << seq << "\t" << par << endl;
		if (seq != par) {
			sendErrorMessage("Resultados distintos de secuencial con paralelo");
		}
	}

	seqFile.close();
	parFile.close();
}

int main() {

	ifstream matrixAFile(MATRIX_A_FILE_NAME);	//Archivo de matriz A
	ifstream matrixBFile(MATRIX_B_FILE_NAME);	//Archivo de matriz B

	ofstream matrixCFilep1(MATRIX_C_FILE_NAME_P1);	//Archivo de matriz resultante C
	ofstream matrixCFilep2(MATRIX_C_FILE_NAME_P2);	//Archivo de matriz resultante C
	ofstream matrixCFileSeq(MATRIX_C_FILE_NAME_SEQ);	//Archivo de matriz resultante C

	/*matrixCFilep1.open(MATRIX_C_FILE_NAME_P1, fstream::in | fstream:: out);
	matrixCFilep2.open(MATRIX_C_FILE_NAME_P2, fstream::in | fstream::out);
	matrixCFileSeq.open(MATRIX_C_FILE_NAME_SEQ, fstream::in | fstream::out);*/
	
	//Se verifica que los archivos se puedan abrir
	/*if (!matrixAFile.is_open() || !matrixBFile.is_open() || !matrixCFilep1.is_open() || !matrixCFilep2.is_open() || !matrixCFileSeq.is_open()) {
		sendErrorMessage("No se pudo abrir el archivo");
	}*/

	/*if ( !matrixCFileSeq.is_open()) {
		sendErrorMessage("No se pudo abrir el archivo");
	}*/

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

	cout << "Matrix A" << endl;
	/*for (int i = 0; i < rowsA; i++) {
		for (int j = 0; j < columnsA; j++) {
			cout << matrixA[i][j] << "\t";
		}

		cout << endl;
	}*/

	cout << "Matrix B" << endl;
	/*for (int i = 0; i < rowsB; i++) {
		for (int j = 0; j < columnsB; j++) {
			cout << matrixB[i][j] << "\t";
		}

		cout << endl;
	}*/


	transpose(matrixB, matrixT, rowsB, columnsB);

	cout << "Matrix B transpose sequential" << endl;

	if (!matrixT) {
		sendErrorMessage("NOPE");
	}

	/*for (int i = 0; i < columnsB; i++) {
		for (int j = 0; j < rowsB; j++) {
			cout << matrixT[i][j] << "\t";
		}

		cout << endl;
	}*/

	sequentialCode(matrixCseq);

	saveMatrix(matrixCseq, matrixCFileSeq, rowsC, columnsC);


	cout << "Matrix C sequential" << endl;
	/*for (int i = 0; i < rowsC; i++) {
		for (int j = 0; j < columnsC; j++) {
			cout << matrixCseq[i][j] << "\t";
		}

		cout << endl;
	}*/

	intrinsicsCode(matrixCparallel1);
	saveMatrix(matrixCparallel1, matrixCFilep1, rowsC, columnsC);

	validateMatrix(MATRIX_C_FILE_NAME_SEQ, MATRIX_C_FILE_NAME_P1);

	openMPCode(matrixCparallel2);
	saveMatrix(matrixCseq, matrixCFilep2, rowsC, columnsC);

	validateMatrix(MATRIX_C_FILE_NAME_SEQ, MATRIX_C_FILE_NAME_P2);
	
	/*for (int i = 0; i < rowsC; i++) {
		for (int j = 0; j < columnsC; j++) {
			//cout << matrixCparallel2[i][j] << "\t";

			if (matrixCparallel2[i][j] != matrixCseq[i][j]) {
				sendErrorMessage("Resultados de OMPintrisnicas y secuencial no son iguales");
			}
		}

		//cout << endl;
	}*/

	/*for (int i = 0; i < rowsC * columnsC; i++) {
		matrixCFilep2 >> lineParallel;
		matrixCFileSeq >> lineSeq;

		if (lineParallel != lineSeq) {
			sendErrorMessage("Resultados de OMPintrisnicas y secuencial no son iguales");
		}
	}*/

	matrixAFile.close();
	matrixBFile.close();
	matrixCFileSeq.close();
	matrixCFilep1.close();
	matrixCFilep2.close();

	free(matrixA);
	free(matrixB);
	free(matrixCseq);
	free(matrixCparallel1);
	free(matrixT);

	return 0;
}