/*
Programa Final - Multiplicaci�n de Matrices

Por:
Alejandro Miguel S�nchez Mora - A01272385
Jessica Tovar Saucedo - A00818101

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
#include <intrin.h>
#include <iomanip>

using namespace std;

#define MATRIX_A_FILE_NAME "matrizA.txt"
#define MATRIX_B_FILE_NAME "matrizB.txt"
#define MATRIX_C_FILE_NAME_SEQ "matrizC.txt"

#define NUMBERS_PER_REGISTER 2
#define NUMBER_OF_RUNS 5
#define ITEMS_PER_REGISTER 2

double **matrixA;
double **matrixB;
double **matrixCseq;		//Apuntador que guarda la matriz resultante del c�digo secuencial
double **matrixCparallel1;	//Apuntador que guarda la matriz resultante del c�digo en paralelo
double** matrixCparallel2;	//Apuntador que guarda la matriz resultante del c�digo en paralelo

double** matrixT;

int columnsA, rowsA;
int columnsB, rowsB;
int columnsC, rowsC;

long tSequential[NUMBER_OF_RUNS];
long tIntrinsics[NUMBER_OF_RUNS];
long tOmp[NUMBER_OF_RUNS];

time_t start, endT;

//Funci�n para estandarizar el formato de mensaje de error
void sendErrorMessage(string message) {
	cout << "ERROR: " << message << endl;
	exit(0);
}

//Funci�n que crea la matriz resultante de la multiplicaci�n de matrices 
void createResultMatrix(double**& matrix) {

	if (matrix) {
		return;
	}

	matrix = (double**)_aligned_malloc(rowsC * sizeof(double*), 16);


	if (!matrix) {
		sendErrorMessage("No se pudo asignar memoria para matriz C ");
	}

	for (int i = 0; i < rowsC; i++) {


		matrix[i] = (double*)_aligned_malloc(columnsC * sizeof(double), 16);


		if (!matrix[i]) {
			sendErrorMessage("No se pudo asignar memoria para matriz C");
		}

		for (int j = 0; j < columnsC; j++) {
			matrix[i][j] = 0;
		}
	}

	
}
void sequentialCode(double**& matrix, int run) {

	createResultMatrix(matrix);

	


		start = clock();

		for (int i = 0; i < rowsC; i++) {
			for (int j = 0; j < columnsC; j++) {

				matrix[i][j] = 0;

				for (int k = 0; k < rowsB; k++) {
					matrix[i][j] += matrixA[i][k] * matrixB[k][j];
				}
			}
		}

		endT = clock();

		tSequential[run] = endT - start;

		//printf("Seq time %d = %d\n", run, tSequential[run]);
	
	
}

//Funci�n que genera una matriz transpuesta
void transpose(double**& matrixOrigin, double**& matrixTarget , int rows, int columns) {

	matrixTarget = (double**)_aligned_malloc((columns + columns % NUMBERS_PER_REGISTER) * sizeof(double*), 16);

	if (!matrixTarget) {
		sendErrorMessage("No se pudo asignar memoria para matriz transpuesta");
	}

	for (int i = 0; i < columns; i++) {
		matrixTarget[i] = (double*)_aligned_malloc((rows + rows % NUMBERS_PER_REGISTER) * sizeof(double), 16);

		if (!matrixTarget[i]) {
			sendErrorMessage("No se pudo asignar memoria para matriz Aux en c�digo secuencial");
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

void intrinsicsCode(double**& matrix, int run) {
	
	__m128d a, b, r;

	createResultMatrix(matrix);

	double* resultado = (double*)malloc(sizeof(double) * ITEMS_PER_REGISTER);
		
	start = clock();

	for (int i = 0; i < rowsC; i++) {
		for (int j = 0; j < columnsC; j++) {
			matrix[i][j] = 0;
			for (int k = 0; k < rowsB / ITEMS_PER_REGISTER + rowsB % ITEMS_PER_REGISTER; k++) {
				a = _mm_load_pd(matrixA[i] + k * ITEMS_PER_REGISTER);
				b = _mm_load_pd(matrixT[j] + k * ITEMS_PER_REGISTER);

				r = _mm_dp_pd(a, b, 255);

				_mm_store_pd(resultado, r);

				matrix[i][j] += resultado[0];

				//matrix[i][j] += matrixA[i[][k] * matrixB[k][j];
			}
		}
	}

	endT = clock();

	tIntrinsics[run] = endT - start;

	//printf("Intrin time %d = %d\n", run, tIntrinsics[run]);
	
}

void openMPCode(double**& matrix, int run) {

	createResultMatrix(matrix);

	start = clock();

	#pragma omp parallel for
	for (int i = 0; i < rowsC; i++) {
		for (int j = 0; j < columnsC; j++) {

			matrix[i][j] = 0;

			for (int k = 0; k < rowsB; k++) {
				matrix[i][j] += matrixA[i][k] * matrixB[k][j];
			}
		}
	}

	endT = clock();

	tOmp[run] = endT - start;

	//printf("OMP time %d = %d\n", run, tOmp[run]);
	
}


//Funci�n para llenar las matrices cuya informaci�n se obtiene de un archivo
void fillMatrix(double** &matrix, ifstream &matrixFile, int rows, int columns) {

	matrix = (double**)_aligned_malloc((rows + rows % NUMBERS_PER_REGISTER) * sizeof(double*), 16);

	if (!matrix) {
		sendErrorMessage("No se pudo asignar la memoria suficiente para la matriz");
	}

	double line;	//Variable que va guardando el valor de cada l�nea del archivo

	for (int i = 0; i < rows; i++) {

		matrix[i] = (double*)_aligned_malloc((columns + columns % NUMBERS_PER_REGISTER) * sizeof(double), 16);

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

//Funci�n que guarda la matriz resultante en un archivo
void saveMatrix(double**& matrix, ofstream& matrixFile, int rows, int columns) {
	double line;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			matrixFile << fixed << setprecision(10) << (matrix[i][j]) << endl;
		}

	}
}

//Funci�n que checa si los valores de dos matrices son iguales con 10 decimales de precisi�n
void validateMatrix(double**& matrixSeq, double**& matrixPar, string codeName) {

	cout << "Validando resultados de codigo secuencial contra " << codeName << endl;

	for (int i = 0; i < rowsC; i++) {
		for (int j = 0; j < columnsC; j++) {
			
			//Para saber si los n�meros son iguales en sus primeros 10 decimales, 
			//Se hace una resta 
			double difference = matrixSeq[i][j] - matrixPar[i][j];

			//Se saca el valor absoluto
			if (difference < 0)
				difference *= -1;

			//Si la diferencia es mayor a los 10 decimales, entonces el resultado es incorrecto
			if (difference > 0.0000000001) {
				printf("Seq: %.10f Par: %.10f at [%d][%d] Difference = %0.20f\n", matrixSeq[i][j], matrixPar[i][j], i, j, difference);
				sendErrorMessage("Resultados distintos de secuencial con " + codeName);
			}
		}
	}

	cout << "Los resultados coinciden" << endl;
}

int main() {

	ifstream matrixAFile(MATRIX_A_FILE_NAME);	//Archivo de matriz A
	ifstream matrixBFile(MATRIX_B_FILE_NAME);	//Archivo de matriz B

	ofstream matrixCFileSeq(MATRIX_C_FILE_NAME_SEQ);	//Archivo de matriz resultante C

	double seqAvg = 0;
	double intrinAvg = 0;
	double OMPAvg = 0;

	//Se verifica que los archivos se puedan abrir
	if (!matrixAFile.is_open() || !matrixBFile.is_open() || !matrixCFileSeq.is_open()) {
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

	//Se valida que la multiplicaci�n se pueda hacer
	if (columnsA != rowsB) {
		sendErrorMessage("No se puede hacer la multiplicacion de matrices");
	}

	rowsC = rowsA;
	columnsC = columnsB;

	fillMatrix(matrixA, matrixAFile, rowsA, columnsA);
	fillMatrix(matrixB, matrixBFile, rowsB, columnsB);

	transpose(matrixB, matrixT, rowsB, columnsB);


	if (!matrixT) {
		sendErrorMessage("No se pudo crear la matriz transpuesta");
	}

	printf("Corrida\t\tSerial\t\tIntrinsecas\tOpenMP\n");

	for (int run = 0; run < NUMBER_OF_RUNS; run++) {

		printf("%d", run + 1);

		sequentialCode(matrixCseq, run);
		printf("\t\t%ld", tSequential[run]);

		intrinsicsCode(matrixCparallel1, run);
		printf("\t\t%ld", tIntrinsics[run]);

		openMPCode(matrixCparallel2, run);
		printf("\t\t%ld\n", tOmp[run]);

		seqAvg += tSequential[run];
		intrinAvg += tIntrinsics[run];
		OMPAvg += tOmp[run];
	}
	
	seqAvg /= NUMBER_OF_RUNS;
	intrinAvg /= NUMBER_OF_RUNS;
	OMPAvg /= NUMBER_OF_RUNS;

	printf("Promedio\t%lf\t%lf\t%lf\n", seqAvg, intrinAvg, OMPAvg);

	double intrinImpv = intrinAvg / seqAvg;
	double ompImp = OMPAvg / seqAvg;

	printf("%% vs Serial\t-\t\t%lf\t%lf\n", intrinImpv, ompImp);

	cout << "El codigo ";

	if (OMPAvg <= intrinAvg && OMPAvg <= seqAvg) {
		cout << "de OMP";
	}
	else if (intrinAvg <= OMPAvg && intrinAvg <= seqAvg) {
		cout << "de INTRINSECAS";
	}
	else {
		cout << "SECUENCIAL";
	}

	cout << " fue el mas rapido en hacer la multiplicacion de matrices" << endl;

	saveMatrix(matrixCseq, matrixCFileSeq, rowsC, columnsC);

	//Se valida que el resultado del c�digo de intr�nsecas sea igual que el resultado del c�digo secuencial
	validateMatrix(matrixCseq, matrixCparallel1, "intrinsecas");

	//Se valida que el resultado del c�digo de Open MP sea igual que el resultado del c�digo secuencial
	validateMatrix(matrixCseq, matrixCparallel2, "OMP");
	
	matrixAFile.close();
	matrixBFile.close();
	matrixCFileSeq.close();

	_aligned_free(matrixA);
	_aligned_free(matrixB);
	_aligned_free(matrixCseq);
	_aligned_free(matrixCparallel1);
	_aligned_free(matrixT);
	
	return 0;
}