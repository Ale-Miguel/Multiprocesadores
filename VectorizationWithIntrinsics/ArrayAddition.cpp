#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <intrin.h>
// Single Threaded Array Addition FLOATs

int main() {
	int i;
	float* A = NULL;
	float* B = NULL;
	float* C = NULL;
	int result = 1;
	const int elements = 52428800;
	time_t start, end;

	//Array creation
	size_t datasize = sizeof(int) * elements;
	A = (float*)malloc(datasize);
	B = (float*)malloc(datasize);
	C = (float*)malloc(datasize);

	//Array initialization (Normally you would get this from a file)
	for (i = 0; i < elements; i++) {
		A[i] = (float)i;
		B[i] = (float)i;
	}

	start = clock();
	/*
	//This loop can be optimized using Intrinsics
	for (i = 0; i < elements; i++) {
		C[i] = A[i] + B[i];
	}
	*/

	/*
	__m128 vA, vB, res;
	//Loop with SSE Intrinsics
	for (int i = 0; i < elements / 4; i++) {
		vA = _mm_load_ps(A + i * 4);
		vB = _mm_load_ps(B + i * 4);
		res = _mm_add_ps(vA, vB);

		_mm_store_ps(C + i * 4, res);
	}
	*/

	__m256 vA, vB, res;
	//Loop with AVX Intrinsics
	for (int i = 0; i < elements / 8; i++) {
		 vA = _mm256_load_ps(A + i * 8);
		 vB = _mm256_load_ps(B + i * 8);
		 res = _mm256_add_ps(vA, vB);

		_mm256_store_ps(C + i * 8, res);
	}
	end = clock();

	//Validation
	for (i = 0; i < elements; i++) {
		if (C[i] != i + i) {
			result = 0;
			break;
		}
	}

	//Print first 10 results
	for (i = 0; i < 10; i++) {
		printf("C[%d]=%10.2lf\n", i, C[i]);
	}

	if (result) {
		printf("Results verified!!! (%ld)\n", (long)(end - start));
	}
	else {
		printf("Wrong results!!!\n");
	}

	//Memory deallocation
	free(A);
	free(B);
	free(C);

	return 0;
}