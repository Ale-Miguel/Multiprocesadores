#include <stdio.h>
#include <intrin.h>

int main() {

	int array0[8];
	/*__m128i a4 = _mm_set_epi32(9, -2, 0, 4);
	__m128i b4 = _mm_set_epi32(-7, 8, 3, 1);
	__m128i sum4 = _mm_add_epi32(a4, b4);
	_mm_storeu_si128((__m128i *)&array0, sum4);
	*/

	//Se declaran los dos registros de tamaño 8 (256 bits)
	__m256i a4 = _mm256_set_epi32(75, 65, 51, 86, 65, -212, 0, 44);
	__m256i b4 = _mm256_set_epi32(-85, 35, 4,  -86, -60, 220, 4, 31);

	//Se efectúa la suma de ambos vectores y el resultado se guarda en sum4
	__m256i sum4 = _mm256_add_epi32(a4, b4);

	//Se copia el resultado en el arreglo array0
	_mm256_storeu_si256((__m256i *)&array0, sum4);

	///Se imprime el resultado
	for (int i = 0; i <= 7; i++)
		printf("%d\n", array0[i]);

	return 0;
}