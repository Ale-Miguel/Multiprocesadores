/*
Alejandro Miguel Sánchez Mora
A01272385

Jessica Tovar Saucedo 
A00818101
*/
#include <stdio.h>

#define STEPS 100000000
#define BLOCKS 100
#define THREADS 100

int threadidx;
float pi = 0;

// Kernel
__global__ void pi_calculation(float* sum, int nsteps, double base, int nthreads, int nblocks) {

	int i;
	double x;
	float acum = 0;
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate index for each thread
	for (i = idx; i < nsteps; i += nthreads * nblocks) {
		x = (i + 0.5) * base;
		acum += 4.0 / (1.0 + x * x); //Save result to device memory
	}
	
	//Este atomicAdd usa floats en vez de doubles porque la computadora donde se hizo la tarea
	//tiene CUDA Compute Cabability 5.0 (GTX 850M) y la suma con doubles solo es soportado a partir
	//del Compute Capability 6.X y posterior.
	
	atomicAdd(&sum[0], acum);
}

int main(void) {
	time_t start, endT;
	dim3 dimGrid(BLOCKS, 1, 1); // Grid dimensions
	dim3 dimBlock(THREADS, 1, 1); // Block dimensions
	float* h_sum, * d_sum; // Pointer to host & device arrays
	double base = 1.0 / STEPS; // base size
	size_t size =  sizeof(double); //Array memory size

	//Memory allocation
	h_sum = (float*)malloc(size); // Allocate array on host
	cudaMalloc((void**)&d_sum, size); // Allocate array on device
	// Initialize array in device to 0
	cudaMemset(d_sum, 0, size);

	start = clock();
	// Launch Kernel
	pi_calculation << <dimGrid, dimBlock >> > (d_sum, STEPS, base, THREADS, BLOCKS);

	// Sync
	cudaDeviceSynchronize();

	// Copy results from device to host
	cudaMemcpy(h_sum, d_sum, size, cudaMemcpyDeviceToHost);

	
	pi = *h_sum * base;

	endT = clock();

	//Se pierde presición en la aproximación de pi porque se están utilizando floats en vez de doubles
	// Output Results
	printf("PI = %.10f (%d)\n", pi, endT - start);

	// Cleanup
	free(h_sum);
	cudaFree(d_sum);

	return 0;
}

//CÓDIGO ORIGINAL
/*#include <stdio.h>

#define STEPS 100000000
#define BLOCKS 100
#define THREADS 100

int threadidx;
double pi = 0;

// Kernel
__global__ void pi_calculation(double* sum, int nsteps, double base, int nthreads, int nblocks) {
	int i;
	double x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate index for each thread
	for (i = idx; i < nsteps; i += nthreads * nblocks) {
		x = (i + 0.5) * base;
		sum[idx] += 4.0 / (1.0 + x * x); //Save result to device memory
	}
}

int main(void) {
	time_t start, endT;
	dim3 dimGrid(BLOCKS, 1, 1); // Grid dimensions
	dim3 dimBlock(THREADS, 1, 1); // Block dimensions
	double* h_sum, * d_sum; // Pointer to host & device arrays
	double base = 1.0 / STEPS; // base size
	size_t size = BLOCKS * THREADS * sizeof(double); //Array memory size

	//Memory allocation
	h_sum = (double*)malloc(size); // Allocate array on host
	cudaMalloc((void**)&d_sum, size); // Allocate array on device
	// Initialize array in device to 0
	cudaMemset(d_sum, 0, size);
	
	start = clock();
	// Launch Kernel
	pi_calculation << <dimGrid, dimBlock >> > (d_sum, STEPS, base, THREADS, BLOCKS);

	// Sync
	cudaDeviceSynchronize();
	
	// Copy results from device to host
	cudaMemcpy(h_sum, d_sum, size, cudaMemcpyDeviceToHost);

	// Do the final reduction.
	for (threadidx = 0; threadidx < THREADS * BLOCKS; threadidx++)
		pi += h_sum[threadidx];

	// Multiply by base
	pi *= base;
	endT = clock();
	// Output Results
	printf("PI = %.10f (%d)\n", pi, endT - start);

	// Cleanup
	free(h_sum);
	cudaFree(d_sum);

	return 0;
}

*/