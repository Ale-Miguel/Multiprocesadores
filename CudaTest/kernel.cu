#include <stdio.h>
#include <math.h>

// CUDA kernel to add elements of two arrays
__global__ void add(float* x, float* y) {
	int i = threadIdx.x;
	int blIdx = blockIdx.x;
	y[i + ( 1024 * blIdx) ] = x[i + (1024 * blIdx)] + y[i + (1024 * blIdx)];
	 //y[i ] = x[i ] + y[i];

	
		//printf("Block %d \n", blIdx);
	
}

int main(void) {
	//Variables
	int N = 2048;
	float* h_x, * h_y;
	int size = sizeof(float) * N;

	//Allocate Host Memory
	h_x = (float*)malloc(size);
	h_y = (float*)malloc(size);

	//Create Device Pointers
	float* d_x;
	float* d_y;

	//Allocate Device Memory
	cudaMalloc((void**)&d_x, size);
	cudaMalloc((void**)&d_y, size);

	//Initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		h_x[i] = 1.0;
		h_y[i] = 2.0;
	}

	//Memory copy Host to Device
	cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

	//Launch kernel on N elements on the GPU
	float numBlock = N / 1024 + 0.4;
	add << <numBlock, 1024 >> > (d_x, d_y);

	//Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	//Memory copy Host to Device of the result
	cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		//Print the CUDA error message
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}

	//Print array
	//for (int i = 0; i < N; i++)
	//printf("%d: %f\n", 
	//printf("%d: %f\n", i, h_y[i]);

//Check for errors (all values should be 3.0)
	float maxError = 0.0;
	for (int i = 0; i < N; i++)
		maxError = (float)fmax(maxError, fabs(h_y[i] - 3.0));
	printf("Max error: %lf\n", maxError);

	//Free cuda memory
	cudaFree(d_x);
	cudaFree(d_y);

	//Free memory
	free(h_x);
	free(h_y);

	return 0;
}
