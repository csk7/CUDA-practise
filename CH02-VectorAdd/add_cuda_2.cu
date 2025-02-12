#include<cuda.h>
#include<math.h>
#include <stdio.h>

__global__ void add_cuda(float* A, float* B, float* C, int N)
{
	int pos = blockDim.x*blockIdx.x + threadIdx.x;
	if(pos<N)
	{
		C[pos] = A[pos] + B[pos]; 
	}
}


void vecAdd(float* h_A, float* h_B, float* h_C, int N)
{
	float* d_A;
	float* d_B;
	float* d_C;
	int size = N*sizeof(float);
	cudaMalloc((void**) &d_A,size);
	cudaMemcpy(d_A,h_A,size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_B,size);
	cudaMemcpy(d_B,h_B,size, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_C,size);
	add_cuda<<<ceil(256.0/N),256>>> (d_A,d_B,d_C,N);

	
	cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

}


int main()
{
	int n = 5;
	float* h_A = new float[n];
	float* h_B = new float[n];
	float* h_C = new float[n];

	for(int i=0; i<n; i++)
	{
		h_A[i] = i;
		h_B[i] = i;
	}

	vecAdd(h_A,h_B,h_C,n);
	for(int i=0; i<n; i++)
	{
		printf("%f\t",h_C[i]);
	}

	return 0;

}