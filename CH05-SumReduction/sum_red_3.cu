#include<iostream>
#include<stdio.h>

#include<stdlib.h>
#include<math.h>

#include<cuda.h>

#include "../helper_functions/data_print.h"
#include "../helper_functions/profiling_timer.h"
#include "../helper_functions/query_device_property.h"

using namespace std;

__device__ void warp_reduce(volatile float* input_shared, int tx)
{
    input_shared[tx] += input_shared[tx + 32];
    input_shared[tx] += input_shared[tx + 16];
    input_shared[tx] += input_shared[tx + 8];
    input_shared[tx] += input_shared[tx + 4];
    input_shared[tx] += input_shared[tx + 2];
    input_shared[tx] += input_shared[tx + 1];
}
__global__ void kernel_sum_reduction(float* d_input, float* d_output, int length, int* Dcounter)
{

    __shared__ int sbx;
    if(threadIdx.x == 0)
    {
        sbx = atomicAdd(&Dcounter[0],1);
    }    
    __syncthreads();
    extern __shared__ float sharedInp[];
    int tx = threadIdx.x;
    int bx = sbx;
    int row = bx*(2*blockDim.x) + tx;
    //Copy input to shared Mem
    if(row<length)
        sharedInp[tx] = d_input[row];
    else
        sharedInp[tx] = 0;

    if((row+blockDim.x) < length)
        sharedInp[tx + blockDim.x] = d_input[row+blockDim.x];
    else
        sharedInp[tx + blockDim.x] = 0;

    __syncthreads();
    //Perfrom reduction
    for(int i = blockDim.x; i>1; i = ((i-1)/2 + 1))
    {
        if(tx<i)
            sharedInp[tx] += sharedInp[tx+i];
        __syncthreads();
        if(tx>=i)
        {
            sharedInp[tx] = 0;
        }
        sharedInp[tx + blockDim.x] = 0;
        __syncthreads();
    }
    if(tx == 0)
    {
        sharedInp[0] += sharedInp[1];
    }
    __syncthreads();
    //Copy sharedMem to output
    if(tx == 0)
    {
        d_output[0] = sharedInp[0];
        d_input[bx] = sharedInp[0];
    }
}

void h_sum_reduction(float* h_input, float* h_output, int length)
{
    //Allocate device
    float* d_input;
    int input_size = length*sizeof(float);

    cudaError_t err = cudaMalloc((void**)&d_input, input_size);
    if(err != cudaSuccess)
    {
        printf("%s at %d", cudaGetErrorString(err),__LINE__);
    }

    float* d_output;
    int output_size = sizeof(float);

    cudaMalloc((void**)&d_output, output_size);

    int* Dcounter;
    int dcounter_size = sizeof(int);

    cudaMalloc((void**)&Dcounter, dcounter_size);
    cudaMemset((void*)&Dcounter, 0, dcounter_size);

    int* Dcounter2;

    cudaMalloc((void**)&Dcounter2, dcounter_size);
    cudaMemset((void*)&Dcounter2, 0, dcounter_size);

    //Copy value to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    //Call kernel
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop,0);

    int max_threads_per_block = dev_prop.maxThreadsPerBlock;
    if(length<(2*max_threads_per_block))
        max_threads_per_block = ceil(length/2.0);

    dim3 threadsPerBlock(max_threads_per_block, 1, 1);
    dim3 blocksPerKernel(ceil(float(length)/(2.0*max_threads_per_block)),1,1);

    int sharedMemSize = 2*max_threads_per_block*sizeof(float);

    kernel_sum_reduction<<<blocksPerKernel, threadsPerBlock, sharedMemSize>>>(d_input, d_output, length, Dcounter);

    //level 2
    int length2 = ceil(float(length)/(2.0*max_threads_per_block));
    int max_threads_per_block2 = dev_prop.maxThreadsPerBlock;
    if(length2<(2*max_threads_per_block2))
        max_threads_per_block2 = ceil(length2/2.0);
    
    dim3 threadsPerBlock2(max_threads_per_block2, 1, 1);
    dim3 blocksPerKernel2(ceil(float(length2)/(2.0*max_threads_per_block2)),1,1);

    int sharedMemSize2 = 2*max_threads_per_block2*sizeof(float);

    kernel_sum_reduction<<<blocksPerKernel2, threadsPerBlock2, sharedMemSize2>>>(d_input, d_output, length2, Dcounter2);

    //Copy results back
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
    //Deallocate space
    cudaFree(d_input);
    cudaFree(d_output);
}
 
void h_sum_reduction_cpu(float* h_input, float* h_output, float length)
{
    auto start = get_current_time();
    float Psum = 0;
    for(int i = 0; i< length; i+=1)
    {
        Psum += h_input[i];
    }
    h_output[0] = Psum;
    auto end = get_current_time();
    auto diff = end - start;
    auto ms = get_time_taken_ms(diff);
    cout << "milli seconds since start (cpu): " << ms <<'\n';
}


int main()
{
    const int length = pow(2,17)-5;
    float* h_input;
    h_input = new float[length];

    for(int i = 0; i< length; i+=1)
    {
        h_input[i] = (i+1)%16;
    }

    float* h_output;
    h_output = new float[1];

    h_sum_reduction(h_input, h_output, length);

    cout<<"Final sum is (GPU): "<<h_output[0]<<'\n';

    h_sum_reduction_cpu(h_input, h_output, length);

    cout<<"Final sum is (CPU): "<<h_output[0]<<'\n';
}