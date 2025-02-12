#include<cuda.h>
#include<iostream>

using namespace std;

#include"../helper_functions/data_print.h"

#include<math.h>

void check_mem_allocation(cudaError_t err)
{
    if(err != cudaSuccess)
    {
        printf("%s in %s @ line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);
    }
}

__global__ void hist_kernel(int* d_input_arr, float* d_bins, int length, int total_bin)
{
    extern __shared__ float sharedPartialSum[];

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    int row = blockDim.x * bx + tx;

    //Set shared partial sum to 0

    for(int i = tx ;i<total_bin; i+=blockDim.x)
    {
        sharedPartialSum[i] = 0;
    }
    __syncthreads();

    //Set counter, prev bin, cur bin

    int counter = 1;
    int cur_bin = -1;
    int prev_bin = -1;

    for(int i = row; i<length; i+= (gridDim.x*blockDim.x))
    {
        cur_bin = d_input_arr[i];
        
        if(cur_bin!=prev_bin)
        {
            //Update
            if(prev_bin == -1)
            {
                atomicAdd(&sharedPartialSum[cur_bin], counter);  
            }
            else
            {
                atomicAdd(&sharedPartialSum[prev_bin], counter);
            }
            prev_bin = cur_bin;
            counter = 1;
            
        }
        else
        {
            counter += 1;
            
        }
    }

    __syncthreads();

    for(int i = tx ;i<total_bin; i+=blockDim.x)
    {
        

        atomicAdd(&d_bins[i],sharedPartialSum[i]);
    }

    __syncthreads();

}

void compute_histogram_gpu(int* input_arr, float* gpu_bins, int length, int total_bin)
{
    //Declare and allocate mem for inp/out
    int* d_input_arr;
    int size_input = length*sizeof(int);
    cudaError_t err = cudaMalloc((void**)&d_input_arr, size_input);
    check_mem_allocation(err);

    float* d_bins;
    int size_bins = total_bin*sizeof(float);
    cudaMalloc((void**)&d_bins, size_bins);
    cudaMemset((void*)&d_bins, 0, size_bins);
    
    //Transfer data
    cudaMemcpy(d_input_arr, input_arr, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins, gpu_bins, size_bins, cudaMemcpyHostToDevice);

    //invoke kernel
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    int maxGridSize = dev_prop.maxGridSize[0];
    int maxBlockSize = dev_prop.maxThreadsPerBlock;
    
    int threadsPerBlock_x;

    if(length < maxBlockSize)
    {
        threadsPerBlock_x = length;
    }
    else
    {
        threadsPerBlock_x = maxBlockSize;
    }

    int blocksPerGrid_x;
    blocksPerGrid_x = ceil(length/threadsPerBlock_x);

    if(blocksPerGrid_x > maxGridSize)
    {
        blocksPerGrid_x = maxBlockSize;
    }
    
    dim3 blocksPerGrid(blocksPerGrid_x, 1, 1);
    dim3 threadsPerBlock(threadsPerBlock_x, 1, 1);

    int blockSharedMemSize = size_bins;

    cout<<"blocksPerGrid : "<<blocksPerGrid_x<<'\n';
    cout<<"threadsPerBlock : "<<threadsPerBlock_x<<'\n';

    hist_kernel<<<blocksPerGrid, threadsPerBlock, blockSharedMemSize>>> (d_input_arr, d_bins, length, total_bin);

    //Transfer data
    cudaMemcpy(gpu_bins, d_bins, size_bins, cudaMemcpyDeviceToHost);
    //Remove allocated space
    cudaFree(d_input_arr);
    cudaFree(d_bins);
}



int main()
{
    int length = 100;
    int* input_arr = new int[length];
    for(int i=0; i<length; i++)
    {
        input_arr[i] = i%10;
    }
    //Cpu compute
    int total_bin = 10;
    //Cpu compute
    float* bins = new float[total_bin];
    for(int i = 0; i<total_bin; i++)
        bins[i] = 0;
    for(int i=0; i<length; i++)
    {
        bins[input_arr[i]]++;
    }
    print_1d_array(bins, total_bin, "CPU bins : ");
    //gpu compute

    float* gpu_bins = new float[total_bin];
    for(int i = 0; i<total_bin; i++)
        gpu_bins[i] = 0;
    compute_histogram_gpu(input_arr, gpu_bins, length, total_bin);
    print_1d_array(gpu_bins, total_bin, "GPU bins : ");

    bool successFlag = true;
    for(int i=0; i<total_bin; i++)
    {
        if(gpu_bins[i] != bins[i])
        {
            cout<<"Mismatch at "<<i<<" Val : "<<bins[i]<<" "<<gpu_bins[i]<<"\n";
            successFlag = false;
            break;
        }
    }
    if(successFlag == true)
    {
        cout<<"Success \n";
    }

}

