#include<iostream>
using namespace std;

#include<cuda.h>
#include<math.h>
#include<chrono>

#include"../helper_functions/data_print.h"
#include"../helper_functions/profiling_timer.h"


#define PRINT_FLAG false

__global__ void hierarchial_summation(float* d_input_array, float* d_output_array, int length)
{
    //
}


int round_length_to_upper_power_of_2(int length)
{
    int val = 1;
    while(length>val)
    {
        val = val*2;
    }
    return val;
}
__global__ void kernel_brent_k(float* d_input_array, float* d_output_array, float* scan_value, int*flags, int* Dcounter, int length)
{
    extern __shared__ float shared_xy[];
    __shared__ float prev_sum;
    int tx = threadIdx.x;
    //read Dcounter
    __shared__ int s_bx;
    if(tx == 0)
    {
        s_bx = atomicAdd(&Dcounter[0],1);
    }
    __syncthreads();
    const int bx = s_bx;
    

    int row = bx*(2*blockDim.x) + tx;
    //Load shared mem
    if(row<length)
    {
        shared_xy[tx] = d_input_array[row];
    }
    else
    {
        shared_xy[tx] = 0 ;
    }
    if(row+blockDim.x < length)
    {
        shared_xy[tx + blockDim.x] = d_input_array[row + blockDim.x];
    }
    else
    {
        shared_xy[tx + blockDim.x] = 0 ;
    }
    __syncthreads();
    //Stage 1
    int max_threads;
    int idx;
    for(int stride = 1; stride <= (blockDim.x); stride*=2)
    {
        max_threads = (blockDim.x)/stride;
        idx         = 2*stride*(tx+1) - 1;
        if(tx < max_threads)
        {
            shared_xy[idx] = shared_xy[idx] + shared_xy[idx - stride];
        }
        __syncthreads();
    }
    //stage 2
    for(int stride = (blockDim.x/2); stride >= 1; stride /=2)
    {
        idx         = 2*stride*(tx+1) - 1;
        max_threads = (blockDim.x)/stride - 1;
        if(tx < max_threads)
        {
            shared_xy[idx+stride] = shared_xy[idx+stride] + shared_xy[idx];
        } 
        __syncthreads();
    }
    if(row<length)
    {
        d_output_array[row] = shared_xy[tx];
    }
    if(row+blockDim.x<length)
    {
        d_output_array[row+blockDim.x] = shared_xy[tx+blockDim.x];
    }
    __syncthreads();
    if(tx == 0)
    {
        
        if(bx>0)
        {
            while(atomicAdd(&flags[bx],0)== 0)
            {
                ;
            }
        }
        prev_sum = scan_value[bx];
        scan_value[bx+1] = prev_sum + shared_xy[2*blockDim.x-1];
        __threadfence();
        atomicAdd(&flags[bx+1],1);
        
    }
    __syncthreads();
    if(row < length)
    {
        d_output_array[row] += prev_sum;
    }
    if(row+blockDim.x < length)
    {
        d_output_array[row + blockDim.x] += prev_sum;
    }
    
}


void calculate_sum_gpu(float* h_input_array, float* h_output_array_gpu, int length)
{
    //pre compute
    int thread_count = 1024;
    if(length<2048)
        thread_count = length/2;
        thread_count = round_length_to_upper_power_of_2(thread_count);

    //Allocate device memory
    float* d_input_array;
    int size = length*sizeof(float);
    cudaMalloc((void**)&d_input_array, size);

    float* d_output_array;
    cudaMalloc((void**)&d_output_array, size);

    float* scan_value;
    int number_of_blocks = ceil(length/float(2.0*thread_count));
    int scan_value_size = (number_of_blocks+1)*sizeof(float);
    cudaMalloc((void**)&scan_value, scan_value_size);
    cudaMemset((void*)&scan_value, 0, scan_value_size);

    int* flags;
    int flagSize = (number_of_blocks+1)*sizeof(int);
    cudaMalloc((void**)&flags, flagSize);
    cudaMemset((void*)&flags,0,flagSize);

    int* Dcounter;
    int DcounterSize  = sizeof(int);
    cudaMalloc((void**)&Dcounter, DcounterSize);
    cudaMemset((void*)&Dcounter,0,DcounterSize);

    //Transfer data
    cudaMemcpy(d_input_array, h_input_array, size, cudaMemcpyHostToDevice);

    //call kernel for sub arrays
    cout<<"Thread count "<<thread_count<<'\n';
    dim3 threadsPerBlock(thread_count,1,1);
    dim3 blocksPerGrid(number_of_blocks,1,1);
    int shared_mem_size = 2*thread_count*sizeof(float);

    kernel_brent_k <<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_input_array, d_output_array, scan_value, flags, Dcounter, length);

    //Transfer data
    cudaMemcpy(h_output_array_gpu, d_output_array, size, cudaMemcpyDeviceToHost);

    //Delete allocated space
    cudaFree(d_input_array);
    cudaFree(d_output_array);

}
int main()
{
    int length = 2*4090*10;
    float* h_input_array;
    float* h_output_array_cpu;

    h_input_array = new float[length];
    for(int i = 0; i < length; i++)
    {
        h_input_array[i] = 1;
    }

    h_output_array_cpu = new float[length];

    float acc_cpu = 0.0;
    for(int i = 0; i<length; i++)
    {
        acc_cpu += h_input_array[i];
        h_output_array_cpu[i] = acc_cpu;
    }
    if(PRINT_FLAG)
        print_1d_array(h_output_array_cpu, 16, "Output cpu");

    float* h_output_array_gpu = new float[length];
    auto start_gpu = get_current_time();
    calculate_sum_gpu(h_input_array, h_output_array_gpu, length);
    auto end_gpu = get_current_time();

    if(PRINT_FLAG)
        print_1d_array(h_output_array_gpu, 16, "GPU : ");

    bool eq_flag = true;
    for(int i=0; i<length; i++)
    {
        if(h_output_array_cpu[i] != h_output_array_gpu[i])
        {    
            eq_flag = false;
            cout<<"Err ar : "<<i<<" Value not equal : "<<h_output_array_cpu[i]<<"  "<<h_output_array_gpu[i]<<'\n';
            break;
        }
    }
    if (eq_flag)
        cout<<"Correct result \n";



}