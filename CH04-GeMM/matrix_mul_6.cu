#include<iostream>
#include<cuda.h>

using namespace std;

#include<math.h>
#include<chrono>

#include"../helper_functions/data_print.h"
#include "../helper_functions/data_dimension_conversion.h"
#include "../helper_functions/profiling_timer.h"

#define TILE_WIDTH 2
#define PRINT_FLAG true

void host_matrix_mul(float** h_M, float** h_N, float** h_P, int width)
{
    float sum = 0;
    for(int i = 0; i < width; i++)
    {
        for(int j = 0; j< width; j++)
        {
            sum = 0;
            for(int k = 0; k<width; k++)
            {
                sum+= h_M[i][k] * h_N [k][j];
            }
            h_P[i][j] = sum;
        }
    }
}


__global__ void device_gemm(float* d_M, float* d_N, float* d_P, int width)
{
    extern __shared__ float shared_data[];
    float* shared_M = &shared_data[0];
    float* shared_N = &shared_data[TILE_WIDTH*TILE_WIDTH];
    

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by*blockDim.y + ty;
    int col = bx*(2*blockDim.x) + tx;

    int N_TILE_WIDTH = 2*TILE_WIDTH;

    
    float p_sum[2];
    for(int i=0; i<2; i++)
        p_sum[i] = 0.0;

    
    for(int ph = 0; ph<ceil(float(width)/TILE_WIDTH); ph++)
    {
        //Load data
        if(row<width && (ph*TILE_WIDTH + tx)<width)
            shared_M[ty*TILE_WIDTH +tx] = d_M[row*width + ph*TILE_WIDTH + tx];
        else
            shared_M[ty*TILE_WIDTH +tx] = 0;

        if((ph*TILE_WIDTH + ty) < width && col <width)
            shared_N[ty*N_TILE_WIDTH +tx] = d_N[(ph*TILE_WIDTH + ty)*width + col];
        else
            shared_N[ty*N_TILE_WIDTH +tx] = 0;

        //Extra
        if((ph*TILE_WIDTH + ty) < width && (col + TILE_WIDTH) <width)
            shared_N[ty*N_TILE_WIDTH +tx + TILE_WIDTH] = d_N[(ph*TILE_WIDTH + ty)*width + (col + TILE_WIDTH)];
        else
            shared_N[ty*N_TILE_WIDTH +tx+ TILE_WIDTH] = 0;
        __syncthreads();
        //Calculate psum
        
        for(int k = 0;k<TILE_WIDTH;k++)
        {
            p_sum[0] += shared_M[ty*TILE_WIDTH+k]*shared_N[k*N_TILE_WIDTH+tx];
            p_sum[1] += shared_M[ty*TILE_WIDTH+k]*shared_N[k*N_TILE_WIDTH+tx+TILE_WIDTH];
        }
        __syncthreads();
    }
    if(row<width && col<width)
    {
        d_P[row*width+col] = p_sum[0];
    }
    if(row<width && (col+TILE_WIDTH)<width)
    {
        d_P[row*width+(col+TILE_WIDTH)] = p_sum[1];
    }

}

void gpu_matrix_mul(float** h_M, float** h_N, float** h_P_gpu, int width)
{
    //Declare d variables
    int matrix_size = width*width*sizeof(float);

    float* d_M;
    cudaError_t err = cudaMalloc((void**)&d_M, matrix_size);
    if(err != cudaSuccess)
    {
        printf("Error : %s at %d \n", cudaGetErrorString(err),__LINE__);
    }

    float* d_N;
    cudaMalloc((void**)&d_N, matrix_size);

    float* d_P;
    cudaMalloc((void**)&d_P, matrix_size);
    cudaMemset((void*)&d_P, 255, matrix_size);
    //Copy data
    float* h_M_1D;
    h_M_1D = convert_2d_to_1d(h_M, width, width);
    cudaMemcpy(d_M, h_M_1D, matrix_size, cudaMemcpyHostToDevice);

    float* h_N_1D;
    h_N_1D = convert_2d_to_1d(h_M, width, width);
    cudaMemcpy(d_N, h_N_1D, matrix_size, cudaMemcpyHostToDevice);

    //invoke kernel
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 blocksPerGrid(ceil(float(width/2)/TILE_WIDTH), ceil(float(width)/TILE_WIDTH), 1);

    int shared_mem_size = 3*TILE_WIDTH*TILE_WIDTH*sizeof(float);

    device_gemm<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>> (d_M, d_N, d_P, width);

    //copy data

    float* h_P_1D = new float[width*width];
    cudaMemcpy(h_P_1D, d_P, matrix_size, cudaMemcpyDeviceToHost);
    //Free space
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    convert_1d_to_2d(h_P_1D, h_P_gpu, width, width);
}



void matching_matrix(float** host_matrix, float** device_matrix, int width)
{
    for(int i = 0; i<width ; i++)
    {
        for(int j = 0; j<width ; j++)
        {
            if(host_matrix[i][j] !=  device_matrix[i][j])
            {
                cout<<"Error at ["<<i<<", "<<j<<"]; Value in Host : "<<host_matrix[i][j]<<" != Value at Device : "<<device_matrix[i][j]<<'\n';
                return;
            }
        }
    }
    cout<<"Succesful Match!!! \n";
}



int main()
{
    int width = 7;

    //declare host space
    float** h_M;
    h_M = new float*[width];
    for(int i =0; i<width; i++)
    {
        h_M[i] = new float[width];
        for(int j = 0; j<width; j++)
        {
            h_M[i][j] = i*width + j;
        }
    }
    if(PRINT_FLAG)
        print_2d_array(h_M, width, width, "h_ M : ");

    float** h_N;
    h_N = new float*[width];
    for(int i =0; i<width; i++)
    {
        h_N[i] = new float[width];
        for(int j = 0; j<width; j++)
        {
            h_N[i][j] = i*width + j;
        }
    }
    if(PRINT_FLAG)
        print_2d_array(h_N, width, width, "h_ N : ");

    float** h_P_cpu;
    h_P_cpu = new float*[width];
    for(int i =0; i<width; i++)
    {
        h_P_cpu[i] = new float[width];
    }
    float** h_P_gpu;
    h_P_gpu = new float*[width];
    for(int i =0; i<width; i++)
    {
        h_P_gpu[i] = new float[width];
    }

    //host_mul
    auto cpu_start = get_current_time();
    host_matrix_mul(h_M, h_N, h_P_cpu, width);
    auto cpu_end = get_current_time();

    auto cpu_ms = get_time_taken_ms(cpu_end - cpu_start);
    cout<<"CPU Time : "<<cpu_ms<<'\n';

    if(PRINT_FLAG)
        print_2d_array(h_P_cpu, width, width, "CPU array");

    //gpu_mul
    auto gpu_start = get_current_time();
    gpu_matrix_mul(h_M, h_N, h_P_gpu, width);
    auto gpu_end = get_current_time();

    auto gpu_ms = get_time_taken_ms(gpu_end - gpu_start);
    cout<<"GPU Time : "<<gpu_ms<<'\n';

    if(PRINT_FLAG)
        print_2d_array(h_P_gpu, width, width, "GPU array");

    //checking
    matching_matrix(h_P_cpu, h_P_gpu, width);

    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop,0);
    cout<<"SharedMem : "<<dev_prop.sharedMemPerBlock<<'\n';
}