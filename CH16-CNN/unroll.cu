#include<cuda.h>
#include<math.h>

#include<iostream>
using namespace std;

#include"../helper_functions/data_dimension_conversion.h"
#include"../helper_functions/data_print.h"

#define PRINT_FLAG false


void cpu_unrolling_function(float*** X, float** X_unroll, int C, int H, int W, int K)
{
    int H_out = H-K+1;
    int W_out = H-K+1;

    for(int i=0;i<H_out; i++)
    {
        for(int j = 0; j<W_out; j++)
        {
            for(int c = 0; c<C; c++)
            {
                for(int p = 0; p<K; p++)
                {
                    for(int q=0; q<K; q++)
                    {
                        X_unroll[c*K*K + p*K +q][i*W_out + j] = X[c][i+p][j+q];
                    }
                }
            }
        }
    }

}

void convert_3d_to_1d(float*** input_image_3d, float* X, int C, int H, int W)
{
    for(int i =0; i<C; i++)
    {
        for(int j=0; j< H; j++)
        {
            for(int k=0; k<W; k++)
            {
                X[i*H*W + j*W + k] = input_image_3d[i][j][k];
            }
        }
    }
}

__global__ void gpu_kernel_x_unroll(float* d_X, float* d_X_Unrolled, int H, int W, int C, int K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    int H_out = H-K+1;
    int W_out = W-K+1;

    int width = H_out*W_out;
    int c,h,w,s, w_base, w_actual;
    if(row<(H_out*W_out*C))
    {
        c = row/width;
        s = row%width;

        h = s/W_out;
        w = s%W_out;

        w_base =  h*W_out + w;

        for(int p = 0; p<K; p++)
        {
            for(int q=0; q<K ; q++)
            {
                w_actual = w_base + (c*K*K + p*K + q)*H_out*W_out;
                d_X_Unrolled[w_actual] = d_X[c*H*W + (h+p)*W + (w+q)];
            }
        }
    }
}

void gpu_host_unrolling_function(float*** input_image_3d, float** gpu_unrolled, int C, int H, int W, int K)
{
    int H_out = H-K+1;
    int W_out = W-K+1;

    //Allocate device memory
    float* d_X;
    int input_size = C*H*W*sizeof(float);
    cudaMalloc((void**)&d_X, input_size);

    float* X;
    X = new float[C*H*W];

    convert_3d_to_1d(input_image_3d, X, C, H, W);

    float* d_X_Unrolled;
    int output_size = H_out*W_out*C*K*K*sizeof(float);

    cudaMalloc((void**)&d_X_Unrolled, output_size);

    //Transfer to Memory
    cudaMemcpy(d_X, X, input_size, cudaMemcpyHostToDevice);

    // Call kernel
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop,0);
    int max_threads = dev_prop.maxThreadsPerBlock;
    if(H_out*W_out*C < max_threads)
    {
        max_threads = C*H_out*W_out;
    }
    dim3 threadsPerBlock(max_threads,1,1);
    dim3 blocksPerKernel(ceil(H_out*W_out*C/float(max_threads)),1,1);

    gpu_kernel_x_unroll<<<blocksPerKernel, threadsPerBlock>>> (d_X, d_X_Unrolled, H, W, C, K);

    //Transfer to Host

    float* h_output_1d;
    h_output_1d = new float[C*K*K*H_out*W_out];

    cudaMemcpy(h_output_1d, d_X_Unrolled, output_size, cudaMemcpyDeviceToHost);

    convert_1d_to_2d(h_output_1d, gpu_unrolled, C*K*K, H_out*W_out);

    //Free space
    cudaFree(d_X);
    cudaFree(d_X_Unrolled);

    
}

int main()
{
    float*** input_image_3d;
    int C = 3;
    int H = 218;
    int W = 218;
    int K = 3;

    int H_out = H-K+1;
    int W_out = W-K+1;
    //Declate inputs
    input_image_3d = new float**[C];
    for(int i=0;i<C; i++)
    {
        input_image_3d[i] = new float*[H];
        for(int j = 0; j<H; j++)
        {
            input_image_3d[i][j] = new float[W];
            for(int k = 0; k<W; k++)
            {
                input_image_3d[i][j][k] = (i*(H*W) + j*W + k)%10;
            }
        }
    }

    //Convert 3D to 2D on cpu

    float** cpu_unrolled;

    cpu_unrolled = new float*[C*K*K];
    for(int i = 0; i<(C*K*K); i++)
    {
        cpu_unrolled[i] = new float[H_out*W_out];
        for(int j=0; j<(H_out*W_out); j++)
        {
            cpu_unrolled[i][j] = 0;
        }
    }
    cpu_unrolling_function(input_image_3d, cpu_unrolled, C, H, W, K);

    if(PRINT_FLAG)
        print_2d_array(cpu_unrolled, C*K*K, H_out*W_out, "CPU : ");

    //Convert 3d to 2d on gpu

    float** gpu_unrolled;

    gpu_unrolled = new float*[C*K*K];
    for(int i = 0; i<(C*K*K); i++)
    {
        gpu_unrolled[i] = new float[H_out*W_out];
        for(int j=0; j<(H_out*W_out); j++)
        {
            gpu_unrolled[i][j] = 0;
        }
    }

    gpu_host_unrolling_function(input_image_3d, gpu_unrolled, C, H, W, K);

    if(PRINT_FLAG)
        print_2d_array(gpu_unrolled, C*K*K, H_out*W_out, "GPU : ");

    //Check

    bool flag_equal = true;

    for(int  i = 0; i<(C*K*K) ;i++)
    {
        for(int j=0; j<(H_out*W_out); j++)
        {
            if(cpu_unrolled[i][j] != gpu_unrolled[i][j])
            {
                flag_equal = false;
                printf("Error at %d,%d ; the val %f is not eual to %f \n",i,j,cpu_unrolled[i][j], gpu_unrolled[i][j]);
                break;
            }
        }
    }
    if(flag_equal)
    {
        printf("Correct \n");
    }

}