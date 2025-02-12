#include<iostream>
#include<cuda.h>
#include<chrono>
#include<math.h>
using namespace std;

#define MASK_WIDTH 3
__constant__ float d_M[MASK_WIDTH*MASK_WIDTH];

#define TILE_WIDTH 16

#define PRINT_FLAG false

#include "../helper_functions/data_print.h"
#include "../helper_functions/profiling_timer.h"
#include "../helper_functions/data_dimension_conversion.h"

void Conv2d_CPU(float** h_input_matrix, float** h_M, float** h_output_matrix, int height, int width)
{
    float p_sum;
    int current_x_pixel;
    int current_y_pixel;
    int offset = MASK_WIDTH/2;

    auto start_time = get_current_time();

    for(int i = 0; i< height; i++)
    {
        for(int j =0 ;j<width; j++)
        {
            p_sum = 0.0;
            for(int k_h = 0 ; k_h < MASK_WIDTH ; k_h++)
            {
                for(int k_w = 0; k_w < MASK_WIDTH ;k_w++)
                {
                    current_y_pixel = i - offset + k_h;
                    current_x_pixel = j - offset + k_w;
                    if((current_y_pixel >=0 && current_y_pixel < height) && (current_x_pixel >= 0 && current_x_pixel <width))
                    {
                        p_sum += (h_input_matrix[current_y_pixel][current_x_pixel] * h_M[k_h][k_w]);
                    }
                }
            }
            h_output_matrix[i][j] = p_sum;
        }
    }

    auto end_time = get_current_time();

    auto ms = get_time_taken_ms(end_time - start_time);
    cout<<"CPU time : "<<ms<<" ms \n";
}

__global__ void device_conv2d_kernel(float* d_input_matrix ,float* d_output_1D ,int height, int width)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by*blockDim.y + ty;
    int col = bx*blockDim.x + tx;

    //Load Shared Mem
    extern __shared__ float sharedInput[];
    if(row<height && col<width)
        sharedInput[ty*TILE_WIDTH+tx] = d_input_matrix[row*width+col];
    else  
        sharedInput[ty*TILE_WIDTH+tx] = 0;
    __syncthreads();
    //Compute  
    int start_idx_x = (bx)*blockDim.x;
    int start_idx_y = (by)*blockDim.y;

    int end_idx_x = (bx+1)*blockDim.y;
    int end_idx_y = (by+1)*blockDim.y;

    float psum = 0.0;
    int cur_idx_x;
    int cur_idx_y;
    int offset = MASK_WIDTH/2;
    int local_idx_x;
    int local_idx_y;

    for(int i=0;i<MASK_WIDTH;i++)
    {
        for(int j=0;j<MASK_WIDTH; j++)
        {
            cur_idx_x = col - offset + j;
            cur_idx_y = row - offset + i;
            if((cur_idx_x>=start_idx_x && cur_idx_x<end_idx_x) && (cur_idx_y>=start_idx_y && cur_idx_y<end_idx_y))
            {
                local_idx_x = tx - offset + j;
                local_idx_y = ty - offset + i;
                psum += (sharedInput[local_idx_y*TILE_WIDTH+ local_idx_x]*d_M[i*MASK_WIDTH+j]);
            }
            else
            {
                if((cur_idx_y < height && cur_idx_y>=0)&& (cur_idx_x < width && cur_idx_x>=0))
                    psum += (d_input_matrix[cur_idx_y*width + cur_idx_x]*d_M[i*MASK_WIDTH+j]);
            }
        }
    }

    if(row<height && col<width)
    {
        d_output_1D[row*width+col] = psum;
    }

}


void conv_2D_gpu(float** h_input_matrix, float** h_mask, float** h_output, int height, int width)
{
    //Allocate space
    float* d_input_matrix;
    int size = height*width*sizeof(float);

    float* d_output_matrix_1D;

    cudaError_t err  = cudaMalloc((void**)&d_input_matrix, size);
    if(err != cudaSuccess)
    {
        printf("Error : %s, at %d line",cudaGetErrorString(err),__LINE__);
    }
    cudaMalloc((void**)&d_output_matrix_1D, size);

    //TransferVal
    float* h_input_1D;
    h_input_1D  = convert_2d_to_1d(h_input_matrix, height, width);
    cudaMemcpy(d_input_matrix, h_input_1D, size, cudaMemcpyHostToDevice);

    float* h_mask_1D = new float[MASK_WIDTH*MASK_WIDTH];
    h_mask_1D = convert_2d_to_1d(h_mask, MASK_WIDTH, MASK_WIDTH);
    cudaMemcpyToSymbol(d_M, h_mask_1D, MASK_WIDTH*MASK_WIDTH*sizeof(float));
    //Invoke kernel
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop,0);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 blocksPerKernel(ceil(float(width)/TILE_WIDTH),ceil(float(height)/TILE_WIDTH),1);

    int sharedMemsize = TILE_WIDTH*TILE_WIDTH*sizeof(float);
    device_conv2d_kernel<<<blocksPerKernel, threadsPerBlock, sharedMemsize>>>(d_input_matrix , d_output_matrix_1D ,height, width);    

    //TransferVal
    float* h_output_1d;
    h_output_1d = new float[height*width];
    cudaMemcpy(h_output_1d, d_output_matrix_1D, size, cudaMemcpyDeviceToHost);

    convert_1d_to_2d(h_output_1d, h_output, height, width);
    //Delete Space
    cudaFree(d_input_matrix);
    cudaFree(d_output_matrix_1D);

}

int main()
{
    int height = 1920;
    int width = 1080;

    //Declare inputs and outputs
    float** h_input_matrix;
    h_input_matrix = new float*[height];
    for(int i = 0; i<height; i++)
    {
        h_input_matrix[i] = new float[width];
        for(int j = 0; j<width; j++)
        {
            h_input_matrix[i][j] = i*width + j;
        }
    }

    if (PRINT_FLAG)
        print_2d_array(h_input_matrix, height, width, "Input matrix");

    float** h_M;
    h_M = new float*[MASK_WIDTH];
    for(int i=0; i<MASK_WIDTH; i++)
    {
        h_M[i] = new float[MASK_WIDTH];
        for(int j=0; j<MASK_WIDTH; j++)
        {
            h_M[i][j] = i*MASK_WIDTH + j;
        }
    }

    if (PRINT_FLAG)
        print_2d_array(h_M, MASK_WIDTH, MASK_WIDTH, "Mask");

    float** h_output_matrix;
    h_output_matrix = new float*[height];
    for(int i = 0; i<height; i++)
    {
        h_output_matrix[i] = new float[width];
        for(int j=0; j<width; j++)
        {
            h_output_matrix[i][j] = 0;
        }
    }

    float** h_output_matrix_gpu;
    h_output_matrix_gpu = new float*[height];
    for(int i = 0; i<height; i++)
    {
        h_output_matrix_gpu[i] = new float[width];
        for(int j=0; j<width; j++)
        {
            h_output_matrix_gpu[i][j] = 0;
        }
    }

    
    //Perform Conv 2D on CPU
    Conv2d_CPU(h_input_matrix, h_M, h_output_matrix, height, width);
    conv_2D_gpu(h_input_matrix, h_M, h_output_matrix_gpu, height, width);

    if (PRINT_FLAG)
        print_2d_array(h_output_matrix, height, width, "Output matrix cpu");

    if (PRINT_FLAG)
        print_2d_array(h_output_matrix_gpu, height, width, "Output matrix gpu");

    //check if all is correct
    bool flag_correct = true;
    for(int i = 0; i<height; i++)
    {
        for(int j =0 ;j< width; j++)
        {
            if(h_output_matrix[i][j] != h_output_matrix_gpu[i][j])
            {
                cout<<"Not matching at "<<i<<" , "<<j<<'\n';
                flag_correct = false;
                break;
            }
        }
        if(flag_correct == false)
            break;
    }
    if(flag_correct == true)
    {
        cout<<"All mathced \n";
    }

}