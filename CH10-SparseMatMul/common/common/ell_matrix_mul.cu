#include<iostream>
#include<cuda.h>
using namespace std;

#define TOTAL_ROWS 4
__constant__ float d_X[TOTAL_ROWS];

__global__ void ell_matrix_mul_on_device(float* d_data, int* d_col_idx, float* d_Y, int total_elems, int rows)
{
    int current_row = blockIdx.x*blockDim.x + threadIdx.x;
    float p_sum = 0.0;
    if(current_row == 0)
    {
        for(int i =0;i<rows; i++)
        {
            printf("%f, \t", d_X[i]);
        }
        printf("\n");
    }
    if(current_row < rows)
    {
        for(int i = current_row; i<total_elems; i+=rows)
        {
            p_sum += (d_data[i] * d_X[d_col_idx[i]]);
        }
        d_Y[current_row] += p_sum;
    }
}

void ell_matrix_mul(float* h_data, int* h_col_idx, float* h_X, float* h_Y, int total_elems, int rows)
{
    //Declare space
    cout<<"Device declaration start \n";
    float* d_data;
    int input_size_float = total_elems * sizeof(float);
    cudaMalloc((void**)&d_data,input_size_float);

    int* d_col_idx;
    int input_size_int = total_elems * sizeof(int);
    cudaMalloc((void**)&d_col_idx, input_size_int);

    int output_size = rows * sizeof(float);

    float* d_Y;
    cudaMalloc((void**)&d_Y,output_size);

    cout<<"Device data declaration done \n";
    
    //Copy to device
    cout<<"Copy to device start \n";
    cudaMemcpy(d_data, h_data, input_size_float, cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, input_size_int, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_X, h_X, output_size);
    cudaMemcpy(d_Y, h_Y, output_size, cudaMemcpyHostToDevice);
    cout<<"Copy to device done \n";

    //Call device function
     cout<<"device mul start \n";
    dim3 threadsPerBlock(4,1,1);
    dim3 blocksPerGrid(1,1,1);

    ell_matrix_mul_on_device<<<blocksPerGrid,threadsPerBlock>>>(d_data, d_col_idx, d_Y, total_elems, rows);

    cout<<"Device Mul finished \n";
    //Transfer data back
    cout<<"Result copy to device start\n";
    cudaMemcpy(h_Y, d_Y, output_size, cudaMemcpyDeviceToHost);

    cout<<"data transferred back to host done \n";


}