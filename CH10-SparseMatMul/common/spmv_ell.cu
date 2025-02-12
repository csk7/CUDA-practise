#include<iostream>

#include<stdio.h>
#include<cuda.h>

#include<math.h>



using namespace std;

void host_direct_SpMV(float** h_A, float* h_X, float* h_Y, int rows, int cols)
{
    for(int i = 0; i<rows; i+=1)
    {
        int dot  = 0;
        for(int j=0; j<cols; j+=1)
        {
            dot += h_A[i][j] * h_X[j]; 
        }
        h_Y[i] += dot;
    }
}

int count_non_zero_elems(float** h_A, int rows, int cols)
{
    int non_zero_elem_counter = 0;
    for(int i = 0; i<rows; i+=1)
    {
        for(int j=0; j< cols; j+=1)
        {
            if(h_A[i][j] != 0)
            {
                non_zero_elem_counter += 1;
            }
        }
    }
    return non_zero_elem_counter;
}

void convert_direct_to_csr_format(float** h_A, float* csr_data, float* col_idx, float* row_ptr, int rows, int cols, int non_zero_elem_counter)
{
    
}

int main()
{
    const int rows = 4;
    const int cols = 4;

    float** h_A;
    h_A = new float*[rows];
    for(int i = 0; i<rows; i+=1)
    {
        h_A[i] = new float[cols];
        for(int j=0;j<cols; j+=1)
        {
            h_A[i][j] = 0;
        }
    }

    h_A[0][0] = 3; h_A[0][2] = 1;
    h_A[2][1] = 2; h_A[2][2] = 4; h_A[1][3] = 1;
    h_A[3][0] = 1; h_A[3][3] = 1;


    float* h_X;
    h_X = new float[rows];

    for(int i=0;i<rows; i+=1)
    {
        h_X[i] = i+1;
    }

    float* h_Y;
    h_Y = new float[rows];
    for(int i=0;i<rows; i+=1)
    {
        h_Y[i] = i+4;
    }

    host_direct_SpMV(h_A, h_X, h_Y, rows, cols);

    float* csr_data;
    float* col_idx;
    float* row_ptr;

    int non_zero_elem_counter = count_non_zero_elems(h_A, rows, cols);

    csr_data = new float[non_zero_elem_counter];
    col_idx = new float[non_zero_elem_counter];
    row_ptr = new float[non_zero_elem_counter];

    convert_direct_to_csr_format(h_A, csr_data, col_idx, row_ptr, rows, cols, non_zero_elem_counter);





    
}