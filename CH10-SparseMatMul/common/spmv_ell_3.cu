#include<iostream>
using namespace std;

#include"common/display_helper.h"
#include"common/ell.h"
#include"common/ell_matrix_mul.cu"

int main()
{
    float** raw_matrix;
    int rows = 4;
    int cols = 4;

    //Input
    raw_matrix = new float*[rows];
    for(int i =0 ; i<rows; i+=1)
    {
        raw_matrix[i] = new float[cols];
        for(int j = 0; j<cols; j+=1)
        {
            raw_matrix[i][j] = 0;
        }
    }

    raw_matrix[0][0] = 3;
    raw_matrix[0][2] = 1;

    raw_matrix[2][1] = 2;
    raw_matrix[2][2] = 4;
    raw_matrix[2][3] = 1;

    raw_matrix[3][0] = 1;
    raw_matrix[3][3] = 1;

    cout<<"Raw matrix : \n";
    display_2d<float>(raw_matrix, rows, cols);

    float* h_X;
    h_X = new float[rows];
    h_X[0] = 1;
    h_X[1] = 2;
    h_X[2] = 3;
    h_X[3] = 4;
    cout<<"X : \n";
    display_1d<float>(h_X, rows);

    float* h_Y;
    h_Y = new float[rows];
    h_Y[0] = 0;
    h_Y[1] = 0;
    h_Y[2] = 0;
    h_Y[3] = 0;
    cout<<"Y : \n";
    display_1d<float>(h_Y, rows);

    //Converting data to ell format
    ell ell_matrix;

    ell_matrix.convert_raw_to_ell(raw_matrix, rows, cols);

    //Call device to perform Y += A*X
    ell_matrix_mul(ell_matrix.data, ell_matrix.col_idx, h_X, h_Y, ell_matrix.total_non_zero_elements*rows, rows);

    //Display Y
    cout<<"Final Result: \n";
    cout<<"Y : \n";
    display_1d<float>(h_Y, rows);
}
