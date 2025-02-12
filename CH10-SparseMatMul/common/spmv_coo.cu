#include<iostream>
using namespace std;

#include<cuda.h>
#include<math.h>


#include "../helper_functions/data_print.h"

#define PRINT_FLAG false

class parse_matrix_for_stats
{
    public:
    float ratio;

    int ell_row_length;
    int ell_num_elems;

    int coo_num_elems;
    parse_matrix_for_stats(float ratio)
    {
        this->ratio = ratio;
        ell_row_length = -1;
        coo_num_elems = 0;

    }
    void calculate_for_matrix(float** A, int n_rows, int n_cols)
    {
        int cur_row_length;
        for(int i=0;i<n_rows; i++)
        {
            cur_row_length = 0;
            for(int j=0;j<n_cols;j++)
            {
                if(A[i][j] != 0)
                {
                    cur_row_length += 1;
                }
            }
            if(cur_row_length>ell_row_length)
            {
                ell_row_length = cur_row_length;
            }
        }
        ell_row_length = floor(ell_row_length*ratio);
        ell_num_elems = ell_row_length*n_rows;

        //coo
        cur_row_length = 0;
        for(int i=0;i<n_rows; i++)
        {
            cur_row_length = 0;
            for(int j=0;j<n_cols;j++)
            {
                if(A[i][j] != 0)
                {
                    cur_row_length += 1;
                }
                if(cur_row_length>ell_row_length)
                {
                    coo_num_elems++;
                }
            }
        }

        printf("ELL Row length : %d, ELL elements : %d\n",ell_row_length, ell_num_elems);
        printf("COO elements : %d \n", coo_num_elems);
    }
};

class ell
{
    public:
    float* data;
    int* col;
    int n_elems;
    int max_non_zero_elem_per_row;

    ell(int n_elems, int row_length)
    {
        data = new float[n_elems];
        col = new int[n_elems];
        max_non_zero_elem_per_row = row_length;
        this->n_elems = n_elems;
    }
    void convert_2d_matrix_to_ell(float** A, int n_rows, int n_cols)
    {
        int elem_counter = 0;
        int elem_in_each_row = 0;

        float** intermediate_ell_data;
        intermediate_ell_data = new float*[n_rows];
        for(int i=0;i<n_rows;i++)
            intermediate_ell_data[i] = new float[max_non_zero_elem_per_row];

        float** intermediate_ell_col;
        intermediate_ell_col = new float*[n_rows];
        for(int i=0;i<n_rows;i++)
            intermediate_ell_col[i] = new float[max_non_zero_elem_per_row];

        for(int i =0; i<n_rows; i++)
        {   
            elem_in_each_row = 0;
            for(int j=0;j<n_cols; j++)
            {
                if(A[i][j] != 0)
                {
                    intermediate_ell_data[i][elem_in_each_row] = A[i][j];
                    intermediate_ell_col[i][elem_in_each_row] = j;
                    elem_counter++;
                    elem_in_each_row++;
                }
                if(elem_in_each_row == max_non_zero_elem_per_row)
                {
                    break;
                }
            }
            while(elem_in_each_row < max_non_zero_elem_per_row)
            {
                intermediate_ell_data[i][elem_in_each_row] = 0;
                intermediate_ell_col[i][elem_in_each_row] = 0;
                elem_counter++;
                elem_in_each_row++;
            }
        }

        int temp_elem_counter = 0;
        for(int i =0; i<max_non_zero_elem_per_row; i++)
        {   
            for(int j=0;j<n_rows; j++)
            {
                data[temp_elem_counter] = intermediate_ell_data[j][i];
                col[temp_elem_counter] = intermediate_ell_col[j][i];
                temp_elem_counter++;
            }
        }
        
    }

};

class coo
{
    public:
    float* data;
    int* col;
    int* row;
    int cut_off_length;
    int n_elems;

    coo(int n_elems, int cut_off_length)
    {
        this->cut_off_length = cut_off_length;
        this->n_elems = n_elems;

        data = new float[n_elems];
        col = new int[n_elems];
        row = new int[n_elems];
    }

    void convert_2d_matrix_to_coo(float** A, int n_rows, int n_cols)
    {
        int counter_non_zero_elem_per_row=0;
        int counter_elem =0;
        for(int i =0;i<n_rows; i++)
        {
            counter_non_zero_elem_per_row=0;
            for(int j=0;j<n_cols; j++)
            {
                if(A[i][j] != 0)
                {
                    if(counter_non_zero_elem_per_row>=cut_off_length)
                    {
                        data[counter_elem] = A[i][j];
                        col[counter_elem] = j;
                        row[counter_elem] = i;
                        counter_elem++;
                    }
                    counter_non_zero_elem_per_row++;
                }
            }
        }
    }
};


//Host SpMV
void host_SpMV(float** A, float* X, float* Y, int n_rows, int n_cols)
{
    float sum = 0.0;
    for(int i=0; i< n_rows; i++)
    {
        sum = 0.0;
        for(int j=0; j<n_cols; j++)
        {
            sum += A[i][j]*X[j];
        }
        Y[i] = sum;
    }
}

//Device Kernel
//ELL
__global__ void kernel_ell_spmv(float* d_ELL_data, int* d_ELL_col, float* d_X, float* d_Y, int n_rows, int n_elems)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int row =  bx*blockDim.x + tx;

    float sum = 0.0;
    
    if(row<n_rows)
    {
        for(int i=row;i<n_elems;i+=n_rows)
        {
            sum += d_ELL_data[i]*d_X[d_ELL_col[i]];
        }
        d_Y[row] = sum;
        
    }
}

//COO
__global__ void kernel_coo_spmv(float* d_COO_data, int* d_COO_col, int* d_COO_row, float* d_X, float* d_Y, int n_rows, int n_elems)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int row = bx*blockDim.x+tx;

    if(row<n_elems)
    {
        atomicAdd(&d_Y[d_COO_row[row]],d_COO_data[row]*d_X[d_COO_col[row]]);
    }
}

//GPU kernel calling function
void gpu_SpMV(float** A, float* X, float* Y, int n_rows, int n_cols, float ratio)
{
    //Convert data to ELL and COO
    parse_matrix_for_stats A_stats(0.75);
    A_stats.calculate_for_matrix(A, n_rows, n_cols);

    ell A_ell(A_stats.ell_num_elems, A_stats.ell_row_length);
    A_ell.convert_2d_matrix_to_ell(A, n_rows, n_cols);

    coo A_coo(A_stats.coo_num_elems, A_stats.ell_row_length);
    A_coo.convert_2d_matrix_to_coo(A, n_rows, n_cols);
    //ELL
        //Allocate space for ELL
        float* d_ELL_data;
        int* d_ELL_col;
        float* d_X;
        float* d_Y;

        int ell_size_float = A_ell.n_elems*sizeof(float);
        int ell_size_int = A_ell.n_elems*sizeof(int);

        int input_size = n_cols*sizeof(float);
        int output_size = n_rows*sizeof(float);

        cudaMalloc((void**)&d_ELL_data,ell_size_float);
        cudaMalloc((void**)&d_ELL_col, ell_size_int);

        cudaMalloc((void**)&d_X,input_size);
        cudaMalloc((void**)&d_Y,output_size);

        cudaMemset((void*)&d_Y, 0, output_size);

        //Data transfer
        cudaMemcpy(d_ELL_data, A_ell.data, ell_size_float, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ELL_col, A_ell.col, ell_size_int, cudaMemcpyHostToDevice);
        cudaMemcpy(d_X, X, input_size, cudaMemcpyHostToDevice);
        //Call kernel for ELL
        cudaDeviceProp d_prop;
        cudaGetDeviceProperties(&d_prop,0);

        int thread_count = 0;
        if(n_rows<d_prop.maxThreadsPerBlock)
        {
            thread_count = n_rows;
        }
        else
        {
            thread_count = d_prop.maxThreadsPerBlock;
        }

        dim3 threadsPerBlock(thread_count,1,1);
        dim3 blocksPerKernel(ceil(n_rows/float(thread_count)),1,1);

        kernel_ell_spmv<<<blocksPerKernel, threadsPerBlock>>>(d_ELL_data, d_ELL_col, d_X, d_Y, n_rows, A_ell.n_elems);

        

        
    //COO
        //Allocate space for COO
        float* d_COO_data;
        int* d_COO_col;
        int* d_COO_row;

        int coo_size_float = A_coo.n_elems*sizeof(float);
        int coo_size_int = A_coo.n_elems*sizeof(int);

        cudaMalloc((void**)&d_COO_data, coo_size_float);
        cudaMalloc((void**)&d_COO_col, coo_size_int);
        cudaMalloc((void**)&d_COO_row, coo_size_int);

        //Data transfer for COO
        cudaMemcpy(d_COO_data, A_coo.data, coo_size_float, cudaMemcpyHostToDevice);
        cudaMemcpy(d_COO_col, A_coo.col, coo_size_int, cudaMemcpyHostToDevice);
        cudaMemcpy(d_COO_row, A_coo.row, coo_size_int, cudaMemcpyHostToDevice);
        
        //Call kernel for COO
        thread_count = 0;
        if(A_coo.n_elems<d_prop.maxThreadsPerBlock)
        {
            thread_count = A_coo.n_elems;
        }
        else
        {
            thread_count = d_prop.maxThreadsPerBlock;
        }

        dim3 threadsPerBlock_COO(thread_count,1,1);
        dim3 blocksPerKernel_COO(ceil(A_coo.n_elems/float(thread_count)),1,1);

        kernel_coo_spmv<<<blocksPerKernel_COO, threadsPerBlock_COO>>>(d_COO_data, d_COO_col, d_COO_row, d_X, d_Y, n_rows, A_coo.n_elems);

    //Transfer Y back to HOST
    cudaMemcpy(Y, d_Y, output_size, cudaMemcpyDeviceToHost);

    //Delete device memory data
    cudaFree((void*)&d_COO_data);
    cudaFree((void*)&d_COO_col);
    cudaFree((void*)&d_COO_row);

    cudaFree((void*)&d_ELL_data);
    cudaFree((void*)&d_ELL_col);

    cudaFree((void*)&d_X);
    cudaFree((void*)&d_Y);

}

void set_A(float** A, int n_rows, int n_cols)
{
    A[0][0] = 3;
    A[0][2] = 1;

    A[2][1] = 2;
    A[2][2] = 4;
    A[2][3] = 1;

    A[3][0] = 1;
    A[3][3] = 1;
}

int main()
{
    //Declare input A,x
    int n_rows = 256;
    int n_cols = 256;

    float ratio = 0.75;

    float** A;
    A = new float*[n_rows];
    for(int i=0;i<n_rows; i++)
    {
        A[i] = new float[n_cols];
        for(int j=0; j<n_cols; j++)
        {
            A[i][j] = (j%2)*(i*j);
        }
    }
    if(PRINT_FLAG)
    {
        set_A(A, n_rows, n_cols);
    }
    float* X;
    X = new float[n_cols];
    for(int i = 0; i<n_cols; i++)
    {
        X[i] = i+1;
    }
    float* Y;
    Y = new float[n_rows];
    for(int i = 0; i<n_rows; i++)
    {
        Y[i] = 0;
    }

    //print_2d_array(A, n_rows, n_cols, "A : ");
    //print_1d_array(X, n_cols,"X :");
    //Calculate Y in Host
    host_SpMV(A, X, Y, n_rows, n_cols);
    //print_1d_array(Y, n_rows,"Y :");
    
    float* Y_gpu;
    Y_gpu = new float[n_rows];
    for(int i = 0; i<n_rows; i++)
    {
        Y_gpu[i] = 0;
    }
    //Calculate Y in GPU
    gpu_SpMV(A, X, Y_gpu, n_rows, n_cols, ratio);
    //print_1d_array(Y_gpu, n_rows,"Y Device :");

    bool flag_match = true;
    for(int i = 0; i<n_rows ; i++)
    {
        if(Y_gpu[i] != Y[i])
        {
            printf("Not matching at : %d, Val %f != %f \n", i, Y[i], Y_gpu[i]);
            flag_match = false;
            break;
        }
    }

    if(flag_match)
    {
        printf("Success \n");
    }
}