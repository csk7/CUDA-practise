#include<iostream>
#include<cuda.h>
#include<math.h>

using namespace std;


void display_1d(float* input_1d, int length)
{
    for(int i =0 ;i <length; i++)
    {
        cout<<input_1d[i]<<'\t';
    }
    cout<<'\n';
}
void display_2d(float** input_2d, int rows, int cols)
{
    for(int i =0 ;i <rows; i++)
    {
        for(int j=0; j<cols;j++)
            cout<<input_2d[i][j]<<'\t';
        cout<<"\n";
    }
    cout<<'\n';
}

class csr_format
{
    public:
    float* data;
    float* col_idx;
    float* row_ptr;
    int total_non_zero_elems;
    int rows;

    
    csr_format(void)
    {
        ;
    }
    csr_format(int total_non_zero_elems, int rows)
    {
        total_non_zero_elems = total_non_zero_elems;
        rows = rows;
        data = new float[total_non_zero_elems];
        col_idx = new float[total_non_zero_elems];
        row_ptr = new float[rows+1];
        row_ptr[0] = 0;
    }
    csr_format(csr_format &ex_csr_dormat)
    {
        rows = ex_csr_dormat.rows;
        total_non_zero_elems = ex_csr_dormat.total_non_zero_elems;
        data = new float[total_non_zero_elems];
        col_idx = new float[total_non_zero_elems];
        row_ptr = new float[rows+1];
        data = ex_csr_dormat.data;
        col_idx = ex_csr_dormat.col_idx;
        row_ptr = ex_csr_dormat.row_ptr;
    }
    void set_vals_from_raw_data(float** raw_matrix, int rows, int cols)
    {
        int elem_index = 0;
        for(int i = 0; i<rows ; i++)
        {
            for(int j=0; j<cols; j++)
            {
                if(raw_matrix[i][j] != 0)
                {
                    data[elem_index] = raw_matrix[i][j]; 
                    col_idx[elem_index] = j;
                    elem_index+=1;
                }
                row_ptr[i+1] = elem_index;
            }
        }
    }
};

class ell_format
{
    public:
    float* data;
    int* col_idx;
    int max_non_zero_row_size;
    int rows;

    public :
    ell_format(void)
    {
        ;
    }
    ell_format(int max_non_zero_row_size, int rows)
    {
        data = new float[max_non_zero_row_size * rows];
        col_idx = new int[max_non_zero_row_size * rows];
        rows = rows;
        max_non_zero_row_size = max_non_zero_row_size;
    }

    void convert_csr_to_ell_format(csr_format csr_input_data)
    {
        float** ell_non_transpose_2d;
        ell_non_transpose_2d = new float*[rows];
        for(int i =0 ;i<rows;i++)
        {
            ell_non_transpose_2d[i] = new float[max_non_zero_row_size];
        }

        float** ell_non_transpose_col_2d;
        ell_non_transpose_col_2d = new float*[rows];
        for(int i = 0 ;i<rows;i++)
        {
            ell_non_transpose_col_2d[i] = new float[max_non_zero_row_size];
        }


        int csr_data_pointer = 0;
        for(int i = 0; i<rows; i+=1)
        {
            int elements_added_count = 0;
            for(int j = csr_input_data.row_ptr[i]; j<csr_input_data.row_ptr[i+1]; j++)
            {
                ell_non_transpose_2d[i][elements_added_count] = csr_input_data.data[csr_data_pointer];
                ell_non_transpose_col_2d[i][elements_added_count] = csr_input_data.col_idx[csr_data_pointer];
                elements_added_count += 1;
                csr_data_pointer += 1;
            }
            if(elements_added_count < max_non_zero_row_size)
            {
                ell_non_transpose_2d[i][elements_added_count] = 0;
                ell_non_transpose_col_2d[i][elements_added_count] = 0;
                elements_added_count += 1;
            }
        }

        int data_ptr = 0;
        for(int i = 0 ;i<max_non_zero_row_size; i++)
        {
            for(int j =0; j<rows; j++)
            {
                data[data_ptr] = ell_non_transpose_2d[j][i];
                col_idx[data_ptr] = ell_non_transpose_col_2d[j][i];
                data_ptr++;
            }
        }
    }
};



int count_non_zero_elem(float** raw_matrix, int rows, int cols)
{
    int total_non_zero_elems = 0;
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
        {
            if(raw_matrix[i][j] != 0)
            {
                total_non_zero_elems +=1 ;
            }
        }
    }
    return total_non_zero_elems;
    cout<<"Non zero element :"<<total_non_zero_elems;
}

int find_max_non_zero_row(csr_format csr_matrix)
{
    int max_non_zero_row_size = -1;
    cout<<max_non_zero_row_size;
    int row_size;
    for(int i=0;i<csr_matrix.rows;i++)
    {
        cout<<max_non_zero_row_size;
        row_size = csr_matrix.row_ptr[i+1] - csr_matrix.row_ptr[i];
        if(row_size > max_non_zero_row_size)
        {
            max_non_zero_row_size = row_size;
        }
    }
    
    return max_non_zero_row_size;
}





csr_format generate_csr_input_sparse_matrix(float** raw_data, int rows, int cols)
{
    int total_elems = count_non_zero_elem(raw_data, rows, cols);

    csr_format csr_input(total_elems, rows);
    
    csr_input.set_vals_from_raw_data(raw_data, rows, cols);
    cout<<"Before returning : \n";
    display_1d(csr_input.data, csr_input.total_non_zero_elems);
    return csr_input;
}

ell_format generate_ell_format_from_csr(csr_format csr_input)
{
    cout<<"Hello";
    int max_non_zero_row = find_max_non_zero_row(csr_input);
    ell_format ell_input(max_non_zero_row, csr_input.rows);

    ell_input.convert_csr_to_ell_format(csr_input);

    return ell_input;

}


__global__ void spmv_ell(float* d_ell_data_array, int* d_ell_col_idx_array, float* d_x, float* d_y, int num_elements, int n_rows)
{
    int row = blockIdx.x *blockDim.x + threadIdx.x;

    float p_val = 0;
    for(int i = row; i < num_elements; i+=n_rows)
    {
        p_val += d_ell_data_array[i] * d_x[d_ell_col_idx_array[i]];
    }
    if(row < n_rows)
        d_y[row] = p_val;
}

void solve_equation(ell_format ell_input_data, float* x_input, float* y_output, int rows)
{
    //declare variables
    float* d_ell_data_array;

    int num_elements = ell_input_data.max_non_zero_row_size * rows;
    int size_data = num_elements * sizeof(float);
    int size_data_int = num_elements * sizeof(int);
    cudaMalloc((void**)&d_ell_data_array,size_data);

    int* d_ell_col_idx_array;

    cudaMalloc((void**)&d_ell_col_idx_array, size_data_int);

    int size_inputs = rows * sizeof(float);

    float* d_x;
    cudaMalloc((void**)&d_x, size_inputs);

    float* d_y;
    cudaMalloc((void**)&d_y, size_inputs);

    //copy to device

    cudaMemcpy(d_ell_data_array, ell_input_data.data, size_data, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ell_col_idx_array, ell_input_data.col_idx, size_data_int, cudaMemcpyDeviceToHost);

    cudaMemcpy(d_x, x_input, size_inputs, cudaMemcpyHostToDevice);
    //invoke kernel
    dim3 dimGrid(ceil(rows/2.0),1,1);
    dim3 dimBlock(2,1,1);

    spmv_ell<<<dimGrid,dimBlock>>>(d_ell_data_array, d_ell_col_idx_array, d_x, d_y, num_elements, rows);
    //copy results back
    cudaMemcpy(y_output, d_y, size_inputs, cudaMemcpyDeviceToHost);

    return;

}
int main()
{
    //declare a matrix, a vector
    float** h_input_sparse_matrix;

    int rows = 4;
    int cols = 4;
    cout<<"Hello";

    h_input_sparse_matrix = new float*[rows];
    for(int i =0;i<rows;i++)
    {
        h_input_sparse_matrix[i] = new float[cols];
        for(int j = 0; j<cols; j++)
        {
            h_input_sparse_matrix[i][j] = 0;
        }
    }
    h_input_sparse_matrix[0][0] = 3;
    h_input_sparse_matrix[0][2] = 1;

    h_input_sparse_matrix[2][1] = 2;
    h_input_sparse_matrix[2][2] = 4;
    h_input_sparse_matrix[2][3] = 1;

    h_input_sparse_matrix[3][0] = 1;
    h_input_sparse_matrix[3][3] = 1;

    cout<<"Hello";
    //Generate the csr format
    csr_format csr_input_data;
    csr_input_data = generate_csr_input_sparse_matrix(h_input_sparse_matrix, rows, cols);
    cout<<"csr data: \n";
    cout<<csr_input_data.total_non_zero_elems<<'\n';
    //display_1d(csr_input_data.data, csr_input_data.total_non_zero_elems);
    /*
    //Generate ELL format
    ell_format ell_input_data;
    ell_input_data = generate_ell_format_from_csr(csr_input_data);
    /*
    //Call host function that call gpu function and returns result
    
    float* x_input;
    x_input = new float[rows];

    x_input[0] = 1;
    x_input[1] = 2;
    x_input[2] = 3;
    x_input[3] = 4;


    float* y_output;
    y_output = new float[rows];

    y_output[0] = 0;
    y_output[1] = 0;
    y_output[2] = 0;
    y_output[3] = 0;
    cout<<"Sparse matrix : ";
    display_2d(h_input_sparse_matrix, rows, cols);
    cout<<"Input :";
    display_1d(x_input, rows);
    cout<<"Output :";
    display_1d(y_output, rows);


    solve_equation(ell_input_data, x_input, y_output, rows);

    //Display final y
    display_1d(y_output, rows);
    */


}