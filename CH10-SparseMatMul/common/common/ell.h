#include<iostream>
using namespace std;

#define DEBUG true

int find_max_non_zero_row_elems(float** raw_matrix, int rows, int col)
{
    int max_non_zero_row = -1;
    for(int i=0;i<rows;i++)
    {
        int current_col_counter = 0;
        for(int j=0; j<col; j++)
        {
            if(raw_matrix[i][j] != 0)
            {
                current_col_counter+=1;
            }
        }
        if(current_col_counter > max_non_zero_row)
        {
            max_non_zero_row = current_col_counter;
        }
    }
    return max_non_zero_row;
}
class ell
{
    public:
    int num_rows;
    int total_non_zero_elements;
    float* data;
    int* col_idx;

    ell()
    {
        data = nullptr;
        col_idx = nullptr;
    }
    ell(int i_num_rows, int i_total_non_zero_elements)
    {
        num_rows = i_num_rows;
        total_non_zero_elements = i_total_non_zero_elements;
    }

    void convert_raw_to_ell(float** raw_matrix, int rows, int cols)
    {

        int max_non_zero_elems_row = find_max_non_zero_row_elems(raw_matrix, rows, cols);
        total_non_zero_elements = max_non_zero_elems_row;

        float** intermediate_matrix;
        intermediate_matrix = new float*[rows];
        for(int i =0 ; i<rows; i+=1)
        {
            intermediate_matrix[i] = new float[max_non_zero_elems_row];
            for(int j = 0; j<cols; j+=1)
            {
                intermediate_matrix[i][j] = 0;
            }
        }

        int** intermediate_col_matrix;
        intermediate_col_matrix = new int*[rows];
        for(int i =0 ; i<rows; i+=1)
        {
            intermediate_col_matrix[i] = new int[max_non_zero_elems_row];
            for(int j = 0; j<cols; j+=1)
            {
                intermediate_col_matrix[i][j] = 0;
            }
        }

        //Construct intermediate matrix
        for(int i = 0; i<rows; i++)
        {
            int non_zero_row_counter = 0;
            for(int j =0 ;j<cols; j++)
            {
                if(raw_matrix[i][j] != 0)
                {
                    intermediate_matrix[i][non_zero_row_counter] = raw_matrix[i][j];
                    intermediate_col_matrix[i][non_zero_row_counter] = j;
                    non_zero_row_counter++;
                }
            }
        }

        //Allocate space
        if(data == nullptr)
        {
            data = new float[max_non_zero_elems_row*rows];
        }
        if(col_idx == nullptr)
        {
            col_idx = new int[max_non_zero_elems_row*rows];
        }
        //Store data row
        int counter_var = 0;
        for(int i = 0; i<max_non_zero_elems_row; i++)
        {
            for(int j = 0; j<rows; j++)
            {
                data[counter_var] = intermediate_matrix[j][i];
                col_idx[counter_var] = intermediate_col_matrix[j][i];
                counter_var++;
            }
        }
        if (DEBUG)
        {
            cout<<"ELL Format data and col index: \n";
            display_1d<float>(data,max_non_zero_elems_row*rows);
            display_1d<int>(col_idx,max_non_zero_elems_row*rows);
        }
        cout<<"ELL Format data and col index set \n";
    }

};