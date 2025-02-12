#include<iostream>
using namespace std;

template <typename T>  
void display_1d(T* data, int length)
{
    for(int i=0;i<length; i++)
    {
        cout<<data[i]<<'\t';
    }
    cout<<"\n";
} 

template<typename T>
void display_2d(T** input_2d, int rows, int cols)
{
    for(int i =0 ;i <rows; i++)
    {
        for(int j=0; j<cols;j++)
            cout<<input_2d[i][j]<<'\t';
        cout<<"\n";
    }
    cout<<'\n';
}