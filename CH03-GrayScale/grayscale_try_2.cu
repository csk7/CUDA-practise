#include <stdint.h>
#include<cuda.h>
#include<math.h>
#include <stdio.h>
#include<iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


//helper function
float* convert_uint8_to_float(uint8_t* input_image, int width, int height, int channels)
{
    float *input_image_float;
    input_image_float = new float[width * height * channels];
    for(int i= 0; i<(width*height*channels); i+=1)
    {
        input_image_float[i] = float(input_image[i]);
    }
    return input_image_float;
}

uint8_t* convert_float_to_uint8(float* input_image, int width, int height, int channels)
{
    uint8_t *input_image_uint8;
    input_image_uint8 = new uint8_t[width * height * channels];
    for(int i= 0; i<(width*height*channels); i+=1)
    {
        input_image_uint8[i] = uint8_t(input_image[i]);
    }
    return input_image_uint8;
}
//device kernel function
__global__ void color_to_grayscale_kernel(float* d_input_image, float* d_output_image, int width, int height, int channels)
{
    int current_pixel =  blockIdx.x*blockDim.x + threadIdx.x;

    if((current_pixel*channels + 2) < (width*height*channels))
    {
        d_output_image[current_pixel] = 0.299 * d_input_image[current_pixel*channels] + 0.587 * d_input_image[current_pixel*channels + 1] + 0.114 * d_input_image[current_pixel*channels + 2];
    }
}

//host driver function
float* color_to_grayscale(float* input_image_float, int width, int height, int channels=3)
{
    float* d_input_image;
    float* d_output_image;

    //Declare device variables
    int size_input = width*height*channels*sizeof(float);
    cudaMalloc((void**)&d_input_image,size_input);

    int size_output = width*height*sizeof(float);
    cudaMalloc((void**)&d_output_image,size_output);

    //Copy values from host to device
    cudaMemcpy(d_input_image, input_image_float, size_input, cudaMemcpyHostToDevice);

    //Call kernel

    dim3 blockDim(256,1,1);
    dim3 gridDim(ceil(width*height*channels/256.0),1,1);

    color_to_grayscale_kernel<<<gridDim,blockDim>>>(d_input_image,d_output_image,width,height,channels);

    cudaFree(d_input_image);

    //Copy results back to host
    float *h_output_image;
    h_output_image = new float[width*height];

    cudaMemcpy(h_output_image, d_output_image, size_output, cudaMemcpyDeviceToHost);
    cudaFree(d_output_image);

    return h_output_image;
}

int main()
{
    int width, height, bpp;
    int channels = 3;

    uint8_t* input_image_1D;

    printf("Hello");

    //read image

    input_image_1D = stbi_load("image_in.png", &width, &height, &bpp, 3);


    //convert image

    float* input_image_float;

    input_image_float = convert_uint8_to_float(input_image_1D, width, height, channels);

    float* grayscale_image_float;

    grayscale_image_float = color_to_grayscale(input_image_float, width, height, channels);

    uint8_t* output_image_1D;

    output_image_1D = convert_float_to_uint8(grayscale_image_float, width, height, 1);

    // write image

    stbi_write_png("image_try2_1.png", width, height, 1, output_image_1D, width * 1);

    stbi_image_free(output_image_1D);

    return 0;
}