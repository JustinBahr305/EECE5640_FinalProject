// sobel.cu
// Created by Justin Bahr on 3/24/2025.
// EECE 5640 - High Performance Computing
// Sobel Filter CUDA Kernel using uchar3

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

using namespace std;

// Sobel Kernels
__constant__ int SOBEL_X[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
};

__constant__ int SOBEL_Y[3][3] = {
    {-1, -2, -1},
    {0,  0,  0},
    {1,  2,  1}
};

// CUDA kernel to convert RGB to Grayscale
__global__ void rgbToGrayscale_uchar3(uchar3 *rgb, unsigned char *gray, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        uchar3 pixel = rgb[idx];
        gray[idx] = (unsigned char)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
    }
}

// CUDA kernel for Sobel edge detection
__global__ void sobelFilter(unsigned char *input, unsigned char *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
    {
        int Gx = 0, Gy = 0;

        // Apply Sobel operators
        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                int pixel = input[(y + i) * width + (x + j)];
                Gx += pixel * SOBEL_X[i + 1][j + 1];
                Gy += pixel * SOBEL_Y[i + 1][j + 1];
            }
        }

        // Compute gradient magnitude
        int magnitude = sqrtf(Gx * Gx + Gy * Gy);
        output[y * width + x] = (magnitude > 255) ? 255 : magnitude;
    }
}

// Function to process the image on GPU
void processImageCUDA(unsigned char *h_rgbData, unsigned char *h_outputData, int width, int height)
{
    size_t rgbSize = width * height * sizeof(uchar3);;
    size_t graySize = width * height;

    uchar3 *d_rgb;
    unsigned char *d_gray, *d_output;

    // Allocate memory on GPU
    cudaMalloc((void **)&d_rgb, rgbSize);
    cudaMalloc((void **)&d_gray, graySize);
    cudaMalloc((void **)&d_output, graySize);

    // Copy RGB data to GPU
    cudaMemcpy(d_rgb, (uchar3 *)h_rgbData, rgbSize, cudaMemcpyHostToDevice);

    // Define CUDA grid/block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Convert to grayscale
    rgbToGrayscale_uchar3<<<gridSize, blockSize>>>(d_rgb, d_gray, width, height);
    cudaDeviceSynchronize();

    // Apply Sobel filter
    sobelFilter<<<gridSize, blockSize>>>(d_gray, d_output, width, height);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_outputData, d_output, graySize, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_output);
}