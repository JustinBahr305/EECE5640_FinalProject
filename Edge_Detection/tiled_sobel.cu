// sobel.cu
// Created by Justin Bahr on 3/24/2025.
// EECE 5640 - High Performance Computing
// Tiled Sobel Filter CUDA Kernel

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

using namespace std;

// Define Sobel kernels
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

// CUDA kernel to convert RGB to Grayscale in tiles
__global__ void rgbToGrayscaleTiled(unsigned char *rgb, unsigned char *gray, int width, int height)
{
    // Define tile size and shared memory
    __shared__ unsigned char tile_rgb[16][16][3]; // RGB channels
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * 16 + tx;
    int y = blockIdx.y * 16 + ty;

    int idx = (y * width + x) * 3;

    if (x < width && y < height)
    {
        // Load RGB into shared memory
        tile_rgb[ty][tx][0] = rgb[idx];     // R
        tile_rgb[ty][tx][1] = rgb[idx + 1]; // G
        tile_rgb[ty][tx][2] = rgb[idx + 2]; // B
    }

    __syncthreads();

    if (x < width && y < height)
    {
        unsigned char r = tile_rgb[ty][tx][0];
        unsigned char g = tile_rgb[ty][tx][1];
        unsigned char b = tile_rgb[ty][tx][2];

        // Compute grayscale intensity
        gray[y * width + x] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// CUDA kernel for tiled Sobel edge detection
__global__ void sobelFilterTiled(unsigned char *input, unsigned char *output, int width, int height)
{
    // Define tile size with 1-pixel halo
    __shared__ unsigned char tile[16 + 2][16 + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // Global image index
    int imgIdx = y * width + x;

    // Shared memory coordinates (+1 for halo)
    int sharedX = tx + 1;
    int sharedY = ty + 1;

    // Load center pixels
    if (x < width && y < height)
        tile[sharedY][sharedX] = input[imgIdx];
    else
        tile[sharedY][sharedX] = 0;

    // Load halo edges (if thread is at edge of block)
    if (tx == 0 && x > 0)
        tile[sharedY][sharedX - 1] = input[y * width + (x - 1)];
    if (tx == blockDim.x - 1 && x < width - 1)
        tile[sharedY][sharedX + 1] = input[y * width + (x + 1)];
    if (ty == 0 && y > 0)
        tile[sharedY - 1][sharedX] = input[(y - 1) * width + x];
    if (ty == blockDim.y - 1 && y < height - 1)
        tile[sharedY + 1][sharedX] = input[(y + 1) * width + x];

    // Load halo corners
    if (tx == 0 && ty == 0 && x > 0 && y > 0)
        tile[sharedY - 1][sharedX - 1] = input[(y - 1) * width + (x - 1)];
    if (tx == 0 && ty == blockDim.y - 1 && x > 0 && y < height - 1)
        tile[sharedY + 1][sharedX - 1] = input[(y + 1) * width + (x - 1)];
    if (tx == blockDim.x - 1 && ty == 0 && x < width - 1 && y > 0)
        tile[sharedY - 1][sharedX + 1] = input[(y - 1) * width + (x + 1)];
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1 && x < width - 1 && y < height - 1)
        tile[sharedY + 1][sharedX + 1] = input[(y + 1) * width + (x + 1)];

    __syncthreads();

    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        int Gx = 0, Gy = 0;

        for (int i = -1; i <= 1; ++i)
        {
            for (int j = -1; j <= 1; ++j)
            {
                int pixel = tile[sharedY + i][sharedX + j];
                Gx += pixel * SOBEL_X[i + 1][j + 1];
                Gy += pixel * SOBEL_Y[i + 1][j + 1];
            }
        }

        int mag = sqrtf(Gx * Gx + Gy * Gy);
        output[imgIdx] = (mag > 255) ? 255 : mag;
    }
}

// Function to process the image on GPU
void processImageCUDA(unsigned char *h_rgbData, unsigned char *h_outputData, int width, int height)
{
    size_t rgbSize = width * height * 3;
    size_t graySize = width * height;

    unsigned char *d_rgb, *d_gray, *d_output;

    // Allocate memory on GPU
    cudaMalloc((void **)&d_rgb, rgbSize);
    cudaMalloc((void **)&d_gray, graySize);
    cudaMalloc((void **)&d_output, graySize);

    // Copy RGB data to GPU
    cudaMemcpy(d_rgb, h_rgbData, rgbSize, cudaMemcpyHostToDevice);

    // Define CUDA grid/block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Convert to grayscale
    rgbToGrayscaleTiled<<<gridSize, blockSize>>>(d_rgb, d_gray, width, height);
    cudaDeviceSynchronize();

    // Apply Sobel filter
    sobelFilterTiled<<<gridSize, blockSize>>>(d_gray, d_output, width, height);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_outputData, d_output, graySize, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_output);
}