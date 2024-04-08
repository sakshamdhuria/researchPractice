#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "cuda_runtime.h"
#include <iostream>
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

__global__ void boxBlurEffect(unsigned char* image_data, unsigned char* output_data, int width, int height, int channels, int blur_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int sumRed = 0;
        int sumGreen = 0;
        int sumBlue = 0;
        int neighboringPixelCount = 0;

        for (int j = y - blur_radius; j <= y + blur_radius; j++) {
            for (int i = x - blur_radius; i <= x + blur_radius; i++) {
                if (i >= 0 && i < width && j >= 0 && j < height) {
                    int neighboringInd = (j * width + i) * channels;
                    sumRed += image_data[neighboringInd];
                    sumGreen += image_data[neighboringInd + 1];
                    sumBlue += image_data[neighboringInd + 2];
                    neighboringPixelCount++;
                }
            }
        }

        if (neighboringPixelCount > 0) {
            sumRed /= neighboringPixelCount;
            sumGreen /= neighboringPixelCount;
            sumBlue /= neighboringPixelCount;
        }

        int currentInd = (y * width + x) * channels;
        output_data[currentInd] = sumRed;
        output_data[currentInd + 1] = sumGreen;
        output_data[currentInd + 2] = sumBlue;
    }
}

int main() {
    const char* input_image = "input_image.jpg";
    int width, height, channels;

    unsigned char* image_data = stbi_load(input_image, &width, &height, &channels, 0);

    if (!image_data) {
        std::cerr << "Error loading input image" << std::endl;
        return 1;
    }

    size_t image_size = width * height * channels;
    unsigned char* output_data = new unsigned char[image_size];
    int boxBlurRadius = 50;

    unsigned char* d_image_data;
    unsigned char* d_output_data;
    cudaMalloc((void**)&d_image_data, image_size * sizeof(unsigned char));
    cudaMalloc((void**)&d_output_data, image_size * sizeof(unsigned char));

    cudaMemcpy(d_image_data, image_data, image_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);  // Each block handles a 16x16 pixel region
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    boxBlurEffect<<<gridDim, blockDim>>>(d_image_data, d_output_data, width, height, channels, boxBlurRadius);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(output_data, d_output_data, image_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    const char* output_image = "output_image.jpg";
    int result = stbi_write_jpg(output_image, width, height, channels, output_data, 100);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (result == 0) {
        std::cerr << "Error writing output image" << std::endl;
    } else {
        std::cout << "Parallel (CUDA): Image of size " << image_size << " pixels blurred using box blur method (radius " << boxBlurRadius << ") within " << milliseconds << " ms" << std::endl;
    }

    cudaFree(d_image_data);
    cudaFree(d_output_data);
    stbi_image_free(image_data);
    delete[] output_data;

    return 0;
}
