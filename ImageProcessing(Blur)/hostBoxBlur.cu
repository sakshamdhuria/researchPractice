#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <chrono>
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

void boxBlurEffect(unsigned char* image_data, int width, int height, int channels, int blur_radius) {
    // Allocate temporary memory for blurred image
    unsigned char* blurred_image = new unsigned char[width * height * channels];

    // Apply box blur effect
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sumRed = 0;
            int sumGreen = 0;
            int sumBlue = 0;
            int neighboringPixelCount = 0;

            // Iterate over neighboring pixels within blur radius
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

            // Calculate average RGB values
            if (neighboringPixelCount > 0) {
                sumRed /= neighboringPixelCount;
                sumGreen /= neighboringPixelCount;
                sumBlue /= neighboringPixelCount;
            }

            // Update blurred image data
            int currentInd = (y * width + x) * channels;
            blurred_image[currentInd] = sumRed;
            blurred_image[currentInd + 1] = sumGreen;
            blurred_image[currentInd + 2] = sumBlue;
        }
    }

    // Copy blurred image data back to original image data
    std::copy(blurred_image, blurred_image + width * height * channels, image_data);

    // Free temporary memory
    delete[] blurred_image;
}

int main() {
    // Load input image
    const char* input_image = "input_image.jpg";
    int width, height, channels;
    unsigned char* image_data = stbi_load(input_image, &width, &height, &channels, 0);

    if (!image_data) {
        std::cerr << "Error loading input image";
        return 1;
    }
    
    // Define blur radius
    int boxBlurRadius = 50;

    // Measure execution time of box blur effect
    auto start_time = std::chrono::steady_clock::now();
    boxBlurEffect(image_data, width, height, channels, boxBlurRadius);
    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Save output image
    const char* output_image = "output_image.jpg";
    int result = stbi_write_jpg(output_image, width, height, channels, image_data, 100);

    if (result == 0) {
        std::cerr << "Error writing output image" << std::endl;
    } else {
        std::cout<<"Sequential: Image of size: "<<width*height*channels<<" pixels, is blurred using box blur method ("<< boxBlurRadius<< " pixel radius) within "<<duration_ms.count()<<" ms"<<std::endl;
    }

    // Free image data
    stbi_image_free(image_data);

    return 0;
}
