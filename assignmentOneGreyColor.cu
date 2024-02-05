// message for Professor Huang : as i was watching the video it refered to a link and when i clicked
// it seemed like the course is expired but in one of the videos these code were refered so I tried 
// my best attempt at this with the information i was given 


#include <stdio.h>

__global__ rgba_to_greyscale(const uchar4* const rgbaImage, 
    unsigned char* const greyImage, int numRows, int numCols){
    //TODO 
    /*
    Fill in the kernel to convert from color to greyscale 
    the mapping from components of a uchar4 to RGBA is : 
    .x -> R; .y -> G; .z -> B; .w -> A

    The output(greyimage) at each pixel should be a result of
    applying the formula: output = .299f *R + .587f *G +.114f *B;
    note: we will be ignoring the alpha channel for this conversion

    first create a mapping from the 2d block and grid locations 
    to an absolute 2d location in the image, then use that to 
    calculate a 1d offset
    */ 
    //location of the block 
    int x= blockIdx.x * blockDim.x * threadSize.x;
    int y =   blockIdx.y * blockDim.y * threadSize.y;

    if(x<numCols && y<numRows){
        //changing 2d to 1d
        int idx = y*numCols +x;

        //getting the RGB of the image 
        unsigned char R = rgbaImage[idx].x;
        unsigned char G = rgbaImage[idx].y;
        unsigned char B = rgbaImage[idx].z;

        //using the formula
        float grey = R*0.299f + G*0.587f + B*0.114f;

        greyImage[idxd]=(unsigned char) grey;
    }

}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4* const d_rgbaImage, 
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols){
    /*
    you fill in the correct sizes for the blocksize and gridsize
    currently only one block with one thread is being launched
    */

    // i searched it up and using 32 * 32 is a common practice
    const dim3 blockSize(32,32,1); //TODO
    const dim3 gridSize((numCols+31)/32,(numRows+31)/32,1); //TODO
    rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
}

