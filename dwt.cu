/* 
 * Copyright (c) 2009, Jiri Matela
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <fcntl.h>
#include <assert.h>
#include <errno.h>
#include <sys/time.h>
#include <unistd.h>
#include <error.h>

#define DWT_BLOCK_SIZE_X 32
#define DWT_BLOCK_SIZE_Y 16

#include "dwt.h"
#include "common.h"
#include "dwt97_kernel.cu"

inline void fdwt(dim3 grid, dim3 threads, float *in, float *tempBuf, int width, int height)
{
        fdwt97<float><<<grid, threads>>>(in, tempBuf, width, height, dwt97);
        cudaCheckAsyncError("Forward DWT 9/7 kernel");
}

inline void fdwt(dim3 grid, dim3 threads, int *in, int *tempBuf, int width, int height)
{
        fdwt97<int><<<grid, threads>>>(in, tempBuf, width, height, dwt53);
        cudaCheckAsyncError("Forward DWT 5/3 kernel");
}

template<typename T>
int nStage2dFDWT(T * in, T * tempBuf, int pixWidth, int pixHeight, int stages)
{
    int i;
    int width  = 2 * pixWidth;
    int height = 2 * pixHeight;
    dim3 threads(DWT_BLOCK_SIZE_X, DWT_BLOCK_SIZE_Y);
    dim3 grid;
    struct timeval tv_start;
    struct timeval tv_end;
    float duration;

    CTIMERINIT;

    printf("*** %d stages of 2D forward DWT:\n", stages);
    gettimeofday(&tv_start, NULL);
    for (i = 0; i < stages; i++) {
        width       = DIVANDRND(width, 2);
        height      = DIVANDRND(height, 2);
        grid.x      = DIVANDRND(width,DWT_BLOCK_SIZE_X);
        grid.y      = DIVANDRND(height,(DWT_BLOCK_SIZE_Y<<1)); // there is 2x more samples than threads on Y axis

        printf("  * Forward 2D DWT, Stage %d:\n", i);
        printf("    Img dim: \t\t%d x %d\n", width, height);
        printf("    Block dim (x,y): \t%d x %d\n    Grid dim(x,y): \t%d x %d\n", threads.x, threads.y, grid.x, grid.y);

        CTIMERSTART(cstart);
        fdwt(grid, threads, in, tempBuf, width, height);
        CTIMERSTOP(cstop);

        printf("    fdwt972d kernel: \t%f ms\n", elapsedTime);

        CTIMERSTART(cstart);
        cudaMemcpy(in, tempBuf, width*height*sizeof(T), cudaMemcpyDeviceToDevice);
        cudaCheckError("Memcopy device to device");
        CTIMERSTOP(cstop);
        printf("    memcpy *tempBuf to *in: \t%f ms, BW: \t%f GB/s\n", elapsedTime,
        ((float)(width*height*sizeof(T)/1024.0f/1024.0f/1024.0f))/((float)(elapsedTime/1000)));
    }
    gettimeofday(&tv_end, NULL);
    duration = tv_end.tv_usec - tv_start.tv_usec;
    printf("> Overal duration of %d stages: %f ms\n", stages, duration/1000.0f);
    
    return 0;
}
template int nStage2dFDWT<float>(float * in, float * tempBuf, int pixWidth, int pixHeight, int stages);
template int nStage2dFDWT<int>(int * in, int * tempBuf, int pixWidth, int pixHeight, int stages);

inline void rdwt(dim3 grid, dim3 threads, float *in, float *tempBuf, int width, int height)
{
        rdwt97<float><<<grid, threads>>>(in, tempBuf, width, height, dwt97);
        cudaCheckAsyncError("Forward DWT 9/7 kernel");
}

inline void rdwt(dim3 grid, dim3 threads, int *in, int *tempBuf, int width, int height)
{
        rdwt97<int><<<grid, threads>>>(in, tempBuf, width, height, dwt53);
        cudaCheckAsyncError("Forward DWT 5/3 kernel");
}

template<typename T>
int nStage2dRDWT(T *in, T *tempBuf, int pixWidth, int pixHeight, int stages)
{
    struct stageDim {
        int x;
        int y;
    };

    struct stageDim *sd;
    int i;
    int width;
    int height;
    dim3 threads(DWT_BLOCK_SIZE_X, DWT_BLOCK_SIZE_Y);
    dim3 grid;

    CTIMERINIT;

    sd = (struct stageDim *)malloc(stages * sizeof(struct stageDim));

    sd[0].x = pixWidth;
    sd[0].y = pixHeight;

    // compute stage dimensions
    for (i = 1; i < stages; i++) {
        sd[i].x = DIVANDRND(sd[i-1].x, 2);
        sd[i].y = DIVANDRND(sd[i-1].y, 2);
    }

    printf("*** %d stages of 2D reverse DWT:\n", stages);
    for (i = stages-1; i >=0; i--) {
        width       = sd[i].x;
        height      = sd[i].y;
        grid.x      = DIVANDRND(width,DWT_BLOCK_SIZE_X);
        grid.y      = DIVANDRND(height,(DWT_BLOCK_SIZE_Y<<1)); // there is 2x more samples than threads on Y axis

        printf("  * Reverse 2D DWT, Stage %d:\n", i);
        printf("    Img dim: \t\t%d x %d\n", width, height);
        printf("    Block dim (x,y): \t%d x %d\n    Grid dim(x,y): \t%d x %d\n", threads.x, threads.y, grid.x, grid.y);

        CTIMERSTART(cstart);
        rdwt(grid, threads, in, tempBuf, width, height);
        cudaCheckAsyncError("Reverse DWT kernel");
        CTIMERSTOP(cstop);

        printf("    rdwt972d kernel: \t%f ms\n", elapsedTime);

        CTIMERSTART(cstart);
        cudaMemcpy(in, tempBuf, width*height*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaCheckError("Memcopy device to device");
        CTIMERSTOP(cstop);

        printf("    memcpy *tempBuf to *in: \t%f ms, BW: \t%f GB/s\n", elapsedTime,
        ((float)(width*height*sizeof(T)/1024.0f/1024.0f/1024.0f))/((float)(elapsedTime/1000)));
    }
    free(sd);

    return 0;
}

int forwardDWT97(float * in, float *out, int pixWidth, int pixHeight, int curStage, int stages)
{
    //timing 
    CTIMERINIT;

    int samplesNum = pixWidth*pixHeight;
    float inputSize = samplesNum*sizeof(float);

    // Kernell 1D DWT
    dim3 threads(DWT_BLOCK_SIZE_X, DWT_BLOCK_SIZE_Y);
    dim3 grid(DIVANDRND(pixWidth,DWT_BLOCK_SIZE_X), DIVANDRND(pixHeight,(2*DWT_BLOCK_SIZE_Y)));
    printf("Block dim: %d x %d\n Grid dim: %d x %d\n", threads.x, threads.y, grid.x, grid.y);

    CTIMERSTART(cstart);
    fdwt97<float><<<grid, threads>>>(in, out, pixWidth, pixHeight, dwt97);
    cudaCheckAsyncError("Forward DWT 9/7 kernel");
    CTIMERSTOP(cstop);


    printf("cuda fdwt972d kernel %f ms\n", elapsedTime);
    printf("cuda device to device %f ms - %f GB/s\n", elapsedTime,
           ((float)(inputSize/1024.0f/1024.0f/1024.0f))/((float)(elapsedTime/1000)));

    return 0;
}

int reverseDWT97(float * in, float *out, int pixWidth, int pixHeight, int curStage, int stages)
{
    //timing 
    CTIMERINIT;

    int samplesNum = pixWidth*pixHeight;

    // Kernell
    dim3 threads(DWT_BLOCK_SIZE_X, DWT_BLOCK_SIZE_Y);
    dim3 grid(DIVANDRND(pixWidth,DWT_BLOCK_SIZE_X), DIVANDRND(pixHeight,(2*DWT_BLOCK_SIZE_Y)));
    printf("Block dim: %d x %d\n Grid dim: %d x %d\n", threads.x, threads.y, grid.x, grid.y);
    assert(samplesNum % 2 == 0);

    CTIMERSTART(cstart);
    rdwt97<float><<<grid, threads>>>(in, out,pixWidth, pixHeight, dwt97);
    cudaCheckAsyncError("Reverse DWT 9/7 kernel");
    CTIMERSTOP(cstop);

    printf("cuda rdwt97 kernel %f ms\n", elapsedTime);

    return 0;
}

void samplesToChar(unsigned char * dst, float * src, int samplesNum)
{
    int i;

    for(i = 0; i < samplesNum; i++) {
        float r = (src[i]+0.5f) * 255;
        if (r > 255) r = 255; 
        if (r < 0)   r = 0; 
        dst[i] = (unsigned char)r;
    }
}

void samplesToChar(unsigned char * dst, int * src, int samplesNum)
{
    int i;

    for(i = 0; i < samplesNum; i++) {
        int r = src[i]+128;
        if (r > 255) r = 255;
        if (r < 0)   r = 0; 
        dst[i] = (unsigned char)r;
    }
}

template<typename T>
int writeNStage2DDWT(T *component_cuda, int pixWidth, int pixHeight, 
                     int stages, const char * filename, const char * suffix) 
{
    struct band {
        int dimX; 
        int dimY;
    };
    struct dimensions {
        struct band LL;
        struct band HL;
        struct band LH;
        struct band HH;
    };

    unsigned char * result;
    T *src, *dst;
    int i,s;
    int size;
    int offset;
    int yOffset;
    int samplesNum = pixWidth*pixHeight;
    struct dimensions * bandDims;

    bandDims = (struct dimensions *)malloc(stages * sizeof(struct dimensions));

    bandDims[0].LL.dimX = DIVANDRND(pixWidth,2);
    bandDims[0].LL.dimY = DIVANDRND(pixHeight,2);
    bandDims[0].HL.dimX = pixWidth - bandDims[0].LL.dimX;
    bandDims[0].HL.dimY = bandDims[0].LL.dimY;
    bandDims[0].LH.dimX = bandDims[0].LL.dimX;
    bandDims[0].LH.dimY = pixHeight - bandDims[0].LL.dimY;
    bandDims[0].HH.dimX = bandDims[0].HL.dimX;
    bandDims[0].HH.dimY = bandDims[0].LH.dimY;

    for (i = 1; i < stages; i++) {
        bandDims[i].LL.dimX = DIVANDRND(bandDims[i-1].LL.dimX,2);
        bandDims[i].LL.dimY = DIVANDRND(bandDims[i-1].LL.dimY,2);
        bandDims[i].HL.dimX = bandDims[i-1].LL.dimX - bandDims[i].LL.dimX;
        bandDims[i].HL.dimY = bandDims[i].LL.dimY;
        bandDims[i].LH.dimX = bandDims[i].LL.dimX;
        bandDims[i].LH.dimY = bandDims[i-1].LL.dimY - bandDims[i].LL.dimY;
        bandDims[i].HH.dimX = bandDims[i].HL.dimX;
        bandDims[i].HH.dimY = bandDims[i].LH.dimY;
    }

#if 0
    printf("Original image pixWidth x pixHeight: %d x %d\n", pixWidth, pixHeight);
    for (i = 0; i < stages; i++) {
        printf("Stage %d: LL: pixWidth x pixHeight: %d x %d\n", i, bandDims[i].LL.dimX, bandDims[i].LL.dimY);
        printf("Stage %d: HL: pixWidth x pixHeight: %d x %d\n", i, bandDims[i].HL.dimX, bandDims[i].HL.dimY);
        printf("Stage %d: LH: pixWidth x pixHeight: %d x %d\n", i, bandDims[i].LH.dimX, bandDims[i].LH.dimY);
        printf("Stage %d: HH: pixWidth x pixHeight: %d x %d\n", i, bandDims[i].HH.dimX, bandDims[i].HH.dimY);
    }
#endif
    
    size = samplesNum*sizeof(T);
    cudaMallocHost((void **)&src, size);
    cudaCheckError("Malloc host");
    dst = (T*)malloc(size);
    memset(src, 0, size);
    memset(dst, 0, size);
    result = (unsigned char *)malloc(samplesNum);
    cudaMemcpy(src, component_cuda, size, cudaMemcpyDeviceToHost);
    cudaCheckError("Memcopy device to host");

    // LL Band
    size = bandDims[stages-1].LL.dimX * sizeof(T);
    for (i = 0; i < bandDims[stages-1].LL.dimY; i++) {
        memcpy(dst+i*pixWidth, src+i*bandDims[stages-1].LL.dimX, size);
    }

    for (s = stages - 1; s >= 0; s--) {
        // HL Band
        size = bandDims[s].HL.dimX * sizeof(T);
        offset = bandDims[s].LL.dimX * bandDims[s].LL.dimY;
        for (i = 0; i < bandDims[s].HL.dimY; i++) {
            memcpy(dst+i*pixWidth+bandDims[s].LL.dimX,
                src+offset+i*bandDims[s].HL.dimX, 
                size);
        }

        // LH band
        size = bandDims[s].LH.dimX * sizeof(T);
        offset += bandDims[s].HL.dimX * bandDims[s].HL.dimY;
        yOffset = bandDims[s].LL.dimY;
        for (i = 0; i < bandDims[s].HL.dimY; i++) {
            memcpy(dst+(yOffset+i)*pixWidth,
                src+offset+i*bandDims[s].LH.dimX, 
                size);
        }

        //HH band
        size = bandDims[s].HH.dimX * sizeof(T);
        offset += bandDims[s].LH.dimX * bandDims[s].LH.dimY;
        yOffset = bandDims[s].HL.dimY;
        for (i = 0; i < bandDims[s].HH.dimY; i++) {
            memcpy(dst+(yOffset+i)*pixWidth+bandDims[s].LH.dimX,
                src+offset+i*bandDims[s].HH.dimX, 
                size);
        }
    }

    /* Write component */
    samplesToChar(result, dst, samplesNum);

    char outfile[strlen(filename)+strlen(suffix)];
    strcpy(outfile, filename);
    strcpy(outfile+strlen(filename), suffix);
    i = open(outfile, O_CREAT|O_WRONLY, 0644);
    if (i == -1) {
        error(0,errno,"cannot access %s", outfile);
        return -1;
    }
    printf("\nWriting to %s (%d x %d)\n", outfile, pixWidth, pixHeight);
    write(i, result, samplesNum);
    close(i);

    cudaFreeHost(src);
    cudaCheckError("Cuda free host memory");
    free(dst);
    free(result);
    free(bandDims);

    return 0;
}
template int writeNStage2DDWT<float>(float *component_cuda, int pixWidth, int pixHeight, int stages, const char * filename, const char * suffix); 
template int writeNStage2DDWT<int>(int *component_cuda, int pixWidth, int pixHeight, int stages, const char * filename, const char * suffix); 
