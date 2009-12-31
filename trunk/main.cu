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

#include <unistd.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <getopt.h>

#include "common.h"
#include "components.h"
#include "dwt.h"

int getImg(char * srcFilename, unsigned char *srcImg, int inputSize)
{
    printf("Loading ipnput: %s\n", srcFilename);

    //read image
    int i = open(srcFilename, O_RDONLY, 0644);
    if (i == -1) { 
        error(0,errno,"cannot access %s", srcFilename);
        return -1;
    }
    int ret = read(i, srcImg, inputSize);
    printf("precteno %d, inputsize %d\n", ret, inputSize);
    close(i);

    return 0;
}


void usage() {
    printf("dwt [otpions] src_img.rgb <out_img.dwt>\n\
  -d, --dimension\t\tdimensions of src img, e.g. 1920x1080\n\
  -c, --components\t\tnumber of color components, default 3\n\
  -b, --depth\t\t\tbit depth, default 8\n\
  -l, --level\t\t\tDWT level, default 3\n\
  -D, --device\t\t\tcuda device\n");
}

int main(int argc, char **argv) 
{
    int optindex = 0;
    char ch;
    struct option longopts[] = {
        {"dimension",   required_argument, 0, 'd'}, //dimensions of src img
        {"components",  required_argument, 0, 'c'}, //numger of components of src img
        {"depth",       required_argument, 0, 'b'}, //bit depth of src img
        {"level",       required_argument, 0, 'l'}, //level of dwt
        {"device",      required_argument, 0, 'D'}, //cuda device
        {"help",        no_argument,       0, 'h'}  
    };

    int pixWidth  = 0; //<real pixWidth
    int pixHeight = 0; //<real pixHeight
    int compCount = 3; //number of components; 3 for RGB or YUV, 4 for RGBA
    int bitDepth  = 8; 
    int dwtLvls   = 3; //default numuber of DWT levels
    int device    = 0;
    int mantissa  = 0;
    int exponent  = 0;
    char * srcFilename;
    char * outFilename;
    char * pos;

    while ((ch = getopt_long(argc, argv, "d:c:b:l:D:h", longopts, &optindex)) != -1) {
        switch (ch) {
        case 'd':
            pixWidth = atoi(optarg);
            pos = strstr(optarg, "x");
            if (pos == NULL || pixWidth == 0 || (strlen(pos) >= strlen(optarg))) {
                usage();
                return -1;
            }
            pixHeight = atoi(pos+1);
            break;
        case 'c':
            compCount = atoi(optarg);
            break;
        case 'b':
            bitDepth = atoi(optarg);
            break;
        case 'l':
            dwtLvls = atoi(optarg);
            break;
        case 'D':
            device = atoi(optarg);
            break;
        case 'h':
            usage();
            return 0;
        case '?':
            return -1;
        default :
            usage();
            return -1;
        }
    }
	argc -= optind;
	argv += optind;

    if (argc == 0) { // at least one filename is expected
        printf("Please supply src file name\n");
        usage();
        return -1;
    }

    if (pixWidth <= 0 || pixHeight <=0) {
        printf("Wrong or missing dimensions\n");
        usage();
        return -1;
    }


    // device init
    int devCount;
    cudaGetDeviceCount(&devCount);
    cudaCheckError("Get device count");
    if (devCount == 0) {
        printf("No CUDA enabled device\n");
        return -1;
    } 
    if (device < 0 || device > devCount -1) {
        printf("Selected device %d is out of bound. Devices on your system are in range %d - %d\n", 
               device, 0, devCount -1);
        return -1;
    }
    cudaDeviceProp devProp;                                          
    cudaGetDeviceProperties(&devProp, device);  
    cudaCheckError("Get device properties");
    if (devProp.major < 1) {                                         
        printf("Device %d does not support CUDA\n", device);
        return -1;
    }                                                                   
    printf("Using device %d: %s\n", device, devProp.name);
    cudaSetDevice(device);
    cudaCheckError("Set selected device");

    // file names
    srcFilename = (char *)malloc(strlen(argv[0]));
    strcpy(srcFilename, argv[0]);
    if (argc == 1) { // only one filename supplyed
        outFilename = (char *)malloc(strlen(srcFilename)+4);
        strcpy(outFilename, srcFilename);
        strcpy(outFilename+strlen(srcFilename), ".dwt");
    } else {
        outFilename = (char *)malloc(strlen(argv[1]));
        strcpy(outFilename, argv[1]);
    }

    //Input review
    printf("Source file:\t\t%s\n", srcFilename);
    printf(" Dimensions:\t\t%dx%d\n", pixWidth, pixHeight);
    printf(" Components count:\t%d\n", compCount);
    printf(" Bit depth:\t\t%d\n", bitDepth);
    printf(" DWT levels:\t\t%d\n", dwtLvls);
    
    //data sizes
    int inputSize = pixWidth*pixHeight*compCount; //<amount of data (in bytes) to proccess
    int componentSize = pixHeight*pixWidth*sizeof(float);

    //load img source image
    unsigned char *srcImg = NULL;
    cudaMallocHost((void **)&srcImg, inputSize);
    cudaCheckError("Alloc host memory");
    if (getImg(srcFilename, srcImg, inputSize) == -1) 
        return -1;

    if (compCount == 3) {
        /* Load components */
        float *c_r_comp, *c_g_comp, *c_b_comp; // color components 
        cudaMalloc((void**)&c_r_comp, componentSize); //< R, aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_r_comp, 0, componentSize);
        cudaCheckError("Memset device memory");

        cudaMalloc((void**)&c_g_comp, componentSize); //< G, aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_g_comp, 0, componentSize);
        cudaCheckError("Memset device memory");

        cudaMalloc((void**)&c_b_comp, componentSize); //< B, aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_b_comp, 0, componentSize);
        cudaCheckError("Memset device memory");

        rgbToComponents(c_r_comp, c_g_comp, c_b_comp, srcImg, pixWidth, pixHeight);

        /* Forward DWT 9/7 */
        float *c_r_wave, *c_g_wave, *c_b_wave;
        cudaMalloc((void**)&c_r_wave, componentSize); //< R, aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_r_wave, 0, componentSize);
        cudaCheckError("Memset device memory");

        cudaMalloc((void**)&c_g_wave, componentSize); //< G, aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_g_wave, 0, componentSize);
        cudaCheckError("Memset device memory");

        cudaMalloc((void**)&c_b_wave, componentSize); //< B, aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_b_wave, 0, componentSize);
        cudaCheckError("Memset device memory");

        /* Compute DWT */
        nStage2dRDWT97(c_r_comp, c_r_wave, pixWidth, pixHeight, mantissa, exponent, dwtLvls);
        nStage2dRDWT97(c_g_comp, c_g_wave, pixWidth, pixHeight, mantissa, exponent, dwtLvls);
        nStage2dRDWT97(c_b_comp, c_b_wave, pixWidth, pixHeight, mantissa, exponent, dwtLvls);
        /* Store DWT to file */
        writeNStage2DDWT(c_r_comp, pixWidth, pixHeight, dwtLvls, srcFilename, ".r.dwt");
        writeNStage2DDWT(c_g_comp, pixWidth, pixHeight, dwtLvls, srcFilename, ".g.dwt");
        writeNStage2DDWT(c_b_comp, pixWidth, pixHeight, dwtLvls, srcFilename, ".b.dwt");

        /* Clean up */
        cudaFree(c_r_wave);
        cudaCheckError("Cuda free device");
        cudaFree(c_g_wave);
        cudaCheckError("Cuda free device");
        cudaFree(c_b_wave);
        cudaCheckError("Cuda free device");
        cudaFree(c_r_comp);
        cudaCheckError("Cuda free device");
        cudaFree(c_g_comp);
        cudaCheckError("Cuda free device");
        cudaFree(c_b_comp);
        cudaCheckError("Cuda free device");
    } else if (compCount == 1) {
        //Load component
        float *c_component; // color component 
        cudaMalloc((void**)&c_component, componentSize); //< R, aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_component, 0, componentSize);
        cudaCheckError("Memset device memory");

        bwToComponent(c_component, srcImg, pixWidth, pixHeight);

        /* Forward DWT 9/7 */
        float *c_wave;
        cudaMalloc((void**)&c_wave, componentSize); //< aligned component size
        cudaCheckError("Alloc device memory");
        cudaMemset(c_wave, 0, componentSize);
        cudaCheckError("Memset device memory");

        /* Compute DWT */
        nStage2dRDWT97(c_component, c_wave, pixWidth, pixHeight, mantissa, exponent, dwtLvls);
        /* Store DWT to file */
        writeNStage2DDWT(c_component, pixWidth, pixHeight, dwtLvls, srcFilename, ".dwt");

        /* Clean up */
        cudaFree(c_wave);
        cudaCheckError("Cuda free device");
        cudaFree(c_component);
        cudaCheckError("Cuda free device");
    }

    //writeComponent(r_cuda, pixWidth, pixHeight, srcFilename, ".g");
    //writeComponent(g_wave_cuda, 512000, ".g");
    //writeComponent(g_cuda, componentSize, ".g");
    //writeComponent(b_wave_cuda, componentSize, ".b");
    cudaFreeHost(srcImg);
    cudaCheckError("Cuda free host");

    return 0;
}
