#include <sys/types.h>
#include <sys/mman.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <cuda.h>
#include <cudnn.h>

#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cuAssert(cudaError_t code, const char *file, int line)
{
    char s[4096] = {0};
    if (code != cudaSuccess) 
    {
        sprintf(s, "GPUError: %s %s %d\n", cudaGetErrorString(code), file, line);
        mexErrMsgIdAndTxt("MATLAB:MatNN", s);
    }
}

#define CUDNN_CHECK(ans) { cudnnAssert((ans), __FILE__, __LINE__); }
inline void cudnnAssert(cudnnStatus_t code, const char *file, int line)
{
    char s[4096] = {0};
    if (code != CUDNN_STATUS_SUCCESS) 
    {
        sprintf(s, "GPUError: %s %s %d\n", cudnnGetErrorString(code), file, line);
        mexErrMsgIdAndTxt("MATLAB:MatNN", s);
    }
}

typedef struct gpuData
{
    int ndims;
    const mwSize *dims;
    mwSize n;
    mwSize c;
    mwSize h;
    mwSize w;
    int nStride;
    int cStride;
    int hStride;
    int wStride;
    int len;
    mxClassID type;
} gpuData;

gpuData createInfoFromMxGPUArray(const mxGPUArray* mxGPUArrayVar) {
    gpuData gData;
    gData.ndims = mxGPUGetNumberOfDimensions(mxGPUArrayVar);
    gData.dims  = mxGPUGetDimensions(mxGPUArrayVar);
    if (gData.ndims == 4){
        gData.n = gData.dims[3];
        gData.c = gData.dims[2];
        gData.h = gData.dims[0];
        gData.w = gData.dims[1];
    }else if (gData.ndims == 3){
        gData.n = 1;
        gData.c = gData.dims[2];
        gData.h = gData.dims[0];
        gData.w = gData.dims[1];
    }else if (gData.ndims == 2){
        gData.n = 1;
        gData.c = 1;
        gData.h = gData.dims[0];
        gData.w = gData.dims[1];
    }else if (gData.ndims == 1){
        gData.n = 1;
        gData.c = 1;
        gData.h = gData.dims[0];
        gData.w = 1;
    }else{
        mexErrMsgIdAndTxt("MATLAB:MatNN", "data dimensions must >= 1.");
    }
    
    gData.len   = mxGPUGetNumberOfElements(mxGPUArrayVar);
    gData.type  = mxGPUGetClassID(mxGPUArrayVar);
    if (gData.type!=mxSINGLE_CLASS) mexErrMsgIdAndTxt("MATLAB:MatNN", "data must be single type.");

    gData.nStride = gData.h*gData.w*gData.c;
    gData.cStride = gData.h*gData.w;
    gData.hStride = 1;
    gData.wStride = gData.h;

    return gData;
}