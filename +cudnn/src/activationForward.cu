#include "utils.h"
/***
 PRHS[0] = ActivationMode
 PRHS[1] = input data (Format in Matlab = HWCN)
 PLHS[0] = output data (Format in Matlab = HWCN)
 ------------
 int ActivationMode, 0 = CUDNN_ACTIVATION_SIGMOID
                     1 = CUDNN_ACTIVATION_RELU
                     2 = CUDNN_ACTIVATION_TANH
                     3 = CUDNN_ACTIVATION_CLIPPED_RELU
 */
cudnnHandle_t cudnnHandle;
bool initialized = false;
void destroyHandle(){
  CUDNN_CHECK( cudnnDestroy(cudnnHandle) );
}
void mexFunction( int nlhs,       mxArray *plhs[], 
                  int nrhs, const mxArray *prhs[]  )
{
    // Check nargin / nargout
    if (nrhs != 2) mexErrMsgIdAndTxt("MATLAB:MatNN", "Accepts 2 inputs.");
    if (nlhs != 1) mexErrMsgIdAndTxt("MATLAB:MatNN", "Accepts 1 output.");

    // Check prhs[0] == int
    cudnnActivationMode_t mode = (cudnnActivationMode_t)mxGetScalar(prhs[0]);

    // Get inputs / outputs data
    mxGPUArray const *input1 = mxGPUCreateFromMxArray(prhs[1]);
    gpuData input1Info = createInfoFromMxGPUArray(input1);
    float *input1_data = (float *)mxGPUGetDataReadOnly(input1);

    
    mxGPUArray *output1 = mxGPUCreateGPUArray(input1Info.ndims,
                                input1Info.dims,
                                input1Info.type,
                                mxREAL,
                                MX_GPU_DO_NOT_INITIALIZE); // MX_GPU_INITIALIZE_VALUES
    float *output1_data = (float *)mxGPUGetData(output1);

    // init
    float alpha = 1.0;
    float beta  = 0.0;
    
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    if (!initialized){
      CUDNN_CHECK( cudnnCreate(&cudnnHandle) );
      mexAtExit(destroyHandle);
      initialized = true;
    }
    CUDNN_CHECK( cudnnCreateTensorDescriptor(&srcTensorDesc) );
    CUDNN_CHECK( cudnnCreateTensorDescriptor(&dstTensorDesc) );

    // Set sizes
    CUDNN_CHECK( cudnnSetTensor4dDescriptorEx(srcTensorDesc,
                                              CUDNN_DATA_FLOAT,
                                              input1Info.n, input1Info.c, input1Info.h, input1Info.w,
                                              input1Info.nStride, input1Info.cStride, input1Info.hStride, input1Info.wStride) );
    CUDNN_CHECK( cudnnSetTensor4dDescriptorEx(dstTensorDesc,
                                              CUDNN_DATA_FLOAT,
                                              input1Info.n, input1Info.c, input1Info.h, input1Info.w,
                                              input1Info.nStride, input1Info.cStride, input1Info.hStride, input1Info.wStride) );

    // compute
    CUDNN_CHECK( cudnnActivationForward(cudnnHandle,
                                        mode,
                                        &alpha,
                                        srcTensorDesc,
                                        input1_data,
                                        &beta,
                                        dstTensorDesc,
                                        output1_data) );

    // Set outputs
    //cudnnDestroy(cudnnHandle);
    CUDNN_CHECK( cudnnDestroyTensorDescriptor(srcTensorDesc) );
    CUDNN_CHECK( cudnnDestroyTensorDescriptor(dstTensorDesc) );
    mxGPUDestroyGPUArray(input1);
    plhs[0] = mxGPUCreateMxArrayOnGPU(output1);
    mxGPUDestroyGPUArray(output1);
}