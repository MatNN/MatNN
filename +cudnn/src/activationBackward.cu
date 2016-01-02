#include "utils.h"
/***
 PRHS[0] = ActivationMode
 PRHS[1] = input data 
 PRHS[2] = output data (Format in Matlab = HWCN)
 PRHS[3] = output diff (Format in Matlab = HWCN)
 PLHS[0] = input diff (Format in Matlab = HWCN)
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
    if (nrhs != 4) mexErrMsgIdAndTxt("MATLAB:MatNN", "Accepts 4 inputs."); // mode, input, output, output_diff
    if (nlhs != 1) mexErrMsgIdAndTxt("MATLAB:MatNN", "Accepts 1 output.");

    // Check prhs[0] == int
    cudnnActivationMode_t mode = (cudnnActivationMode_t)mxGetScalar(prhs[0]);

    // Get inputs / outputs data
    mxGPUArray const *input = mxGPUCreateFromMxArray(prhs[1]);
    gpuData inputInfo = createInfoFromMxGPUArray(input);
    float *input_data = (float *)mxGPUGetDataReadOnly(input);

    mxGPUArray const *out = mxGPUCreateFromMxArray(prhs[2]);
    gpuData out_info = createInfoFromMxGPUArray(out);
    float *out_data = (float *)mxGPUGetDataReadOnly(out);

    mxGPUArray const *out_diff = mxGPUCreateFromMxArray(prhs[3]);
    gpuData out_diff_info = createInfoFromMxGPUArray(out_diff);
    float *out_diff_data = (float *)mxGPUGetDataReadOnly(out_diff);

    mxGPUArray *in_diff = mxGPUCreateGPUArray(inputInfo.ndims,
                                inputInfo.dims,
                                inputInfo.type,
                                mxREAL,
                                MX_GPU_DO_NOT_INITIALIZE); // MX_GPU_INITIALIZE_VALUES
    float *in_diff_data = (float *)mxGPUGetData(in_diff);

    // init
    float alpha = 1.0;
    float beta  = 0.0;
    
    cudnnTensorDescriptor_t srcTensorDesc, srcDiffTensorDesc, dstTensorDesc, dstDiffTensorDesc;
    if (!initialized){
      CUDNN_CHECK( cudnnCreate(&cudnnHandle) );
      mexAtExit(destroyHandle);
      initialized = true;
    }
    CUDNN_CHECK( cudnnCreateTensorDescriptor(&srcTensorDesc) );
    CUDNN_CHECK( cudnnCreateTensorDescriptor(&srcDiffTensorDesc) );
    CUDNN_CHECK( cudnnCreateTensorDescriptor(&dstTensorDesc) );
    CUDNN_CHECK( cudnnCreateTensorDescriptor(&dstDiffTensorDesc) );
    

    // Set sizes
    CUDNN_CHECK( cudnnSetTensor4dDescriptorEx(srcTensorDesc,
                                              CUDNN_DATA_FLOAT,
                                              inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w,
                                              inputInfo.nStride, inputInfo.cStride, inputInfo.hStride, inputInfo.wStride) );
    CUDNN_CHECK( cudnnSetTensor4dDescriptorEx(srcDiffTensorDesc,
                                              CUDNN_DATA_FLOAT,
                                              inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w,
                                              inputInfo.nStride, inputInfo.cStride, inputInfo.hStride, inputInfo.wStride) );
    CUDNN_CHECK( cudnnSetTensor4dDescriptorEx(dstDiffTensorDesc,
                                              CUDNN_DATA_FLOAT,
                                              inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w,
                                              inputInfo.nStride, inputInfo.cStride, inputInfo.hStride, inputInfo.wStride) );
    CUDNN_CHECK( cudnnSetTensor4dDescriptorEx(dstTensorDesc,
                                              CUDNN_DATA_FLOAT,
                                              inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w,
                                              inputInfo.nStride, inputInfo.cStride, inputInfo.hStride, inputInfo.wStride) );
    

    // compute
    CUDNN_CHECK( cudnnActivationBackward(cudnnHandle,
                                        mode,
                                        &alpha,
                                        dstTensorDesc,
                                        out_data,
                                        dstDiffTensorDesc,
                                        out_diff_data,
                                        srcTensorDesc,
                                        input_data,
                                        &beta,
                                        srcDiffTensorDesc,
                                        in_diff_data) );

    // Set outputs
    //cudnnDestroy(cudnnHandle);
    CUDNN_CHECK( cudnnDestroyTensorDescriptor(srcTensorDesc) );
    CUDNN_CHECK( cudnnDestroyTensorDescriptor(srcDiffTensorDesc) );
    CUDNN_CHECK( cudnnDestroyTensorDescriptor(dstTensorDesc) );
    CUDNN_CHECK( cudnnDestroyTensorDescriptor(dstDiffTensorDesc) );
    mxGPUDestroyGPUArray(input);
    mxGPUDestroyGPUArray(out_diff);
    mxGPUDestroyGPUArray(out);
    plhs[0] = mxGPUCreateMxArrayOnGPU(in_diff);
    mxGPUDestroyGPUArray(in_diff);
}