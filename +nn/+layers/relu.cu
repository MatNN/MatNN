__global__ void forward(float * bottom, const int len) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) return;
    
    bottom[idx] = max(bottom[idx], 0.0f);
}
__global__ void backward(float * bottom, const float * dzdy, const int len) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) return;
    
    bottom[idx] = bottom[idx] > 0.0f ? dzdy[idx]:0.0f;
}
