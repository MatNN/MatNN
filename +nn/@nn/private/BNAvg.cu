__global__ void BNAvg(const float lr, float *w, const float *dzdw, 
                      const float ts, const float len) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;
    w[index] = (1-lr)*w[index]+lr*dzdw[index]/ts;
}