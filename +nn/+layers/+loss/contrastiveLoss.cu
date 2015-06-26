__global__ void forward( float * loss, const float * y, const float * d, const float margin, int len) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) return;
    float kk = ( y[idx]*d[idx] + (1-y[idx])*max(margin-d[idx], 0.0f) )*0.5;
    atomicAdd(loss, kk);
}

__global__ void backward(float * b1, float * b2, const float dzdy , const float * y, const float * d, const float margin, const int len, const int factor) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) return;
    float d_ = b1[idx] - b2[idx];
    int y_idx = idx/factor;
    b1[idx] = dzdy * ( d_*y[y_idx] - (1.0f-y[y_idx])*( (margin-d[y_idx]) > 0.0f ? d_ : 0.0f) );
    b2[idx] = -b1[idx];
}
