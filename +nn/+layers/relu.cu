__global__ void backward( float * dzdx, const float dzdy, const float * bottom, const int len) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) return;
    
    dzdx[idx] = bottom[idx] > 0.0f ? dzdy:0.0f;
}
