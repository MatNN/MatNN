__global__ void forward( float * weight, float * momentum, const float lr, const float wDecay, const float optMomentum, const float * dzdw, const int len) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) return;

    momentum[idx] = optMomentum * momentum[idx] - lr*(wDecay*weight[idx] + dzdw[idx]);
    weight[idx] += momentum[idx];
}
