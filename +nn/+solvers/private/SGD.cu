__global__ void SGD(const float mo,       float *MO, 
                    const float lr, const float LR,
                    const float wd, const float WD,
                          float *w, const float *dzdw, 
                    const float ts, const float len) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;

    MO[index] = mo*MO[index] - (lr*LR)*((wd*WD)*w[index] + dzdw[index]/ts);
    w[index] += MO[index];

}