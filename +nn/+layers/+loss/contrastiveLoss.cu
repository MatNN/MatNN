__global__ void forward( float * loss, const float * y, const float * d, const float margin, int len) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) return;
    float kk = (y[idx]*d[idx]+(1-y[idx])*max(margin-d[idx],0.0f))*0.5;
    atomicAdd(loss, kk);
}

__global__ void backward( float * dzdx1, float * dzdx2, const float * dzdy , const float * y, const float * d_, const float * d, const float margin, const int * d__dim, const int * y_dim) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int d__len = d__dim[0]*d__dim[1]*d__dim[2]*d__dim[3];
    if (idx >= d__len) return;

    int y_idx = idx/(d__len/(y_dim[0]*y_dim[1]*y_dim[2]*y_dim[3]));
    dzdx1[idx] = dzdy* ( d_[idx]*y[y_idx] - (1.0f-y[y_idx])*( (margin-d[y_idx]) > 0.0f ? d_[idx] : 0.0f) );
    dzdx2[idx] = -dzdx1[idx];
}
