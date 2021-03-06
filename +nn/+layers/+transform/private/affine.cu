/***
  Implementation of Spatial Transformer Networks[1]

  Under Simplified BSD License
  by Che-Wei Lin

  [1] Max Jaderberg et al. Spatial Transformer Networks. NIPS 2015
***/

__global__ void AffineForward(const float* bottom_data,
    const int* bs, const float* affine, const int len, float* top_data) {
    // bs = bottomSize

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;

    // get current index, [h,w,c,n]
    int h = index % bs[0];
    int w = (index / bs[0]) % bs[1];
    int c = (index / bs[0] / bs[1]) % bs[2];
    int n = index / bs[0] / bs[1] / bs[2];

    // get current affine start index
    const float* a = affine + n*6;

    // calc bottom index
    //                    [a0 a1 0]
    // [x y 1] = [u v 1] *[a2 a3 0] 
    //                    [a4 a5 1]
    float nw = 2.0*((float)w/(float)bs[1]-0.5); //-1~1
    float nh = 2.0*((float)h/(float)bs[0]-0.5); //-1~1

    float w_new = ((a[0]*nw + a[2]*nh + a[4])/2.0+0.5)*(float)bs[1];
    float h_new = ((a[1]*nw + a[3]*nh + a[5])/2.0+0.5)*(float)bs[0];

    // calc neighbor pixel index, if > size or < size, do
    float v = 0.0;
    for (int x = floor(w_new); x <= ceil(w_new); x++) {
      for (int y = floor(h_new); y <= ceil(h_new); y++) {
        if (x < 0 || x>= bs[1] || y < 0 || y >= bs[0]){
          v = 0.0;
        }else{
          v = bottom_data[n*len/bs[3] + c*bs[1]*bs[0] + x*bs[0] + y];
        }
        top_data[index] += v * (1-abs(w_new - (float)x)) * (1-abs(h_new - (float)y));
      }
    }

}

__global__ void AffineBackward(const float* bottom_data,
    const int* bs, const float* affine, const int len, const float* top_data, const float* top_diff, float* bottom_diff1, float* bottom_diff2) {
    // bs = bottomSize

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;

    // get current index, [h,w,c,n]
    int h = index % bs[0];
    int w = (index / bs[0]) % bs[1];
    int c = (index / bs[0] / bs[1]) % bs[2];
    int n = index / bs[0] / bs[1] / bs[2];

    // get current affine start index
    const float* a = affine + n*6;

    // calc bottom index
    //                    [a0 a1 0]
    // [x y 1] = [u v 1] *[a2 a3 0] 
    //                    [a4 a5 1]
    float nw = 2.0*((float)w/(float)bs[1]-0.5); //-1~1
    float nh = 2.0*((float)h/(float)bs[0]-0.5); //-1~1

    float w_new = ((a[0]*nw + a[2]*nh + a[4])/2.0+0.5)*(float)bs[1];
    float h_new = ((a[1]*nw + a[3]*nh + a[5])/2.0+0.5)*(float)bs[0];

    float u = 0.0;
    float dx = 0.0;
    float dy = 0.0;
    for (int x = max(floor(w_new),0.0); x <= min(ceil(w_new),(float)bs[1]); x++) {
      for (int y = max(floor(h_new),0.0); y <= min(ceil(h_new),(float)bs[0]); y++) {
          u = bottom_data[n*bs[2]*bs[1]*bs[0] + c*bs[1]*bs[0] + x*bs[0] + y];
          atomicAdd(bottom_diff1 + (n*bs[2]*bs[1]*bs[0] + c*bs[1]*bs[0] + x*bs[0] + y),  top_diff[index] * (1-abs(w_new - (float)x)) * (1-abs(h_new - (float)y))  );
          dx += u * (1-abs(h_new - (float)y)) * ((float)x >= w_new ? 1.0:-1.0 );
          dy += u * (1-abs(w_new - (float)x)) * ((float)y >= h_new ? 1.0:-1.0 );
      }
    }
    
    atomicAdd((bottom_diff2+n*6)+0, nw *dx*top_diff[index]);
    atomicAdd((bottom_diff2+n*6)+2, nh *dx*top_diff[index]);
    atomicAdd((bottom_diff2+n*6)+4, 1.0*dx*top_diff[index]);
    atomicAdd((bottom_diff2+n*6)+1, nw *dy*top_diff[index]);
    atomicAdd((bottom_diff2+n*6)+3, nh *dy*top_diff[index]);
    atomicAdd((bottom_diff2+n*6)+5, 1.0*dy*top_diff[index]);
}