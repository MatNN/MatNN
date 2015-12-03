__global__ void BilinearInterpolationForward(const float* bottom_data,
    const int* bs, const float* pos_data, float* top_data, const int* ts) {
    // bs = bottom_data size, ps = pos_data size, ts = top_data size
    // input position = -1~1
    // pos_data[:,:,1,:] = x, pos_data[:,:,2,:] = y

    // top_data size = (ps[0], ps[1], bs[2], ps[3])
    // note: bs[3] === ps[3]

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int len = ts[0]*ts[1]*ts[2]*ts[3];
    if (index >= len) return;

    // get current index, [h,w,c,n]
    int h = index % ts[0];
    int w = (index / ts[0]) % ts[1];
    int c = (index / ts[0] / ts[1]) % ts[2];
    int n = index / ts[0] / ts[1] / ts[2];

    int xp = n*2*ts[1]*ts[0] + 0*ts[1]*ts[0] + w*ts[0] + h;
    int yp = n*2*ts[1]*ts[0] + 1*ts[1]*ts[0] + w*ts[0] + h;

    float w_new = (pos_data[xp]/2.0+0.5)*(float)(bs[1]-1);
    float h_new = (pos_data[yp]/2.0+0.5)*(float)(bs[0]-1);

    // calc neighbor pixel index, if > size or < size, do
    float v = 0.0;
    for (int x = floor(w_new); x <= ceil(w_new); x++) {
      for (int y = floor(h_new); y <= ceil(h_new); y++) {
        if (x < 0 || x>= bs[1] || y < 0 || y >= bs[0]){
          v = 0.0;
        }else{
          v = bottom_data[n*bs[2]*bs[1]*bs[0] + c*bs[1]*bs[0] + x*bs[0] + y];
        }
        top_data[index] += v * (1-abs(w_new - (float)x)) * (1-abs(h_new - (float)y));
      }
    }
}

__global__ void BilinearInterpolationBackward(const float* bottom_data,
    const int* bs, const float* pos_data, const float* top_data, const int* ts, const float* top_diff, float* bottom_diff, float* pos_diff) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int len = ts[0]*ts[1]*ts[2]*ts[3];
    if (index >= len) return;

    // get current index, [h,w,c,n]
    int h = index % ts[0];
    int w = (index / ts[0]) % ts[1];
    int c = (index / ts[0] / ts[1]) % ts[2];
    int n = index / ts[0] / ts[1] / ts[2];

    int xp = n*2*ts[1]*ts[0] + 0*ts[1]*ts[0] + w*ts[0] + h;
    int yp = n*2*ts[1]*ts[0] + 1*ts[1]*ts[0] + w*ts[0] + h;

    float w_new = (pos_data[xp]/2.0+0.5)*(float)(bs[1]-1);
    float h_new = (pos_data[yp]/2.0+0.5)*(float)(bs[0]-1);

    float u = 0.0;
    for (int x = floor(w_new); x <= ceil(w_new); x++) {
      for (int y = floor(h_new); y <= ceil(h_new); y++) {
        if (x >= 0 && x < bs[1] && y >= 0 && y < bs[0]){

          atomicAdd(bottom_diff + n*bs[2]*bs[1]*bs[0] + c*bs[1]*bs[0] + x*bs[0] + y, top_diff[index] * (1-abs(w_new - (float)x)) * (1-abs(h_new - (float)y)) );
          u = bottom_data[n*bs[2]*bs[1]*bs[0] + c*bs[1]*bs[0] + x*bs[0] + y];

          atomicAdd(pos_diff + xp, top_diff[index] *u* (1-abs(h_new - (float)y)) * ((float)x >= w_new ? 1.0:-1.0 )  );
          atomicAdd(pos_diff + yp, top_diff[index] *u* (1-abs(w_new - (float)x)) * ((float)y >= h_new ? 1.0:-1.0 )  );
        }
      }
    }
}