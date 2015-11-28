//parameters: shiftX,Y, scaleX,Y,  shearX,Y = 6dimensions
//            multiply angle = 7
__global__ void AffineForward(const float* bottom_data,
    const int* bottomSize, const float* affine, const int len, float* top_data) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;

    // get current index, [h,w,c,n]
    int h = index % bottomSize[0];
    int w = (index / bottomSize[0]) % bottomSize[1];
    int c = (index / bottomSize[0] / bottomSize[1]) % bottomSize[2];
    int n = index / bottomSize[0] / bottomSize[1] / bottomSize[2];

    // get current affine start index
    const float* a = affine + n*7;

    // calc bottom index
    //                    [a0 a1 0] [cos6  sin6 0]             [0cos6-1sin6 0sin6+1cos6 0]
    // [x y 1] = [u v 1] *[a2 a3 0]*[-sin6 cos6 0] = [u v 1] * [2cos6-3sin6 2sin6+3cos6 0]
    //                    [a4 a5 1] [0     0    1]             [4cos6-5sin6 4sin6+5cos6 1]
    float nw = 2.0*((float)w/(float)bottomSize[1]-0.5); //-1~1
    float nh = 2.0*((float)h/(float)bottomSize[0]-0.5); //-1~1

    float w_new = nw*(a[0]*cos(a[6])-a[1]*sin(a[6])) + nh*(a[2]*cos(a[6])-a[3]*sin(a[6])) + (a[4]*cos(a[6])-a[5]*sin(a[6]));
    float h_new = nw*(a[0]*sin(a[6])+a[1]*cos(a[6])) + nh*(a[2]*sin(a[6])+a[3]*cos(a[6])) + (a[4]*sin(a[6])+a[5]*cos(a[6]));
    w_new = (w_new/2.0+0.5)*(float)bottomSize[1];
    h_new = (h_new/2.0+0.5)*(float)bottomSize[0];

    // calc neighbor pixel index, if > size or < size, do
    float v = 0.0;
    for (int x = floor(w_new); x<=ceil(w_new); x++) {
      for (int y = floor(h_new); y<=ceil(h_new); y++) {
        if (x<0 || x>= bottomSize[1] || y < 0 || y >= bottomSize[0]){
          v = 0.0;
        }else{
          v = bottom_data[n*bottomSize[2]*bottomSize[1]*bottomSize[0] + c*bottomSize[1]*bottomSize[0] + x*bottomSize[0] + y];
        }
        top_data[index] += v * (1-abs(w_new - (float)x)) * (1-abs(h_new - (float)y));
      }
    }

}

__global__ void AffineBackward(const float* bottom_data,
    const int* bottomSize, const float* affine, const int len, const float* top_data, const float* top_diff, float* bottom_diff1, float* bottom_diff2) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) return;

    // get current index, [h,w,c,n]
    int h = index % bottomSize[0];
    int w = (index / bottomSize[0]) % bottomSize[1];
    int c = (index / bottomSize[0] / bottomSize[1]) % bottomSize[2];
    int n = index / bottomSize[0] / bottomSize[1] / bottomSize[2];

    // get current affine start index
    const float* a = affine + n*7;

    // calc bottom index
    //                    [a0 a1 0]
    // [x y 1] = [u v 1] *[a2 a3 0] 
    //                    [a4 a5 1]
    float nw = 2.0*((float)w/(float)bottomSize[1]-0.5); //-1~1
    float nh = 2.0*((float)h/(float)bottomSize[0]-0.5); //-1~1

    float w_new = nw*(a[0]*cos(a[6])-a[1]*sin(a[6])) + nh*(a[2]*cos(a[6])-a[3]*sin(a[6])) + (a[4]*cos(a[6])-a[5]*sin(a[6]));
    float h_new = nw*(a[0]*sin(a[6])+a[1]*cos(a[6])) + nh*(a[2]*sin(a[6])+a[3]*cos(a[6])) + (a[4]*sin(a[6])+a[5]*cos(a[6]));
    w_new = (w_new/2.0+0.5)*(float)bottomSize[1];
    h_new = (h_new/2.0+0.5)*(float)bottomSize[0];

    float v = 0.0;
    float dx = 0.0;
    float dy = 0.0;
    for (int x = floor(w_new); x<=ceil(w_new); x++) {
      for (int y = floor(h_new); y<=ceil(h_new); y++) {
        if (x<0 || x>= bottomSize[1] || y < 0 || y >= bottomSize[0]){
          v = 0.0;
        }else{
          v = bottom_data[n*bottomSize[2]*bottomSize[1]*bottomSize[0] + c*bottomSize[1]*bottomSize[0] + x*bottomSize[0] + y];
        }
        bottom_diff1[n*bottomSize[2]*bottomSize[1]*bottomSize[0] + c*bottomSize[1]*bottomSize[0] + x*bottomSize[0] + y] += top_diff[index] * (1-abs(w_new - (float)x)) * (1-abs(h_new - (float)y));
        dx += v * (1-abs(h_new - (float)y)) * ((float)x > w_new ? 1.0:-1.0 );
        dy += v * (1-abs(w_new - (float)x)) * ((float)y > h_new ? 1.0:-1.0 );
      }
    }
    
    
    atomicAdd((bottom_diff2+n*7)+0, nw *(cos(a[6])+sin(a[6])) *dx*top_diff[index]);
    atomicAdd((bottom_diff2+n*7)+2, nh *(cos(a[6])+sin(a[6])) *dx*top_diff[index]);
    atomicAdd((bottom_diff2+n*7)+4, 1.0*(cos(a[6])+sin(a[6])) *dx*top_diff[index]);
    atomicAdd((bottom_diff2+n*7)+1, nw *(cos(a[6])-sin(a[6])) *dy*top_diff[index]);
    atomicAdd((bottom_diff2+n*7)+3, nh *(cos(a[6])-sin(a[6])) *dy*top_diff[index]);
    atomicAdd((bottom_diff2+n*7)+5, 1.0*(cos(a[6])-sin(a[6])) *dy*top_diff[index]);
    float ba6 = (nw*(-a[0]*sin(a[6])-a[1]*cos(a[6])) + nh*(-a[2]*sin(a[6])-a[3]*cos(a[6])) + (-a[4]*sin(a[6])-a[5]*cos(a[6])))*dx*top_diff[index];
    ba6 += (nw*(a[0]*cos(a[6])-a[1]*sin(a[6])) + nh*(a[2]*cos(a[6])-a[3]*sin(a[6])) + (a[4]*cos(a[6])-a[5]*sin(a[6])))*dy*top_diff[index];
    atomicAdd((bottom_diff2+n*7)+6, ba6);
}