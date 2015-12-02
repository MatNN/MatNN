__global__ void SoftMaxLossForward(const float* bottom_data, const int* bs, const float* label, const float* label_weight, 
    const int* ls, const float threshold, const int label_start, bool hasLabel_weight, float* loss) {
    // bs = bottomSize

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= bs[0]*bs[1]*1*bs[3]) return;

    // get current index, [h,w,'i',n]
    int h = index % bs[0];
    int w = (index / bs[0]) % bs[1];
    int n = index / bs[0] / bs[1];

    int ind_in_btm = 0;
    int label_ind = 0;
    if (ls[0]*ls[1]*ls[2]*ls[3] == bs[4]) {
        if (label[n] < label_start) return;
        label_ind = n;
    }else{ //btm size 0,2,3 == label size 0,2,3, and ls[2] = 1
        label_ind = n*bs[1]*bs[0] + w*bs[0] + h;
    }
    ind_in_btm = n*bs[2]*bs[1]*bs[0] + ((int)label[label_ind]-label_start)*bs[1]*bs[0] + w*bs[0] + h;

    float maxValueInThirdDim = bottom_data[n*bs[2]*bs[1]*bs[0] + w*bs[0] + h];
    for(int i=0; i<bs[2]; i++){
        maxValueInThirdDim = max(maxValueInThirdDim, bottom_data[n*bs[2]*bs[1]*bs[0] + i*bs[1]*bs[0] + w*bs[0] + h]);
    }

    //do the work

    float y = 0;
    for(int i=0; i<bs[2]; i++){
        y += exp(bottom_data[n*bs[2]*bs[1]*bs[0] + i*bs[1]*bs[0] + w*bs[0] + h] - maxValueInThirdDim); // sum third dim
    }

    //get softmax
    y = exp(bottom_data[ind_in_btm]-maxValueInThirdDim)/y;
    if (hasLabel_weight){
        atomicAdd(loss, -label_weight[label_ind]*log(max(y, threshold)));
    }else{
        atomicAdd(loss, -log(max(y, threshold)));
    }
    //the output must divide bs[0]*bs[1]*bs[3] 

}

__global__ void SoftMaxLossBackward(const float* bottom_data, const int* bs, const float* label, const float* label_weight, 
    const int* ls, const float top_diff, const float threshold, const int label_start, bool hasLabel_weight, float* bottom_diff) {
    // bs = bottomSize

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int bs_len = bs[0]*bs[1]*bs[2]*bs[3];
    if (index >= bs_len) return;

    // get current index, [h,w,c,n]
    int h = index % bs[0];
    int w = (index / bs[0]) % bs[1];
    int c = (index / bs[0] / bs[1]) % bs[2];
    int n = index / bs[0] / bs[1] / bs[2];

    int ind_in_btm = 0;
    int label_ind = 0;
    if (ls[0]*ls[1]*ls[2]*ls[3] == bs[4]) {
        if (label[n] < label_start) {
            label_ind = -1;
        }else{
            label_ind = n;
        };
    }else{ //btm size 0,2,3 == label size 0,2,3, and ls[2] = 1
        label_ind = n*bs[1]*bs[0] + w*bs[0] + h;
    }
    if (label_ind == -1)
        ind_in_btm = -1;
    else
        ind_in_btm = n*bs[2]*bs[1]*bs[0] + ((int)label[label_ind]-label_start)*bs[1]*bs[0] + w*bs[0] + h;

    float maxValueInThirdDim = bottom_data[index]; // get current value first
    for(int i=0; i<bs[2]; i++){
        maxValueInThirdDim = max(maxValueInThirdDim, bottom_data[n*bs[2]*bs[1]*bs[0] + i*bs[1]*bs[0] + w*bs[0] + h]);
    }

    //do the work

    float y = 0;
    for(int i=0; i<bs[2]; i++){
        y += exp(bottom_data[n*bs[2]*bs[1]*bs[0] + i*bs[1]*bs[0] + w*bs[0] + h] - maxValueInThirdDim); // sum third dim
    }

    //get derivative
    y = exp(bottom_data[index]-maxValueInThirdDim)/max(y, threshold);
    if (ind_in_btm == index) y = y-1.0;

    if (hasLabel_weight){
        if (label_ind == -1)
            bottom_diff[index] = y*top_diff/bs[0]/bs[1]/bs[3];
        else
            bottom_diff[index] = label_weight[label_ind]*y*top_diff/bs[0]/bs[1]/bs[3];
    }else{
        bottom_diff[index] = y*top_diff/bs[0]/bs[1]/bs[3];
    }

}