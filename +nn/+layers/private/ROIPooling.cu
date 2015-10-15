// LICENSE

// --------------------------------------------------------
// Fast R-CNN
// Copyright (c) Microsoft. All rights reserved.
// Written by Ross Girshick, 2015.
// Licensed under the BSD 2-clause "Simplified" license.
// See LICENSE in the Fast R-CNN project root for license
// information.
// --------------------------------------------------------

// --------------------------------------------------------
// Roi pooling layer (only two CUDA kernels)
// Authored by Ross Girshick
// These Kernels were modified to run on Matlab
// --------------------------------------------------------


__global__ void ROIPoolForward(const int nthreads, const float* bottom_data,
    const float spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const float* bottom_rois, float* top_data, int* argmax_data) {
    
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x){
    // (ph, pw, c, n) is an element in the pooled output
    int ph = index % pooled_height;
    int pw = (index / pooled_height) % pooled_weight;
    int c = (index / pooled_height / pooled_width) % channels;
    int n = index / pooled_height / pooled_width / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1] * spatial_scale);
    int roi_start_h = round(bottom_rois[2] * spatial_scale);
    int roi_end_w = round(bottom_rois[3] * spatial_scale);
    int roi_end_h = round(bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    float bin_size_h = static_cast<float>(roi_height)
                       / static_cast<float>(pooled_height);
    float bin_size_w = static_cast<float>(roi_width)
                       / static_cast<float>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    float maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int w = wstart; w < wend; ++w) {
      for (int h = hstart; h < hend; ++h) {
        int bottom_index = w * height + h;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

__global__ void ROIPoolBackward(const int nthreads, const float* top_diff,
    const int* argmax_data, const int num_rois, const float spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, float* bottom_diff,
    const float* bottom_rois) {
  
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads; index += blockDim.x * gridDim.x){
    // (h, w, c, n) coords in bottom data
    int w = index % height;
    int h = (index / height) % width;
    int c = (index / height / width) % channels;
    int n = index / height / width / channels;

    float gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const float* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
      int roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
      int roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
      int roi_end_h = round(offset_bottom_rois[4] * spatial_scale);

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const float* offset_top_diff = top_diff + offset;
      const int* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      float bin_size_h = static_cast<float>(roi_height)
                         / static_cast<float>(pooled_height);
      float bin_size_w = static_cast<float>(roi_width)
                         / static_cast<float>(pooled_width);

      int phstart = floor(static_cast<float>(h - roi_start_h) / bin_size_h);
      int phend = ceil(static_cast<float>(h - roi_start_h + 1) / bin_size_h);
      int pwstart = floor(static_cast<float>(w - roi_start_w) / bin_size_w);
      int pwend = ceil(static_cast<float>(w - roi_start_w + 1) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int pw = pwstart; pw < pwend; ++pw) {
        for (int ph = phstart; ph < phend; ++ph) {
          if (offset_argmax_data[pw * pooled_height + ph] == (w * height + h)) {
            gradient += offset_top_diff[pw * pooled_height + ph];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}