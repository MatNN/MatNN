# MatNN: A NN framework for MATLAB

**MatNN** is a MATLAB framework for neural network training and testing. It aims to provide the similarity of [Caffe](http://caffe.berkeleyvision.org), and elastic workflow of [MatConvNet](http://www.vlfeat.org/matconvnet).

This toolbox requires [MatConvNet](http://www.vlfeat.org/matconvnet) to be installed in your system and added to your MATLAB path.

## Features

**MatNN** provides some features you may familiar with Caffe or other CNN tools:
- Weight Sharing
- Multiple Losses
- Custom data sampling algorithm
- Custom layer with custom parameters/weights/loss/...
- Multiple GPUs support

Note that we don't provide data layers, you should design your data sampling and fetching routines.

## Functionality

**MatNN** uses the CUDA/C++ code from [MatConvNet](http://www.vlfeat.org/matconvnet)
to make your training progress more efficient, and the entire framework is built on top of [MatConvNet](http://www.vlfeat.org/matconvnet).

## Goal

- Minimal reuqirement of external library. We will provide pure matlab version, matlab+cuda kernel version, mex+cuda+cublas version and mex+cuda+cublas+cuDNN version of this project in the future. Note that even the pure matlab version will have GPU support from parallel computing toolbox of Matlab.
- Provide the elastic workflow and maintain the computation efficiency. We will try to separate each core functions into modules so that you can easily modify to fit your needs.

## License

**MatNN** is under the Simplified BSD License.
If you use **MatNN** in your work, please refer to the homepage of this project.
