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

- ***Minimal reuqirement of external libraries.*** We will provide a pure matlab code version and a matlab+cuda kernel version, so your code is portable (no need to compile external libraries). And also include the basic version based on MatConvNet. Because you can easily customize a layer, so any exists libraries, like CUDA/cuBlas/cuDNN can be added to your workflow if you want. A cuBlas/cuDNN version will be considered after this project is out of beta. Note that even the pure matlab version will have GPU support from parallel computing toolbox of Matlab.
- ***Provide the elastic workflow and maintain the computation efficiency.*** We will separate each core functions into modules so that you can do less work to make them fit your need.
- ***Test your ideas and validate them quickly.*** Although learning other tools/programming languages may not be a problem, use you exists matlab code with MatNN is time reserved compared to learn a new language. We provide familiar definition of Caffe, if you have learned it, then you can get started quickly.

## Installation

Prerequisite: **Parallel computing toolbox** for Matlab

1. Download source code
2. Install [MatConvNet](http://www.vlfeat.org/matconvnet), and set matlab path
2. (optional) use NVCC to compile .cu code into .ptx
3. Done!

## License

**MatNN** is under the Simplified BSD License.
If you use **MatNN** in your work, please refer to the homepage of this project.
