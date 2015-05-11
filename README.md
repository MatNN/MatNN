# MatNN: CNN framework for MATLAB

**MatNN** is a MATLAB toolbox for computer vision applications. It aims to provide the similarity of [Caffe](http://caffe.berkeleyvision.org) , and elastic workflow of [MatConvNet](http://www.vlfeat.org/matconvnet).

This toolbox requires [MatConvNet](http://www.vlfeat.org/matconvnet) toolbox to be installed in your system and added to your Matlab path.

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
to make traning efficiency and the entire framework is built on top of [MatConvNet](http://www.vlfeat.org/matconvnet).

## License

**MatNN** is under the Simplified BSD License.
If you use **MatNN** in your work, please refer to the homepage of this project.
