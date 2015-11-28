function kern = createHandle(N, varargin)
%COUNTBLOCKNUMBER replace the built-in parallel.gpu.CUDAKernel with element number N

d = gpuDevice(); % get current gpu device
kern = parallel.gpu.CUDAKernel(varargin{:});

kern.GridSize = ceil( N/d.MaxThreadsPerBlock );
kern.ThreadBlockSize = d.MaxThreadsPerBlock;

end