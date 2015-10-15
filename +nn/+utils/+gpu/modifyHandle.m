function kern = modifyHandle(N, kern)
%COUNTBLOCKNUMBER replace the built-in parallel.gpu.CUDAKernel with element number N

d = gpuDevice(); % get current gpu device
kern.GridSize = floor( N/d.MaxThreadsPerBlock );
kern.MaxThreadsPerBlock = d.MaxThreadsPerBlock;
kern.ThreadBlockSize = d.MaxThreadsPerBlock;

end