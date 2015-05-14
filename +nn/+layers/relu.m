function o = relu(architecture)
%RELU Compute mean class accuracy for you

o.name         = 'ReLU';
o.generateLoss = false;
if strcmp(architecture, 'cuda kernel')
    o.setup           = @setup_CUDAKernel;
    o.forward         = @forward_CUDAKernel;
    o.backward        = @backward_CUDAKernel;
    cuKernel.forward  = [];
    cuKernel.backward = [];
else
    o.setup        = @setup;
    o.forward      = @forward;
    o.backward     = @backward;
end


    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update
        resource = {};
        assert(numel(l.bottom)==1);
        assert(numel(l.top)==1);
        topSizes = bottomSizes(1);
        %return updated param
        param = {};
    end
    function [resource, topSizes, param] = setup_CUDAKernel(l, bottomSizes)
        [resource, topSizes, param] = setup(l, bottomSizes);
        ptxFile = fullfile('+nn','+layers','relu.ptx');
        cuFile = fullfile('+nn','+layers','relu.cu');

        % setup forward kernel
        cuKernel.forward = parallel.gpu.CUDAKernel(ptxFile, cuFile, 'forward');
        num = floor(prod(bottomSizes{1})/cuKernel.forward.MaxThreadsPerBlock);
        if num >= 1
            maxThreads = cuKernel.forward.MaxThreadsPerBlock;
        else
            maxThreads = bottomSizes{1}(4);
        end
        maxBlocks = ceil(prod(bottomSizes{1})/maxThreads);
        cuKernel.forward.ThreadBlockSize = [maxThreads, 1, 1];
        cuKernel.forward.GridSize = [maxBlocks, 1, 1];

        % setup backward kernel
        cuKernel.backward = parallel.gpu.CUDAKernel(ptxFile, cuFile, 'backward');
        num = floor(prod(bottomSizes{1})/cuKernel.backward.MaxThreadsPerBlock);
        if num >= 1
            maxThreads = cuKernel.backward.MaxThreadsPerBlock;
        else
            maxThreads = bottomSizes{1}(4);
        end
        maxBlocks = ceil(prod(bottomSizes{1})/maxThreads);
        cuKernel.backward.ThreadBlockSize = [maxThreads, 1, 1];
        cuKernel.backward.GridSize = [maxBlocks, 1, 1];
    end


    function [outputBlob, weightUpdate] = forward(opts, l, weights, blob)
        outputBlob{1} = max(blob{1}, 0);
        weightUpdate = {};
    end
    function [outputBlob, weightUpdate] = forward_CUDAKernel(opts, l, weights, blob)
        weightUpdate = {};
        outputBlob{1} = feval(cuKernel.forward, blob{1}, numel(blob{1}));
    end

    function [outputdzdx, outputdzdw] = backward(opts, l, weights, blob, dzdy)
        %numel(outputdzdx) = numel(blob), numel(outputdzdw) = numel(weights)
        outputdzdx{1} = (blob{1} > 0) .* dzdy{1};
        outputdzdw = {};
    end
    function [outputdzdx, outputdzdw] = backward_CUDAKernel(opts, l, weights, blob, dzdy)
        outputdzdx{1} = feval(cuKernel.backward, blob{1}, dzdy{1}, numel(blob{1}));
        outputdzdw = {};
    end

end