function o = relu(architecture)
%RELU Rectified linear unit

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
        cuKernel.forward.ThreadBlockSize = maxThreads;
        cuKernel.forward.GridSize = maxBlocks;

        % setup backward kernel
        cuKernel.backward = parallel.gpu.CUDAKernel(ptxFile, cuFile, 'backward');
        num = floor(prod(bottomSizes{1})/cuKernel.backward.MaxThreadsPerBlock);
        if num >= 1
            maxThreads = cuKernel.backward.MaxThreadsPerBlock;
        else
            maxThreads = bottomSizes{1}(4);
        end
        maxBlocks = ceil(prod(bottomSizes{1})/maxThreads);
        cuKernel.backward.ThreadBlockSize = maxThreads;
        cuKernel.backward.GridSize = maxBlocks;
    end


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        top{1} = max(bottom{1}, 0);
    end
    function [top, weights, misc] = forward_CUDAKernel(opts, l, weights, misc, bottom, top)
        top{1} = feval(cuKernel.forward, bottom{1}, numel(bottom{1}));
    end

    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        bottom_diff{1} = (bottom{1} > 0) .* top_diff{1};
    end
    function [bottom_diff, weights_diff, misc] = backward_CUDAKernel(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        bottom_diff{1} = feval(cuKernel.backward, bottom{1}, top_diff{1}, numel(bottom{1}));
    end

end