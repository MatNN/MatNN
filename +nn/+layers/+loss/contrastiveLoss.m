function o = contrastiveLoss(architecture)
%CONTRASTIVELOSS 
%
% NOTICE
%   label index starts from 0 (compatible with other NN tools)
%   you can specify begining index from parameter

if nargin == 0
    architecture = 'default';
end

o.name         = 'contrastiveLoss';
o.generateLoss = true;

% process architecture
if strcmp(architecture, 'cuda kernel')
    o.forward         = @forward_CUDAKernel;
    o.backward        = @backward_CUDAKernel;
    o.setup           = @setup_CUDAKernel;
    cuKernel.forward  = [];
    cuKernel.backward = [];
    d                 = 0;
else
    o.setup        = @setup;
    o.forward      = @forward;
    o.backward     = @backward;
    d_ = 0;
    d = 0;
end


default_contrastiveLoss_param = {
    'labelIndex_start' single(0)     ...
           'threshold' single(1e-20) ...
              'margin' single(1)     ...
};

    function [resource, topSizes, param] = setup(l, bottomSizes)
        resource = {};

        if isfield(l, 'contrastiveLoss_param')
            wp = nn.utils.vararginHelper(default_contrastiveLoss_param, l.contrastiveLoss_param);
        else
            wp = nn.utils.vararginHelper(default_contrastiveLoss_param, default_contrastiveLoss_param);
        end
        param.contrastiveLoss_param = wp;

        assert(numel(l.bottom)==3);
        assert(numel(l.top)==1);
        
        resSize1 = bottomSizes{1};
        resSize2 = bottomSizes{1};
        ansSize = bottomSizes{3};

        assert(isequal(resSize1, resSize2));
        assert(isequal(resSize1(1:2), resSize2(1:2)) && isequal(resSize1(1:2), [1 1]));
        assert(isequal(ansSize(1:3), [1 1 1]));
        assert(isequal(ansSize(4), resSize1(4)));
        % Label size must be Nx1, 1xN or 1x1x1xN;

        topSizes = {[1, 1, 1, 1]};

    end
    function [resource, topSizes, param] = setup_CUDAKernel(l, bottomSizes)
        [resource, topSizes, param] = setup(l, bottomSizes);
        ptxFile = fullfile('+nn','+layers','+loss','contrastiveLoss.ptx');
        cuFile = fullfile('+nn','+layers','+loss','contrastiveLoss.cu');

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


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        d_ = bottom{1}-bottom{2};
        d = sum((d_).^2, 3);
        y = bottom{3};
        E = 0.5 * sum(  y.*d + (1-y).*max(l.contrastiveLoss_param.margin - d, single(0))  );%/size(blob{1},4);
        top{1} = E;
    end
    function [top, weights, misc] = forward_CUDAKernel(opts, l, weights, misc, bottom, top)
        d = sum((bottom{1}-bottom{2}).^2, 3);
        %d = sum(arrayfun(@(a,b) (a-b).^2, blob{1}, blob{2} ), 3);
        E = feval(cuKernel.forward, 0, bottom{3}, d, l.contrastiveLoss_param.margin, numel(d));
        top{1} = E;
    end

    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        bottom_diff = cell(1,3);
        m_d = l.contrastiveLoss_param.margin - d;
        rightTerm = d_;
        rightTerm(:,:,:,m_d(:)<=0) = 0;
        y = bottom{3};
        bottom_diff{1} = top_diff{1} * (bsxfun(@times, d_, y) - bsxfun(@times, rightTerm, 1-y));% / size(blob{1}, 4);
        bottom_diff{2} = -bottom_diff{1};
    end
    function [bottom_diff, weights_diff, misc] = backward_CUDAKernel(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        [bottom_diff{1}, bottom_diff{2}] = feval(cuKernel.backward, bottom{1}, bottom{2}, top_diff{1}, bottom{3}, d, l.contrastiveLoss_param.margin, numel(bottom{1}), numel(bottom{1})/numel(bottom{3}));
    end
end