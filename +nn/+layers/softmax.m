function o = softmax(varargin)
%SOFTMAX
%
% NOTICE
%   label index starts from 0 (compatible with other NN tools)
%   you can specify begining index from parameter

o.name         = 'Softmax';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;

    function [resource, topSizes, param] = setup(l, bottomSizes)
        resource = {};

        assert(numel(l.bottom)==1);
        assert(numel(l.top)==1);

        topSizes = bottomSizes;

        param = {};

    end
    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        y = exp( bsxfun(@minus, bottom{1}, max(bottom{1}, [], 3)) );
        y = bsxfun(@rdivide, y, sum(y,3));
        top{1} = y;

    end
    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        y = exp( bsxfun(@minus, bottom{1}, max(bottom{1}, [], 3)) );
        y = bsxfun(@rdivide, y, sum(y,3));
        y = y .* bsxfun(@minus, top_diff{1}, sum(top_diff{1} .* y, 3));
        bottom_diff = { y };
    end
end
