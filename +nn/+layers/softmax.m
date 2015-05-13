function o = softmax(varargin)
%LOGISTICLOSS 
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
    function [outputBlob, weightUpdate] = forward(opts, l, weights, blob)
        weightUpdate = {};

        y = exp( bsxfun(@minus, blob{1}, max(blob{1}, [], 3)) );
        y = bsxfun(@rdivide, y, sum(y,3));
        outputBlob = { y };

    end
    function [outputdzdx, outputdzdw] = backward(opts, l, weights, blob, dzdy)
        outputdzdw = {};

        y = exp( bsxfun(@minus, blob{1}, max(blob{1}, [], 3)) );
        y = bsxfun(@rdivide, y, sum(y,3));
        y = y .* bsxfun(@minus, dzdy{1}, sum(dzdy{1} .* y, 3));
        outputdzdx = { y };

    end
end