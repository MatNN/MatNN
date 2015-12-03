classdef SoftMax < nn.layers.template.BaseLayer

    methods
        function out = f(~, in)
            out = exp( bsxfun(@minus, in, max(in, [], 3)) );
            out = bsxfun(@rdivide, out, sum(y,3));
        end
        function in_diff = b(~, out, out_diff)
            in_diff = out .* bsxfun(@minus, out_diff, sum(out_diff .* out, 3));
        end
        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            y = exp( bsxfun(@minus, bottom{1}, max(bottom{1}, [], 3)) );
            y = bsxfun(@rdivide, y, sum(y,3));
            data.val{l.top} = y;
        end
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            bottom_diff = data.val{l.top} .* bsxfun(@minus, data.diff{l.top}, sum(data.diff{l.top} .* data.val{l.top}, 3));
            data = nn.utils.accumulateData(opts, data, l, bottom_diff);
        end
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.bottom)==1);
            assert(numel(l.top)==1);
        end
    end

end