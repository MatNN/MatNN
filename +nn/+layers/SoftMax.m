classdef SoftMax < nn.layers.template.BaseLayer

    methods
        function out = f(~, in)
            out = exp( bsxfun(@minus, in, max(in, [], 3)) );
            out = bsxfun(@rdivide, out, sum(y,3));
        end
        function in_diff = b(~, out, out_diff)
            in_diff = out .* bsxfun(@minus, out_diff, sum(out_diff .* out, 3));
        end
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            y = exp( bsxfun(@minus, bottom{1}, max(bottom{1}, [], 3)) );
            y = bsxfun(@rdivide, y, sum(y,3));
            top{1} = y;
        end
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            bottom_diff{1} = top{1} .* bsxfun(@minus, top_diff{1}, sum(top_diff{1} .* top{1}, 3));
        end
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==1);
            assert(numel(baseProperties.top)==1);
        end
    end

end