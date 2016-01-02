classdef SoftMax < nn.layers.template.BaseLayer

    methods
        function out = f(~, in)
            out = exp( bsxfun(@minus, in, max(in, [], 3)) );
            out = bsxfun(@rdivide, out, sum(y,3));
        end
        function in_diff = b(~, out, out_diff)
            in_diff = out .* bsxfun(@minus, out_diff, sum(out_diff .* out, 3));
        end
        function forward(obj)
            data = obj.net.data;
            y = exp( bsxfun(@minus, data.val{obj.bottom}, max(data.val{obj.bottom}, [], 3)) );
            y = bsxfun(@rdivide, y, sum(y,3));
            data.val{obj.top} = y;
            data.forwardCount(obj.bottom, obj.top);
        end
        function backward(obj)
            data = obj.net.data;
            bottom_diff = data.val{obj.top} .* bsxfun(@minus, data.diff{obj.top}, sum(data.diff{obj.top} .* data.val{obj.top}, 3));
            data.backwardCount(obj.bottom,  obj.top, bottom_diff);
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.bottom)==1);
            assert(numel(obj.top)==1);
        end
    end

end