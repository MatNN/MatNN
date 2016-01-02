classdef SimpleLayer < nn.layers.template.BaseLayer
%SIMPLELAYER
% Inherit this class if your layer accepts 1 input, 1 output, GPUcode=CPUcode and no params

    methods
        function out = f(~, in)
        end
        function in_diff = b(~, in, out_diff)
        end
        function forward(obj)
            net = obj.net;
            data = net.data;
            data.val{obj.top} = obj.f(data.val{obj.bottom});
            data.forwardCount(obj.bottom, obj.top);
        end
        function backward(obj)
            net = obj.net;
            data = net.data;
            data.backwardCount(obj.bottom,  obj.top, obj.b(data.val{obj.bottom}, data.diff{obj.top}));
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.bottom)==1);
            assert(numel(obj.top)==1);
        end
    end

end
