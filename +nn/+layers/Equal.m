classdef Equal < nn.layers.template.BaseLayer
%EQUAL Compare two inputs
% NO BACKWARD

    methods
        function out = f(~, in1, in2)
            out = single(in1 == in2);
        end
        function forward(obj)
            data = obj.net.data;
            data.val{obj.top} = single(data.val{obj.bottom(1)} == data.val{obj.bottom(2)});
            data.forwardCount(obj.bottom, obj.top);
        end
        function backward(obj)
            obj.net.data.backwardCount(obj.bottom, obj.top, [], []);
        end
        function outSizes = outputSizes(~, inSizes)
            assert(isequal(size(inSizes{1}), size(inSizes{2})));
            outSizes = inSizes(1);
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.bottom)==2);
            assert(numel(obj.top)==1);
        end
    end

end
