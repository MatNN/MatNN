classdef Equal < nn.layers.template.BaseLayer
%EQUAL Compare two inputs
% NO BACKWARD

    methods
        function out = f(~, in1, in2)
            out = single(in1 == in2);
        end
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            top{1} = single(bottom{1} == bottom{2});
        end
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            bottom_diff = {[],[]};
        end
        function outSizes = outputSizes(obj, opts, inSizes)
            assert(isequal(size(inSizes{1}), size(inSizes{2})));
            outSizes = inSizes(1);
        end
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==2);
            assert(numel(baseProperties.top)==1);
        end
    end

end
