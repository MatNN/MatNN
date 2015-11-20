classdef Tanh < nn.layers.template.BaseLayer
%TANH tanh()

    methods
        function out = f(~, in)
            out = tanh(in);
        end
        function in_diff = b(~, in, out_diff)
            in_diff = out_diff.*(1-tanh(in).^2);
        end
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            top{1} = tanh(bottom{1});
        end
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            bottom_diff{1} = top_diff{1}.*(1-tanh(bottom{1}).^2);
        end
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==1);
            assert(numel(baseProperties.top)==1);
        end
    end

end
