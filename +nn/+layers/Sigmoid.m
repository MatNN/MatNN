classdef Sigmoid < nn.layers.template.BaseLayer
%SIGMOID

    methods
        function out = f(~, in)
            out = 1./(1+exp(-in));
        end
        function in_diff = b(~, in, out_diff)
            sigmoid =  1./(1+exp(-in)) ;
            in_diff = out_diff.*(sigmoid.*(1-sigmoid));
        end
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            top{1} = 1./(1+exp(-bottom{1}));
        end
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            sigmoid =  1./(1+exp(-bottom{1})) ;
            bottom_diff{1} = top_diff{1}.*(sigmoid.*(1-sigmoid));
        end
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==1);
            assert(numel(baseProperties.top)==1);
        end
    end

end
