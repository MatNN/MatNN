classdef Sigmoid < nn.layers.template.SimpleLayer
%SIGMOID
    methods
        function out = f(~, in)
            out = 1./(1+exp(-in));
        end
        function in_diff = b(~, in, out_diff)
            sigmoid =  1./(1+exp(-in)) ;
            in_diff = out_diff.*(sigmoid.*(1-sigmoid));
        end
    end
end
