classdef Tanh < nn.layers.template.SimpleLayer
%TANH tanh()
    methods
        function out = f(~, in)
            out = tanh(in);
        end
        function in_diff = b(~, in, out_diff)
            in_diff = out_diff.*(1-tanh(in).^2);
        end
    end
end
