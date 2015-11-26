classdef Tanh < nn.layers.template.BaseLayer
%TANH tanh()

    methods
        function out = f(~, in)
            out = tanh(in);
        end
        function in_diff = b(~, in, out_diff)
            in_diff = out_diff.*(1-tanh(in).^2);
        end
        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            data.val{l.top} = tanh(data.val{l.bottom});
        end
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            data = nn.utils.accumulateData(opts, data, l, data.diff{l.top}.*(1-data.val{l.top}.^2));
        end
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.bottom)==1);
            assert(numel(l.top)==1);
        end
    end

end
