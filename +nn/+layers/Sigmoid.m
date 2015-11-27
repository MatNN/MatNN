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
        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            data.val{l.top} = 1./(1+exp(-data.val{l.bottom}));
        end
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            bottom_diff = data.diff{l.top}.*(data.val{l.top}.*(1-data.val{l.top}));
            data = nn.utils.accumulateData(opts, data, l, bottom_diff);
        end
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.bottom)==1);
            assert(numel(l.top)==1);
        end
    end

end
