classdef Equal < nn.layers.template.BaseLayer
%EQUAL Compare two inputs
% NO BACKWARD

    methods
        function out = f(~, in1, in2)
            out = single(in1 == in2);
        end
        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            data.val{l.top} = single(data.val{l.bottom{1}} == data.val{l.bottom{2}});
        end
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            data = nn.utils.accumulateData(opts, data, l);
        end
        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            assert(isequal(size(inSizes{1}), size(inSizes{2})));
            outSizes = inSizes(1);
        end
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.bottom)==2);
            assert(numel(l.top)==1);
        end
    end

end
