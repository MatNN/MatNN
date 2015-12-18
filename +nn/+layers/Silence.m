classdef Silence < nn.layers.template.BaseLayer

    methods
        function f(~, varargin)
            error('Not supported.');
        end
        function varargout = b(~, varargin)
            error('Not supported.');
        end
        function forward(obj)
            data.forwardCount(obj.bottom, []);
        end
        function backward(obj)
            a = cell(1, numel(obj.bottom));
            data.backwardCount(obj.bottom,  obj.top, a{:});
        end
        function outSizes = outputSizes(obj, inSizes)
            outSizes = {};
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.bottom)>=0);
            assert(numel(obj.top)==0);
        end
    end

end