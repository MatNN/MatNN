classdef Silence < nn.layers.template.BaseLayer

    methods
        function f(~, varargin)
            error('Not supported.');
        end
        function varargout = b(~, varargin)
            error('Not supported.');
        end
        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            %top = {};
        end
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            if opts.gpuMode
                zero = gpuArray(single(0));
                for i=1:numel(l.bottom)
                    bottom_diff = data.val{l.bottom}*zero;
                end
            else
                for i=1:numel(l.bottom)
                    bottom_diff = data.val{l.bottom}*single(0);
                end
            end
            data = nn.utils.accumulateData(opts, data, l, bottom_diff);
        end
        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            outSizes = {};
        end
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.bottom)>=0);
            assert(numel(l.top)==0);
        end
    end

end