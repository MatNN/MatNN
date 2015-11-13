classdef Silence < nn.layers.template.BaseLayer

    methods
        function f(~, varargin)
            error('Not supported.');
        end
        function varargout = b(~, varargin)
            error('Not supported.');
        end
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            top = {};
        end
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            if opts.gpuMode
                zero = gpuArray(single(0));
                for i=1:numel(bottom)
                    bottom_diff{1} = bottom{1}*zero;
                end
            else
                for i=1:numel(bottom)
                    bottom_diff{1} = bottom{1}*single(0);
                end
            end
        end
        function outSizes = outputSizes(obj, opts, inSizes)
            outSizes = {};
        end
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)>=0);
            assert(numel(baseProperties.top)==0);
        end
    end

end