classdef Replicate < nn.layers.template.BaseLayer
% REPLICATE
% This layer accepts assigned data and copy to top layer; however, data should be cell.
% Each element of cell should be 4D array.

    properties (SetAccess = protected, Transient)
        default_replicate_param = {
            'value' rand(4,4)...
        };
    end

    methods
        function f(~, varargin)
            error('Not supported.');
        end

        function b(~,varargin)
            error('Not supported.');
        end

        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            p = obj.params.replicate;
            value = p.value;
            for i=1:numel(value)
                singleValue = single(value{i});
                if opts.gpuMode
                    data.val{l.top(i)} = gpuArray(singleValue);
                else
                    data.val{l.top(i)} = singleValue;
                end
            end
        end

        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            data = nn.utils.accumulateData(opts, data, l);
        end

        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            p = obj.params.replicate;
            sizes = {};
            for i = 1:numel(p.value)
                sizes{i} = size(p.value{i});
            end
            outSizes = sizes;
        end

        function setParams(obj, l)
            obj.setParams@nn.layers.template.BaseLayer(l);
            p = obj.params.replicate;
            assert(~isempty(p.value));
        end

        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.top)==1);
        end
    end

end
