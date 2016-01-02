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

        function forward(obj)
            p = obj.params.replicate;
            data = obj.net.data;
            value = p.value;
            for i=1:numel(value)
                singleValue = single(value{i});
                if obj.net.opts.gpu
                    data.val{obj.top(i)} = gpuArray(singleValue);
                else
                    data.val{obj.top(i)} = singleValue;
                end
            end
            data.forwardCount(obj.bottom, obj.top);
        end

        function backward(obj)
            obj.net.data.backwardCount(obj.bottom,  obj.top);
        end

        function outSizes = outputSizes(obj, inSizes)
            p = obj.params.replicate;
            sizes = {};
            for i = 1:numel(p.value)
                sizes{i} = size(p.value{i});
            end
            outSizes = sizes;
        end

        function setParams(obj)
            obj.setParams@nn.layers.template.BaseLayer();
            p = obj.params.replicate;
            assert(~isempty(p.value));
        end

        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.top)==1);
        end
    end

end
