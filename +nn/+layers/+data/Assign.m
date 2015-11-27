classdef Assign < nn.layers.template.BaseLayer
% ASSIGN
% This layer accepts assigned data; however, data should be cell.
% Each element of cell should be 4D array.

    properties (SetAccess = protected, Transient)
        default_assign_param = {
            'value' random(4,4)...
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
            p = obj.params.assign;
            value = p.value;
            input_data = p.value;
            for i=1:numel(sizes)
                if opts.gpuArray
                    data.val{l.top(i)} = gpuArray(value{i}, 'single');
                else
                    data.val{l.top(i)} = single(value{i});
                end
            end
        end

        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            data = nn.utils.accumulateData(opts, data, l);
        end

        function setParams(obj, l)
            obj.setParams@nn.layers.template.BaseLayer(l);
            p = obj.params.assign;
            assert(~isempty(p.value));
        end

        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.top)==1);
        end
    end

end
