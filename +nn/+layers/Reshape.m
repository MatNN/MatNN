classdef Reshape < nn.layers.template.BaseLayer

    properties (SetAccess = protected, Transient)
        default_reshape_param = {
            'output_size'   {[],1,1,0} % 0=current size
        };
    end


    methods
        function out = f(~, in, output_size)
            origShape = nn.utils.size4D(in);
            for i=1:4
                if output_size{i} == 0
                    output_size{i} = origShape(i);
                end
            end
            out = reshape(in, output_size{:});
        end
        function in_diff = b(~, in, out_diff)
            in_diff = reshape(out_diff, nn.utils.size4D(in));
        end
        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            data.val{l.top} = obj.f(data.val{l.bottom}, obj.params.reshape.output_size);
        end
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            data = nn.utils.accumulateData(opts, data, l, obj.b(data.val{l.bottom}, data.diff{l.top}));
        end
        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            % replace 0
            os = obj.params.reshape.output_size;
            for i=1:4
                if os{i} == 0
                    os{i} = inSizes{1}(i);
                end
            end
            % test reshape size
            tmpData = false(inSizes{1});
            tmpData = reshape(tmpData, os{:});
            outSizes = {nn.utils.size4D(tmpData)};
        end
        function setParams(obj, l)
            obj.setParams@nn.layers.template.BaseLayer(l);
            if sum(cellfun('isempty', obj.params.reshape.output_size)) >=2
                error('there should be 1 unknown output size');
            end
        end
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.bottom)==1);
            assert(numel(l.top)==1);
        end
    end
end