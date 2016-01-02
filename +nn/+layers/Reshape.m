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
        function forward(obj)
            data = obj.net.data;
            data.val{obj.top} = obj.f(data.val{obj.bottom}, obj.params.reshape.output_size);
            data.forwardCount(obj.bottom, obj.top);
        end
        function backward(obj)
            data = obj.net.data;
            data.backwardCount(obj.bottom,  obj.top, obj.b(data.val{obj.bottom}, data.diff{obj.top}));
        end
        function outSizes = outputSizes(obj, inSizes)
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
        function setParams(obj)
            obj.setParams@nn.layers.template.BaseLayer();
            if sum(cellfun('isempty', obj.params.reshape.output_size)) >=2
                error('there should be 1 unknown output size');
            end
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.bottom)==1);
            assert(numel(obj.top)==1);
        end
    end
end