classdef Reshape < handle

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
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            if opts.gpuMode
                top{1} = obj.gf(bottom{1}, obj.params.reshape.output_size);
            else
                top{1} = obj.f(bottom{1}, obj.params.reshape.output_size);
            end
        end
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            if opts.gpuMode
                bottom_diff{1} = obj.gb(bottom{1}, top_diff{1});
            else
                bottom_diff{1} = obj.b(bottom{1}, top_diff{1});
            end
        end
        function outSizes = outputSizes(obj, opts, inSizes)
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
        function setParams(obj, baseProperties)
            obj.setParams@nn.layers.template.BaseLayer(baseProperties);
            if sum(cellfun('isempty', obj.params.reshape.output_size)) >=2
                error('there should be 1 unknown output size');
            end
        end
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==1);
            assert(numel(baseProperties.top)==1);
        end
    end
end