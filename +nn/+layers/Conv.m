classdef Conv < nn.layers.template.BaseLayer & nn.layers.template.hasWeight
    
    % Default parameters
    properties (SetAccess = protected, Transient)
        default_conv_param = {
              'num_output' 1     ...
             'kernel_size' [3 3] ...
                     'pad' [0 0] ...
                  'stride' [1 1] ...
        };
    end

    methods
        function out = f(obj, in, w1, w2, pad, stride) %#ok
            out = vl_nnconv(in, w1, w2, 'pad', pad, 'stride', stride);
        end
        function [in_diff, w1_diff, w2_diff] = b(obj, in, out_diff, w1, w2, pad, stride) %#ok
            [in_diff, w1_diff, w2_diff ] = vl_nnconv(in, w1, w2, out_diff, 'pad', pad, 'stride', stride);
        end
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc) %#ok
            p = obj.params.conv;
            top{1} = obj.f(bottom{1}, weights{1}, weights{2}, p.pad, p.stride);
        end
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff) %#ok
            p = obj.params.conv;
            [bottom_diff{1}, weights_diff{1}, weights_diff{2}] = obj.b(bottom{1}, top_diff{1}, weights{1}, weights{2}, p.pad, p.stride);
        end

        % Create resources (weight, misc)
        function resources = createResources(obj, opts, inSizes)
            p = obj.params.conv;
            btmSize = inSizes{1};

            resources.weight = {[],[]};
            if obj.params.weight.enable_terms(1)
                resources.weight{1} = obj.params.weight.generator{1}([p.kernel_size(1), p.kernel_size(2), btmSize(3), p.num_output], obj.params.weight.generator_param{1});
            end
            if obj.params.weight.enable_terms(2)
                resources.weight{2} = obj.params.weight.generator{2}([1, p.num_output], obj.params.weight.generator_param{2});
            end
        end

        function outSizes = outputSizes(obj, opts, inSizes)
            p = obj.params.conv;
            btmSize     = inSizes{1};

            outSizes = {[floor([(btmSize(1)+p.pad(1)+p.pad(2)-p.kernel_size(1))/p.stride(1)+1, ...
                                (btmSize(2)+p.pad(3)+p.pad(4)-p.kernel_size(2))/p.stride(2)+1]), ...
                         p.num_output, ...
                         btmSize(4)]};
        end
        function setParams(obj, baseProperties)
            obj.setParams@nn.layers.template.BaseLayer(baseProperties);
            p = obj.params.conv;
            assert(all(p.stride~=0));
            if numel(p.kernel_size) == 1
                p.kernel_size = [p.kernel_size, p.kernel_size];
            end
            if numel(p.stride) == 1
                p.stride = [p.stride, p.stride];
            end
            if numel(p.pad) == 1
                p.pad = [p.pad, p.pad, p.pad, p.pad];
            elseif numel(p.pad) == 2
                p.pad = [p.pad(1), p.pad(1), p.pad(2), p.pad(2)];
            end
            obj.params.conv = p;
        end
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==1);
            assert(numel(baseProperties.top)==1);
        end

    end
    
end
