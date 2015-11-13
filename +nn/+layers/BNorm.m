classdef BNorm < nn.layers.template.BaseLayer & nn.layers.template.hasWeight
%BNORM Batch normalization

    properties (SetAccess = protected, Transient)
        default_conv_param = {
              'num_output' 1     ...
             'kernel_size' [3 3] ...
                     'pad' [0 0] ...
                  'stride' [1 1] ...
        };
    end

    methods
        function out = f(~, in, w1, w2)
            out = vl_nnbnorm(in, w1, w2);
        end
        function [in_diff, w1_diff, w2_diff] = b(~, in, out_diff, w1, w2)
            [ in_diff, w1_diff, w2_diff ] = vl_nnbnorm(in, w1, w2, out_diff);
        end
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            if ~opts.layerSettings.enableBnorm
                top{1} = bottom{1};
            else
                top{1} = obj.f(bottom{1}, weights{1}, weights{2});
            end
        end
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff) %#ok
            if ~opts.layerSettings.enableBnorm
                bottom_diff{1} = top_diff{1};
            else
                [bottom_diff{1}, weights_diff{1}, weights_diff{2}] = obj.b(bottom{1}, top_diff{1}, weights{1}, weights{2});
            end
        end
        function resources = createResources(obj, opts, inSizes)
            resources.weight = {[],[]};
            resources.weight{1} = obj.params.weight.generator{1}([1, 1, inSizes{1}(3), 1], obj.params.weight.generator_param{1});
            resources.weight{2} = obj.params.weight.generator{2}([1, 1, inSizes{1}(3), 1], obj.params.weight.generator_param{2});
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
            assert(all(obj.params.weight.enable_terms), 'All weights must be enabled.');
        end
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==1);
            assert(numel(baseProperties.top)==1);
        end
    end
    
end