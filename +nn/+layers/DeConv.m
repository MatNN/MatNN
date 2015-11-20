classdef DeConv < nn.layers.template.BaseLayer & nn.layers.template.hasWeight
    
    % Default parameters
    properties (SetAccess = protected, Transient)
        default_deconv_param = {
             'num_output' 1     ...
            'kernel_size' [3 3] ...
                   'crop' [0 0] ...
             'upsampling' [1 1] ...
              'num_group' 1     ... 
        };
    end

    methods
        function out = f(obj, in, w1, w2, crop, upsample, numgroups) %#ok
            out = vl_nnconvt(in, w1, w2, 'crop', crop, 'upsample', upsample, 'numgroups', numgroups);
        end
        function [in_diff, w1_diff, w2_diff] = b(obj, in, out_diff, w1, w2, crop, upsample, numgroups) %#ok
            [ in_diff, w1_diff, w2_diff ] = vl_nnconvt(in, w1, w2, out_diff, 'crop', crop, 'Upsample', upsample, 'numgroups', numgroups);
        end
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc) %#ok
            p = obj.params.deconv;
            top{1} = obj.f(bottom{1}, bottom{2}, weights{1}, weights{2}, p.crop, p.upsampling, p.num_group);
        end
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff) %#ok
            p = obj.params.deconv;
            [bottom_diff{1}, weights_diff{1}, weights_diff{2}] = obj.b(bottom{1}, top_diff{1}, weights{1}, weights{2}, p.crop, p.upsampling, p.num_group);
        end
        function resources = createResources(obj, opts, inSizes)
            p = obj.params.deconv;
            btmSize = inSizes{1};

            resources.weight = {[],[]};
            if wp1.enable_terms(1)
                resources.weight{1} = wp1.generator{1}([p.kernel_size(1), p.kernel_size(2), p.num_output, btmSize(3)], obj.params.weight.generator_param{1});
            end
            if wp1.enable_terms(2)
                resources.weight{2} = wp1.generator{2}([1, p.num_output*p.num_group], obj.params.weight.generator_param{2});
            end
        end
        function outSizes = outputSizes(obj, opts, inSizes)
            p = obj.params.deconv;
            btmSize = inSizes{1};

            assert(mod(btmSize(3), p.num_group)==0 && btmSize(3)>=p.num_group);

            outSizes = {[floor([(btmSize(1)-1)*p.upsampling(1)-p.crop(1)-p.crop(2)+p.kernel_size(1), ...
                           (btmSize(2)-1)*p.upsampling(2)-p.crop(3)-p.crop(4)+p.kernel_size(2)]), ...
                            p.num_output*p.num_group, btmSize(4)]};
        end
        function setParams(obj, baseProperties)
            obj.setParams@nn.layers.template.BaseLayer(baseProperties);
            p = obj.params.deconv;
            assert(all(p.upsampling~=0));
            if numel(p.kernel_size) == 1
                p.kernel_size = [p.kernel_size, p.kernel_size];
            end
            if numel(p.upsampling) == 1
                p.upsampling = [p.upsampling, p.upsampling];
            end
            if numel(p.crop) == 1
                p.crop = [p.crop, p.crop, p.crop, p.crop];
            elseif numel(p.crop) == 2
                p.crop = [p.crop(1), p.crop(1), p.crop(2), p.crop(2)];
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
