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
        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            p = obj.params.deconv;
            data.val{l.top} = obj.f(data.val{l.bottom}, net.weights{l.weights(1)}, net.weights{l.weights(2)}, p.crop, p.upsampling, p.num_group);
        end
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            p = obj.params.deconv;
            [bottom_diff, weights_diff{1}, weights_diff{2}] = obj.b(data.val{l.bottom}, data.diff{l.top}, net.weights{l.weights(1)}, net.weights{l.weights(2)}, p.crop, p.upsampling, p.num_group);
            data = nn.utils.accumulateData(opts, data, l, bottom_diff);
            net  = nn.utils.accumulateWeight(net, l.weights, weights_diff{:});
        end
        function resources = createResources(obj, opts, l, inSizes, varargin)
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
        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            p = obj.params.deconv;
            btmSize = inSizes{1};

            assert(mod(btmSize(3), p.num_group)==0 && btmSize(3)>=p.num_group);

            outSizes = {[floor([(btmSize(1)-1)*p.upsampling(1)-p.crop(1)-p.crop(2)+p.kernel_size(1), ...
                           (btmSize(2)-1)*p.upsampling(2)-p.crop(3)-p.crop(4)+p.kernel_size(2)]), ...
                            p.num_output*p.num_group, btmSize(4)]};
        end
        function setParams(obj, l)
            obj.setParams@nn.layers.template.BaseLayer(l);
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
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.bottom)==1);
            assert(numel(l.top)==1);
        end

    end
    
end
