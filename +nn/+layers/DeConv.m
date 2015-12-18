classdef DeConv < nn.layers.template.WeightLayer
    
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
        function forward(obj)
            p = obj.params.deconv;
            data = obj.net.data;
            data.val{obj.top} = obj.f(data.val{obj.bottom}, data.val{obj.weights}, p.crop, p.upsampling, p.num_group);
            data.forwardCount(obj.bottom, obj.top);
        end
        function backward(obj)
            p = obj.params.deconv;
            data = obj.net.data;
            [bottom_diff, weights_diff1, weights_diff2] = obj.b(data.val{obj.bottom}, data.diff{obj.top}, net.weights{obj.weights}, p.crop, p.upsampling, p.num_group);
            data.backwardCount(obj.bottom,  obj.top, bottom_diff);
            data.backwardCount(obj.weights, [],      weights_diff1, weights_diff2);
        end
        function createResources(obj, inSizes)
            p = obj.params.deconv;
            btmSize = inSizes{1};

            if obj.params.weight.enable_terms(1)
                w1Size = [p.kernel_size(1), p.kernel_size(2), p.num_output, btmSize(3)];
            end
            if obj.params.weight.enable_terms(2)
                w2Size = [1, p.num_output*p.num_group];
            end
            obj.createResources@nn.layers.template.WeightLayer(inSizes, w1Size, w2Size);
        end
        function outSizes = outputSizes(obj, inSizes)
            p = obj.params.deconv;
            btmSize = inSizes{1};

            assert(mod(btmSize(3), p.num_group)==0 && btmSize(3)>=p.num_group);

            outSizes = {[floor([(btmSize(1)-1)*p.upsampling(1)-p.crop(1)-p.crop(2)+p.kernel_size(1), ...
                           (btmSize(2)-1)*p.upsampling(2)-p.crop(3)-p.crop(4)+p.kernel_size(2)]), ...
                            p.num_output*p.num_group, btmSize(4)]};
        end
        function setParams(obj)
            obj.setParams@nn.layers.template.BaseLayer();
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
            obj.params.deconv = p;
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.bottom)==1);
            assert(numel(obj.top)==1);
        end

    end
    
end
