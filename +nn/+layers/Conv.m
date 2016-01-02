classdef Conv < nn.layers.template.WeightLayer
    
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
            out = vl_nnconv(in, w1, w2, 'pad', pad, 'stride', stride, 'CudnnWorkspaceLimit', 512*1024*1024);
        end
        function [in_diff, w1_diff, w2_diff] = b(obj, in, out_diff, w1, w2, pad, stride) %#ok
            [in_diff, w1_diff, w2_diff ] = vl_nnconv(in, w1, w2, out_diff, 'pad', pad, 'stride', stride, 'CudnnWorkspaceLimit', 512*1024*1024);
        end
        function forward(obj)
            p = obj.params.conv;
            data = obj.net.data;

            data.val{obj.top} = vl_nnconv(data.val{obj.bottom}, data.val{obj.weights}, 'pad', p.pad, 'stride', p.stride, 'CudnnWorkspaceLimit', 512*1024*1024);
            data.forwardCount(obj.bottom, obj.top);
            data.forwardCount(obj.weights, []);
        end
        function backward(obj)
            p = obj.params.conv;
            data = obj.net.data;

            [bottom_diff, weights_diff1, weights_diff2] = vl_nnconv(data.val{obj.bottom}, data.val{obj.weights}, data.diff{obj.top}, 'pad', p.pad, 'stride', p.stride, 'CudnnWorkspaceLimit', 512*1024*1024);
            data.backwardCount(obj.bottom,  obj.top, bottom_diff);
            data.backwardCount(obj.weights, [],      weights_diff1, weights_diff2);
            %nn.utils.accumulateDiff(data, obj.bottom,  obj.top, bottom_diff);
            %nn.utils.accumulateDiff(data, obj.weights, [],      weights_diff1, weights_diff2);
        end
        function createResources(obj, inSizes)
            p = obj.params.conv;
            btmSize = inSizes{1};
            if obj.params.weight.enable_terms(1)
                w1Size = [p.kernel_size(1), p.kernel_size(2), btmSize(3), p.num_output];
            end
            if obj.params.weight.enable_terms(2)
                w2Size = [1, p.num_output];
            end
            obj.createResources@nn.layers.template.WeightLayer(inSizes, w1Size, w2Size);
        end
        function outSizes = outputSizes(obj, inSizes)
            p = obj.params.conv;
            btmSize     = inSizes{1};

            outSizes = {[floor([(btmSize(1)+p.pad(1)+p.pad(2)-p.kernel_size(1))/p.stride(1)+1, ...
                                (btmSize(2)+p.pad(3)+p.pad(4)-p.kernel_size(2))/p.stride(2)+1]), ...
                         p.num_output, ...
                         btmSize(4)]};
        end
        function setParams(obj)
            obj.setParams@nn.layers.template.BaseLayer();
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
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.bottom)==1);
            assert(numel(obj.top)==1);
        end

    end
    
end
