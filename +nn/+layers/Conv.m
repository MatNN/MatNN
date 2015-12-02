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
        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            p = obj.params.conv;
            data.val{l.top} = vl_nnconv(data.val{l.bottom}, net.weights{l.weights(1)}, net.weights{l.weights(2)}, 'pad', p.pad, 'stride', p.stride);
        end
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            p = obj.params.conv;
            [bottom_diff, weights_diff{1}, weights_diff{2}] = vl_nnconv(data.val{l.bottom}, net.weights{l.weights(1)}, net.weights{l.weights(2)}, data.diff{l.top}, 'pad', p.pad, 'stride', p.stride);
            data = nn.utils.accumulateData(opts, data, l, bottom_diff);
            net  = nn.utils.accumulateWeight(net, l.weights, weights_diff{:});
        end
        function resources = createResources(obj, opts, l, inSizes, varargin)
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
        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            p = obj.params.conv;
            btmSize     = inSizes{1};

            outSizes = {[floor([(btmSize(1)+p.pad(1)+p.pad(2)-p.kernel_size(1))/p.stride(1)+1, ...
                                (btmSize(2)+p.pad(3)+p.pad(4)-p.kernel_size(2))/p.stride(2)+1]), ...
                         p.num_output, ...
                         btmSize(4)]};
        end
        function setParams(obj, l)
            obj.setParams@nn.layers.template.BaseLayer(l);
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
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.bottom)==1);
            assert(numel(l.top)==1);
        end

    end
    
end
