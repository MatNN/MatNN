classdef Pooling < nn.layers.template.BaseLayer

    properties (SetAccess = protected, Transient)
        default_pooling_param = {
                 'method' 'max' ...
            'kernel_size' [1 1] ...
                    'pad' 0     ...
                 'stride' [1 1] ...
        };
    end

    methods
        function out = f(~, in, kernel, pad, stride, method)
            out = vl_nnpool(in, kernel, 'pad', pad, 'stride', stride, 'method', method);
        end
        function in_diff = b(~, in, out_diff, kernel, pad, stride, method) %#ok
            in_diff = vl_nnpool(in, kernel, out_diff, 'pad', pad, 'stride', stride, 'method', method);
        end

        function forward(obj)
            p = obj.params.pooling;
            data = obj.net.data;
            data.val{obj.top} = vl_nnpool(data.val{obj.bottom}, p.kernel_size, 'pad', p.pad, 'stride', p.stride, 'method', p.method);
            obj.net.data.forwardCount(obj.bottom, obj.top);
        end
        function backward(obj)
            p = obj.params.pooling;
            data = obj.net.data;
            data.backwardCount(obj.bottom,  obj.top, vl_nnpool(data.val{obj.bottom}, p.kernel_size, data.diff{obj.top}, 'pad', p.pad, 'stride', p.stride, 'method', p.method));
        end

        function outSizes = outputSizes(obj, inSizes)
            p = obj.params.pooling;
            btmSize = inSizes{1};

            outSizes = {[floor([(btmSize(1)+p.pad(1)+p.pad(2)-p.kernel_size(1))/p.stride(1)+1, ...
                                (btmSize(2)+p.pad(3)+p.pad(4)-p.kernel_size(2))/p.stride(2)+1]), ...
                         btmSize(3), btmSize(4)]};
        end
        function setParams(obj)
            obj.setParams@nn.layers.template.BaseLayer();
            p = obj.params.pooling;
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
            obj.params.pooling = p;
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.bottom)==1);
            assert(numel(obj.top)==1);
        end
    end

end
