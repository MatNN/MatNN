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
        % CPU Forward
        function out = f(~, in, kernel, pad, stride, method)
            out = vl_nnpool(in, kernel, 'pad', pad, 'stride', stride, 'method', method);
        end
        % CPU Backward
        function in_diff = b(~, in, out_diff, kernel, pad, stride, method) %#ok
            in_diff = vl_nnpool(in, kernel, out_diff, 'pad', pad, 'stride', stride, 'method', method);
        end

        % Forward function for training/testing routines
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            p = obj.params.pooling;
            top{1} = obj.f(bottom{1}, p.kernel_size, p.pad, p.stride, p.method);
        end
        % Backward function for training/testing routines
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            p = obj.params.pooling;
            bottom_diff{1} = obj.b(bottom{1}, top_diff{1}, p.kernel_size, p.pad, p.stride, p.method);
        end

        function outSizes = outputSizes(obj, opts, inSizes)
            p = obj.params.pooling;
            btmSize = inSizes{1};

            outSizes = {[floor([(btmSize(1)+p.pad(1)+p.pad(2)-p.kernel_size(1))/p.stride(1)+1, ...
                                (btmSize(2)+p.pad(3)+p.pad(4)-p.kernel_size(2))/p.stride(2)+1]), ...
                         btmSize(3), btmSize(4)]};
        end
        function setParams(obj, baseProperties)
            obj.setParams@nn.layers.template.BaseLayer(baseProperties);
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

        % Setup function for training/testing routines
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==1);
            assert(numel(baseProperties.top)==1);
        end

    end

end
