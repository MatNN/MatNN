classdef ReLU < nn.layers.template.BaseLayer

    properties (Access = protected, Transient)
        zero = single(0);
    end

    methods
        function v = propertyDevice(obj)
            v = obj.propertyDevice@nn.layers.template.BaseLayer();
            v.zero = 2;
        end
        function out = f(~, in)
            out = max(in, 0);
        end
        function in_diff = b(~, in, out, out_diff) %#ok
            in_diff = (in > 0) .* out_diff;
        end
        function out = gf(obj, in)
            out = max(in, obj.zero); %fatest
            %  ptx kernel            %sceond
            %out = in.*(in > 0);     %third
            %in(in < 0) = 0          % ?
            %out = (in + abs(in))/2  % ?
            %
        end
        function in_diff = gb(obj, in, out, out_diff) %#ok
            in_diff = nn.utils.gpu.relu_b(in, out_diff, obj.zero);
            %in_diff = (in > obj.zero) .* out_diff;
        end
        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            if opts.gpuMode
                data.val{l.top} = obj.gf(data.val{l.bottom});
            else
                data.val{l.top} = obj.f(data.val{l.bottom});
            end
        end
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            if opts.gpuMode
                data = nn.utils.accumulateData(opts, data, l, obj.gb(data.val{l.bottom}, data.val{l.top}, data.diff{l.top}));
            else
                data = nn.utils.accumulateData(opts, data, l, obj.b(data.val{l.bottom}, data.val{l.top}, data.diff{l.top}));
            end
        end
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.bottom)==1);
            assert(numel(l.top)==1);
            if opts.gpuMode
                obj.zero = gpuArray.zeros(1, 'single');
            end
        end
    end

end
