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
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            if opts.gpuMode
                top{1} = obj.gf(bottom{1});
            else
                top{1} = obj.f(bottom{1});
            end
        end
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            if opts.gpuMode
                bottom_diff{1} = obj.gb(bottom{1}, top{1}, top_diff{1});
            else
                bottom_diff{1} = obj.b(bottom{1}, top{1}, top_diff{1});
            end
        end
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==1);
            assert(numel(baseProperties.top)==1);
            if opts.gpuMode
                obj.zero = gpuArray.zeros(1, 'single');
            end
        end
        % function varargout = moveTo(obj, dest)
        %     if numel(nargout) == 0
        %         obj.params = obj.moveTo_private(dest, obj.params);
        %         obj.zero = obj.moveTo_private(dest, obj.zero);
        %     elseif numel(nargout) == 1
        %         o = struct();
        %         o.params = obj.moveTo_private(dest, obj.params);
        %         o.zero = obj.moveTo_private(dest, obj.zero);
        %         o.didSetup = obj.didSetup;
        %         varargout{1} = o;
        %     else
        %         error('Too many outputs.');
        %     end
        % end

    end

end
