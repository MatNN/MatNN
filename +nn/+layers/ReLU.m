classdef ReLU < nn.layers.template.SimpleLayer

    properties (Access = protected, Transient)
        zero = single(0); %to be removed
    end

    methods
        function v = propertyDevice(obj)
            v = obj.propertyDevice@nn.layers.template.BaseLayer();
            v.zero = 2;
        end
        function out = f(~, in)
            %out = max(in, 0);
            out = cudnn.activationForward(1, in);
        end
        function in_diff = b(~, in, out_diff)
            %in_diff = (in > 0) .* out_diff;
            in_diff = cudnn.activationBackward(1, in, out_diff, out_diff);
        end
    end

end
