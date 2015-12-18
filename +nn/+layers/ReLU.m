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
            out = max(in, 0);
        end
        function in_diff = b(~, in, out_diff)
            in_diff = (in > 0) .* out_diff;
        end
    end

end
