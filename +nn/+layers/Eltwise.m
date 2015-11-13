classdef Eltwise < nn.layers.template.BaseLayer
% this layer did not implement .f(), .gf(), .b(), .gb()

    properties (SetAccess = protected, Transient)
        default_eltwise_param = {
            'operation' 'sum' ... % sum, prod, minus, max
        };
    end

    methods
        function varargout = plus(~, in1, in2, out_diff)
            if isempty(out_diff)
                varargout{1} = in1+in2;
            else
                varargout{1} = out_diff;
                varargout{2} = out_diff;
            end
        end

        function varargout = prod(~, in1, in2, out_diff)
            if isempty(out_diff)
                varargout{1} = in1.*in2;
            else
                varargout{1} = in2.*out_diff;
                varargout{2} = in1.*out_diff;
            end
        end

        function varargout = max(~, in1, in2, out_diff)
            if isempty(out_diff)
                varargout{1} = max(in1, in2);
            else
                r = max(in1,in2);
                varargout{1} = out_diff.*(r==in1);
                varargout{2} = out_diff.*(r==in2);
            end
        end

        function varargout = minus(~, in1, in2, out_diff)
            if isempty(out_diff)
                varargout{1} = in1-in2;
            else
                varargout{1} = out_diff;
                varargout{2} = -out_diff;
            end
        end

        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            top{1} = obj.(obj.params.eltwise.operation)(bottom{1}, bottom{2}, []);
        end
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            [bottom_diff{1}, bottom_diff{2}] = obj.(obj.params.eltwise.operation)(bottom{1}, bottom{2}, top_diff{1});
        end

        function outSizes = outputSizes(obj, opts, inSizes)
            assert(isequal(inSizes{1},inSizes{2}));
            outSizes = inSizes(1);
        end
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==2);
            assert(numel(baseProperties.top)==1);
        end

    end

end