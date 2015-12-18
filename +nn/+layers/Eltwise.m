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

        function forward(obj)
            data = obj.net.data;
            data.val{obj.top} = obj.(obj.params.eltwise.operation)(data.val{obj.bottom}, []);
            data.forwardCount(obj.bottom, obj.top);
        end
        function backward(obj)
            data = obj.net.data;
            [bottom_diff{1}, bottom_diff{2}] = obj.(obj.params.eltwise.operation)(data.val{obj.bottom}, data.diff{obj.top});
            data.backwardCount(obj.bottom,  obj.top, bottom_diff{:});
        end

        function outSizes = outputSizes(obj, inSizes)
            assert(isequal(inSizes{1},inSizes{2}));
            outSizes = inSizes(1);
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.bottom)==2);
            assert(numel(obj.top)==1);
        end

    end

end