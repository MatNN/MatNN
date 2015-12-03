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

        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            data.val{l.top} = obj.(obj.params.eltwise.operation)(data.val{l.bottom(1)}, data.val{l.bottom(2)}, []);
        end
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            [bottom_diff{1}, bottom_diff{2}] = obj.(obj.params.eltwise.operation)(data.val{l.bottom(1)}, data.val{l.bottom(2)}, data.diff{l.top});
            data = nn.utils.accumulateData(opts, data, l, bottom_diff{:});
        end

        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            assert(isequal(inSizes{1},inSizes{2}));
            outSizes = inSizes(1);
        end
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.bottom)==2);
            assert(numel(l.top)==1);
        end

    end

end