classdef Bilinear < nn.layers.template.BaseLayer

    properties (SetAccess = protected, Transient)
        default_bilinear_param = {
            'transpose' true...
        };
    end

    methods
        function varargout = f(transpose, in1, in2)
            if transpose
                varargout = in1' * in2;
            else
                varargout = in1 * in2;
            end
        end

        function [in1_diff, in2_diff] = b(transpose, in1, in2, out_diff)
            if transpose
                in1_diff = in2 * out_diff';
                in2_diff = in1 * out_diff;
            else
                in1_diff = in2 * out_diff;
                in2_diff = in1' * out_diff;
            end
        end

        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            p = obj.params.bilinear;
            data.val{l.top} = obj.f(p.transpose, data.val{l.bottom(1)}, data.val{l.bottom(2)});
        end

        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            p = obj.params.bilinear;
            [bottom_diff{1}, bottom_diff{2}] = obj.b(p.transpose, data.val{l.bottom(1)}, data.val{l.bottom(2)}, data.diff{l.top});
            data = nn.utils.accumulateData(opts, data, l, bottom_diff{:});
        end

        function outSizes =  outputSizes(obj, opts, l, inSizes, varargin)
            p = obj.params.bilinear;
            if p.transpose
                outSizes = size(inSizes{1}(2), inSizes{2}(2));
            else
                outSizes = size(inSizes{1}(1), inSizes{2}(2));
            end
        end

        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            p = obj.params.bilinear;
            assert(numel(l.bottom)==2);
            assert(numel(l.top) == 1);
            if p.transpose
                assert(inSizes{1}(1) == inSizes{2}(1));
            else
                assert(inSizes{1}(2) == inSizes{2}(1));
            end
        end
    end
end
