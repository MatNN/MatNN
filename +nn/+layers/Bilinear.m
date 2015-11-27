classdef Bilinear < nn.layers.template.BaseLayer

    properties (SetAccess = protected, Transient)
        default_bilinear_param = {
            'transpose' true...
        };
    end

    methods
        function out = f(~, transpose, in1, in2, outSizes)
            out = zeros(outSizes, 'single');
            for i = 1:size(in2,3)*size(in2,4)
                in1_ = in1(:,:,i);
                in2_ = in2(:,:,i);
                if transpose
                    out(:,:,i) = in1_' * in2_;
                else
                    out(:,:,i) = in1_ * in2_;
                end
            end
        end

        function out = gf(~, transpose, in1, in2, outSizes)
            if transpose
                in1 = permute(in1, [2 1 3 4]);
            end
            out = pagefun(@mtimes, in1, in2);
        end

        function [in1_diff, in2_diff] = b(~, transpose, in1, in2, out_diff)
            in1_diff = zeros(size(in1), 'single');
            in2_diff = zeros(size(in2), 'single');
            for i = 1:size(in2,3)*size(in2,4)
                in1_ = in1(:,:,i);
                in2_ = in2(:,:,i);
                out_diff_ = out_diff(:,:,i);
                if transpose
                    in1_diff(:,:,i) = in2_ * out_diff_';
                    in2_diff(:,:,i) = in1_ * out_diff_;
                else
                    in1_diff(:,:,i) = out_diff_ * in2_';
                    in2_diff(:,:,i) = in1_' * out_diff_;
                end
            end
        end

        function [in1_diff, in2_diff] = gb(~, transpose, in1, in2, out_diff)
            if transpose
                out_diff_T = permute(out_diff, [2 1 3 4]);
                in1_diff = pagefun(@mtimes, in2, out_diff_T);
                in2_diff = pagefun(@mtimes, in1, out_diff);
            else
                in2 = permute(in2, [2 1 3 4]);
                in1 = permute(in1, [2 1 3 4]);
                in1_diff = pagefun(@mtimes, out_diff, in2);
                in2_diff = pagefun(@mtimes, in1, out_diff);
            end
        end

        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            p = obj.params.bilinear;
            in1 = data.val{l.bottom(1)};
            in2 = data.val{l.bottom(2)};
            
            % compute output size
            outSizes = size(in2);
            if p.transpose
                outSizes(1) = size(in1, 2);
            else
                outSizes(1) = size(in1, 1);
            end

            if opts.gpuMode
                data.val{l.top} = obj.gf(p.transpose, in1, in2, outSizes);
            else
                data.val{l.top} = obj.f(p.transpose, in1, in2, outSizes);
            end
        end

        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            p = obj.params.bilinear;
            if opts.gpuMode
                [bottom_diff{1}, bottom_diff{2}] = obj.gb(p.transpose, data.val{l.bottom(1)}, data.val{l.bottom(2)}, data.diff{l.top});
            else
                [bottom_diff{1}, bottom_diff{2}] = obj.b(p.transpose, data.val{l.bottom(1)}, data.val{l.bottom(2)}, data.diff{l.top});
            end
            data = nn.utils.accumulateData(opts, data, l, bottom_diff{:});
        end

        function outSizes =  outputSizes(obj, opts, l, inSizes, varargin)
            p = obj.params.bilinear;
            if p.transpose
                outSizes = {[inSizes{1}(2), inSizes{2}(2), inSizes{1}(3:end)]};
            else
                outSizes = {[inSizes{1}(1), inSizes{2}(2), inSizes{1}(3:end)]};
            end
        end

        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            assert(inSizes{1}(3)==inSizes{2}(3));
            assert(inSizes{1}(4)==inSizes{2}(4));
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
