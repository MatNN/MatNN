classdef Multiply < nn.layers.template.BaseLayer

    properties (SetAccess = protected, Transient)
        default_multiply_param = {
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

        function forward(obj)
            p = obj.params.multiply;
            net = obj.net;
            data = net.data;

            in1 = data.val{obj.bottom(1)};
            in2 = data.val{obj.bottom(2)};
            
            % compute output size
            outSizes = size(in2);
            if p.transpose
                outSizes(1) = size(in1, 2);
            else
                outSizes(1) = size(in1, 1);
            end

            if net.opts.gpu
                data.val{obj.top} = obj.gf(p.transpose, in1, in2, outSizes);
            else
                data.val{obj.top} = obj.f(p.transpose, in1, in2, outSizes);
            end
            data.forwardCount(obj.bottom, obj.top);
        end

        function backward(obj)
            p = obj.params.multiply;
            net = obj.net;
            data = net.data;
            if net.opts.gpu
                [bottom_diff1, bottom_diff2] = obj.gb(p.transpose, data.val{obj.bottom}, data.diff{obj.top});
            else
                [bottom_diff1, bottom_diff2] = obj.b(p.transpose, data.val{obj.bottom}, data.diff{obj.top});
            end
            data.backwardCount(obj.bottom, obj.top, bottom_diff1, bottom_diff2);
        end

        function outSizes =  outputSizes(obj, inSizes)
            p = obj.params.multiply;
            if p.transpose
                outSizes = {[inSizes{1}(2), inSizes{2}(2), inSizes{1}(3:end)]};
            else
                outSizes = {[inSizes{1}(1), inSizes{2}(2), inSizes{1}(3:end)]};
            end
        end

        function outSizes = setup(obj, inSizes)
            assert(inSizes{1}(3)==inSizes{2}(3));
            assert(inSizes{1}(4)==inSizes{2}(4));
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            p = obj.params.multiply;
            assert(numel(obj.bottom)==2);
            assert(numel(obj.top) == 1);
            if p.transpose
                assert(inSizes{1}(1) == inSizes{2}(1));
            else
                assert(inSizes{1}(2) == inSizes{2}(1));
            end
        end
    end
end
