classdef Crop < nn.layers.template.BaseLayer
%CROP Crop Large data to a smaller size
%  NOTICE:
%    bottom{1} is the desired blob need to be croped
%    bottom{2} provide the size to crop bottom{1}

    properties (SetAccess = protected, Transient)
        default_crop_param = {
            'offset' [] ...  % A 2-element vector, indicates the offset of H,W to crop
                          % if set to [], means use the center position
        };
    end

    methods
        function out = f(~, in1, in2, offset)
            s = [1,1,1,1];
            sizeofBlob2 = size(in2);%small
            sizeofBlob1 = size(in1);%large
            s(1:numel(sizeofBlob2)) = sizeofBlob2;

            if isempty(offset)
                offset = -round((s(1:2) - sizeofBlob1(1:2))/2);% compatible with FCN's crop layer??
            end
            out = in1(offset(1)+1:offset(1)+s(1), offset(2)+1:offset(2)+s(2), :, :);
        end
        function [in1_diff, in2_diff] = b(~, in1, in2, out_diff, offset)
            s = [1,1,1,1];
            sizeofBlob2 = size(in2);%small
            sizeofBlob1 = size(in1);%large
            s(1:numel(sizeofBlob2)) = sizeofBlob2;

            if isempty(offset)
                offset = -round((s(1:2) - sizeofBlob1(1:2))/2);% compatible with FCN's crop layer??
            end
            in1_diff = in.*single(0);
            in1_diff(offset(1)+1:offset(1)+s(1), offset(2)+1:offset(2)+s(2), :, :) = out_diff;
            in2_diff = [];
        end
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            top{1} = obj.f(bottom{1}, bottom{2}, obj.params.crop.offset);
        end
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            [bottom_diff{1}, bottom_diff{2}] = obj.b(bottom{1}, bottom{2}, top_diff{1}, obj.params.crop.offset);
        end
        function outSizes = outputSizes(obj, opts, inSizes)
            HW1 = inSizes{1}(1:2);
            HW2 = inSizes{2}(1:2);
            % use the smallest bottom size as top size
            if HW1(1) >= HW2(1) && HW1(2) >= HW1(2)
                outSizes = {[inSizes{2}(1:2), inSizes{1}(3:4)]};
            else
                error('Crop Layer bottom size is wrong.');
            end
        end
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==2);
            assert(numel(baseProperties.top)==1);
        end
    end
end