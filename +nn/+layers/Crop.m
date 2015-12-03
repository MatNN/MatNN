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
            in1_diff = in1.*single(0);
            in1_diff(offset(1)+1:offset(1)+s(1), offset(2)+1:offset(2)+s(2), :, :) = out_diff;
            in2_diff = [];
        end
        function forward(obj, nnObj, l, opts, data, net)
            data.val{l.top} = obj.f(data.val{l.bottom(1)}, data.val{l.bottom(2)}, obj.params.crop.offset);
        end
        function backward(obj, nnObj, l, opts, data, net)
            [bottom_diff{1}, bottom_diff{2}] = obj.b(data.val{l.bottom(1)}, data.val{l.bottom(2)}, data.diff{l.top}, obj.params.crop.offset);
            nn.utils.accumulateData(opts, data, l, bottom_diff{:});
        end
        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            HW1 = inSizes{1}(1:2);
            HW2 = inSizes{2}(1:2);
            % use the smallest bottom size as top size
            if HW1(1) >= HW2(1) && HW1(2) >= HW1(2)
                outSizes = {[inSizes{2}(1:2), inSizes{1}(3:4)]};
            else
                error('Crop Layer bottom size is wrong.');
            end
        end
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.bottom)==2);
            assert(numel(l.top)==1);
        end
    end
end