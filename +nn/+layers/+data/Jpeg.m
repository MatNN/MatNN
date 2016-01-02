classdef Jpeg < nn.layers.template.DataLayer

    properties (SetAccess = protected, Transient, GetAccess=public)
        default_jpeg_param = {
            'is_color' true  ... % if true, all image will have 3 channel, even gray image. if false, all image have one channel anf force to gray
                'mean' []    ... % [r,g,b] for color image, one value for gray image, [] means don't substract mean
              'resize' [256] ... % one value for min(resized_width, resized_height), two value for actual size [H, W], empty = not resize
                'crop' [224] ... % one value for square, two values for rectangle
         'crop_center' false ...
                'flip' false ...
        };
    end

    methods
        function [data, label] = process(obj, rawdata, index, usegpu)
            p = obj.params.jpeg;
            for i = 1:numel(rawdata)
                if usegpu
                    img = gpuArray(single(rawdata{i}));
                end

                % channel manipulation
                if p.is_color
                    if size(img, 3) == 1
                        img = cat(3, img,img,img);
                    end
                else
                    if size(img, 3) == 3
                        img = rgb2gray(img);
                    end
                end

                %resize
                if numel(p.resize) == 1
                    img = imresize(img, p.resize/min(size(img,1),size(img,2)));%, 'bilinear');
                elseif numel(p.resize) == 2
                    img = imresize(img, p.resize, 'bilinear');
                end
                imgSize = size(img);

                %crop
                if numel(p.crop) == 1
                    if ~p.crop_center
                        minB = min(imgSize(1), imgSize(2));
                        Hs = randi(minB-p.crop+1)-1;
                        Ws = randi(minB-p.crop+1)-1;
                        Hs = Hs+floor((imgSize(1)-minB)/2)+1;
                        Ws = Ws+floor((imgSize(2)-minB)/2)+1;
                    else
                        Hs = floor((imgSize(1)-p.crop)/2)+1;
                        Ws = floor((imgSize(2)-p.crop)/2)+1;
                    end
                    img = img(Hs:(Hs+p.crop-1), Ws:(Ws+p.crop-1), :);
                elseif numel(p.crop) == 2
                    if ~p.crop_center
                        Hs = randi(imgSize(1)-p.crop(1)+1);
                        Ws = randi(imgSize(2)-p.crop(2)+1);
                    else
                        Hs = floor((imgSize(1)-p.crop(1))/2)+1;
                        Ws = floor((imgSize(2)-p.crop(2))/2)+1;
                    end
                    img = img(Hs:(Hs+p.crop(1)-1), Ws:(Ws+p.crop(2)-1), :);
                end

                %flip
                if p.flip
                    if rand() > 0.5
                        rawdata{i} = fliplr(img);
                    else
                        rawdata{i} = img;
                    end
                else
                    rawdata{i} = img;
                end
                

            end

            data = single(cat(4,rawdata{:}));

            if numel(p.mean)==size(data,3)
                data = bsxfun(@minus, data, reshape(single(p.mean),1,1,3,1));
            else
                data = data-single(p.mean);
            end

            data  = data(:,:,[3,2,1],:); %BGR
            label = single(reshape(obj.dataLabel(index),1,1,1,[]));
            if usegpu
                label = gpuArray(label);
            end
        end


        function setParams(obj)
            obj.setParams@nn.layers.template.DataLayer();
            %Read file
            p = obj.params.data;
            j = obj.params.jpeg;
            if ischar(p.src)
                f = fopen(p.src, 'r');
                tmp = textscan(f,'%s %d','Delimiter',{' ', '\t'});
                fclose(f);

                assert(all(~cellfun('isempty',tmp{1})) || all(cellfun('isempty',tmp{1})), 'Source file format is wrong.');
                obj.dataPaths = tmp{1};
                assert(numel(tmp{2}) == numel(tmp{1}), 'Source file format is wrong.');
                obj.dataLabel = tmp{2};
            else
                assert(all(~cellfun('isempty',p.src{1})) || all(cellfun('isempty',p.src{1})), 'Source{1} format is wrong.');
                obj.dataPaths = p.src{1};
                assert(numel(p.src{2}) == numel(p.src{1}), 'Source{2} format is wrong.');
                obj.dataLabel = p.src{2};
            end
            assert(numel(obj.dataPaths) == numel(obj.dataLabel), 'Image number and label number do not match.');

            if numel(j.resize) == 1
                if numel(j.crop) == 1
                    assert(j.resize>=j.crop);
                else
                    assert(j.resize>=max(j.crop));
                end
            elseif numel(j.resize) == 2
                if numel(j.crop) == 1
                    assert(min(j.resize)>=j.crop);
                else
                    assert(j.resize(1)>=j.crop(1) && j.resize(2)>=j.crop(2));
                end
            else
                errror('resize must be a value or a 1x2 vector.');
            end
            assert(numel(j.crop)>0 && numel(j.crop)<=2);
        end

        % Calc Output sizes
        function outSizes = outputSizes(obj, inSizes)
            p = obj.params.jpeg;
            if numel(p.crop)==1
                crop = [p.crop,p.crop];
            else
                crop = p.crop;
            end
            outSizes{1} = [crop(1), crop(2), p.is_color*2+1, obj.params.data.batch_size];
            outSizes{2} = [1, 1, 1, obj.params.data.batch_size];
        end

        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.DataLayer(inSizes);
            assert(numel(obj.bottom)==0, 'JPEG layer does not accept inputs.');
        end


        function varargout = read(obj, imgList, nThreads, isPrefetch)
            if isPrefetch
                vl_imreadjpeg(imgList, 'NumThreads', nThreads, 'Prefetch');
            else
                varargout{1} = vl_imreadjpeg(imgList, 'NumThreads', nThreads);
            end
        end
    end
    
end