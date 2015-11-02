function o = jpegData(varargin)
%JPEGDATA
% NOTICE
% 1. the output number of data is <= batch_size
% 2. this function uses MatConvNet's vl_imreadjpeg to do prefetch

o.name         = 'JpegData';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;
o.outputSize   = [];


default_jpegData_param = {
         'source' []    ... % A string of a file, which contains image paths with label, or a cell variable of image paths and label
    'root_folder' []    ...
       'is_color' true  ... % if true, all image will have 3 channel, even gray image. if false, all image have one channel anf force to gray
           'mean' []    ... % [r,g,b] for color image, one value for gray image, [] means don't substract mean
     'batch_size' 1     ...
         'resize' [256] ... % one value for min(resized_width, resized_height), two value for actual size [H, W], empty = not resize
           'crop' [224] ... % one value for square, two values for rectangle
    'crop_center' false ...
        'shuffle' false ...
           'flip' false ...
       'prefetch' true  ...
'prefetch_thread' 2     ...
};

imagePaths   = {};
imageLabel   = [];
batchIndices = []; % 2-D matrix, each column is the image indices. And column ind is the batch ind.
pointer      = 0;
cache        = {};
imgList      = {};
imgListNext  = {};
imgInd       = [];
imgIndNext   = [];

    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update


        if isfield(l, 'jpegData_param')
            wp = nn.utils.vararginHelper(default_jpegData_param, l.jpegData_param);
        else
            wp = nn.utils.vararginHelper(default_jpegData_param, default_jpegData_param);
        end

        assert(numel(l.bottom)==0, [o.name, ' layer does not accept inputs.']);


        %Read file
        if ischar(wp.source)
            f = fopen(wp.source, 'r');
            tmp = textscan(f,'%s %d','Delimiter',{' ', '\t'});
            fclose(f);

            assert(all(~cellfun('isempty',tmp{1})) || all(cellfun('isempty',tmp{1})), 'Source file format is wrong.');
            imagePaths = tmp{1};
            assert(numel(tmp{2}) == numel(tmp{1}), 'Source file format is wrong.');
            imageLabel = tmp{2};
        else
            assert(all(~cellfun('isempty',wp.source{1})) || all(cellfun('isempty',wp.source{1})), 'Source{1} format is wrong.');
            imagePaths = wp.source{1};
            assert(numel(wp.source{2}) == numel(wp.source{1}), 'Source{2} format is wrong.');
            imageLabel = wp.source{2};
        end
        assert(numel(imagePaths) == numel(imageLabel), 'Image number and label number do not match.');
        if numel(wp.resize) == 1
            if numel(wp.crop) == 1
                assert(wp.resize>=wp.crop);
            else
                assert(wp.resize>=max(wp.crop));
            end
        elseif numel(wp.resize) == 2
            if numel(wp.crop) == 1
                assert(min(wp.resize)>=wp.crop);
            else
                assert(wp.resize(1)>=wp.crop(1) && wp.resize(2)>=wp.crop(2));
            end
        else
            errror('resize must be a value or a 1x2 vector.');
        end
        assert(numel(wp.crop)>0 && numel(wp.crop)<=2);

        if numel(wp.crop)==1
            crop = [wp.crop,wp.crop];
        else
            crop = wp.crop;
        end
        topSizes = @(x) {[crop(1), crop(2), wp.is_color*2+1, wp.batch_size], [1,1,1, wp.batch_size]};
        resource = {};

        %return updated param
        param.jpegData_param = wp;
    end


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        % Get images
        if isempty(imgList)
            [imgList, imgInd] = generateNextImgList(opts, l);
        end
        s = numel(imgList); 
            
        if l.jpegData_param.prefetch
            cache = vl_imreadjpeg(imgList, 'NumThreads', l.jpegData_param.prefetch_thread);
            [imgListNext, imgIndNext] = generateNextImgList(opts, l);
            vl_imreadjpeg(imgListNext, 'NumThreads', l.jpegData_param.prefetch_thread, 'Prefetch');
        else
            cache = cell(1,s);
            for i = 1:s
                cache{i} = imread( imgList{i} );
            end
        end

        for i = 1:s
            img = cache{i};

            % channel manipulation
            if l.jpegData_param.is_color
                if size(img, 3) == 1
                    img = cat(3, img,img,img);
                end
            else
                if size(img, 3) == 3
                    img = rgb2gray(img);
                end
            end

            %resize
            if numel(l.jpegData_param.resize) == 1
                img = imresize(img, l.jpegData_param.resize/min(size(img,1),size(img,2)), 'bilinear');
            elseif numel(l.jpegData_param.resize) == 2
                img = imresize(img, l.jpegData_param.resize, 'bilinear');
            end
            imgSize = size(img);

            %crop
            if numel(l.jpegData_param.crop) == 1
                if ~l.jpegData_param.crop_center
                    minB = min(imgSize(1), imgSize(2));
                    Hs = randi(minB-l.jpegData_param.crop+1)-1;
                    Ws = randi(minB-l.jpegData_param.crop+1)-1;
                    Hs = Hs+floor((imgSize(1)-minB)/2)+1;
                    Ws = Ws+floor((imgSize(2)-minB)/2)+1;
                else
                    Hs = floor((imgSize(1)-l.jpegData_param.crop)/2)+1;
                    Ws = floor((imgSize(2)-l.jpegData_param.crop)/2)+1;
                end
                img = img(Hs:(Hs+l.jpegData_param.crop-1), Ws:(Ws+l.jpegData_param.crop-1), :);
            elseif numel(l.jpegData_param.crop) == 2
                if ~l.jpegData_param.crop_center
                    Hs = randi(imgSize(1)-l.jpegData_param.crop(1)+1);
                    Ws = randi(imgSize(2)-l.jpegData_param.crop(2)+1);
                else
                    Hs = floor((imgSize(1)-l.jpegData_param.crop(1))/2)+1;
                    Ws = floor((imgSize(2)-l.jpegData_param.crop(2))/2)+1;
                end
                img = img(Hs:(Hs+l.jpegData_param.crop(1)-1), Ws:(Ws+l.jpegData_param.crop(2)-1), :);
            end

            %flip
            if l.jpegData_param.flip
                if rand() > 0.5
                    cache{i} = fliplr(img);
                else
                    cache{i} = img;
                end
            else
                cache{i} = img;
            end
            

        end

        if opts.gpuMode
            imgs = gpuArray(single(cat(4,cache{:})));
            top{2} = gpuArray(single(reshape(imageLabel(imgInd),1,1,1,[])));
        else
            imgs = single(cat(4,cache{:}));
            top{2} = single(reshape(imageLabel(imgInd),1,1,1,[]));
        end
        if numel(l.jpegData_param.mean)==size(imgs,3)
            imgs = bsxfun(@minus, imgs, reshape(single(l.jpegData_param.mean),1,1,3,1));
        else
            imgs = imgs-single(l.jpegData_param.mean);
        end
        top{1} = imgs(:,:,[3,2,1],:); %BGR


        cache = {};
        if l.jpegData_param.prefetch
            imgList = imgListNext;
            imgListNext = {};
            imgInd = imgIndNext;
            imgIndNext = [];
        else
            imgList = {};
            imgInd = [];
        end
    end

    function [imgList, imgInd] = generateNextImgList(opts, l)
        bs = l.jpegData_param.batch_size;
        if pointer == 0
            batchIndices = zeros(1, ceil(numel(imagePaths)/bs)*bs);
            if l.jpegData_param.shuffle
                batchIndices(1:numel(imagePaths)) = randperm(numel(imagePaths));
            else
                batchIndices(1:numel(imagePaths)) = 1:numel(imagePaths);
            end
            batchIndices = reshape(batchIndices, bs, []);
            pointer = 1;
        end
        imgInd = batchIndices(:, pointer);
        imgInd = imgInd(imgInd~=0);% last batch may have 0 indices
        s = numel(imgInd);

        imgList = cell(1,s);
        for i = 1:s
            imgList{i} = [l.jpegData_param.root_folder, filesep, imagePaths{imgInd(i)}];
        end


        % end process
        if pointer == size(batchIndices,2)
            pointer = 0;
        else
            pointer = pointer+1;
        end
    end


    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff)
        bottom_diff = {};
    end

end
