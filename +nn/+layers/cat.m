function o = cat(networkParameter)
%CAT Concatenate N blobs into one blob

o.name         = 'Cat';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;

default_cat_param = {
        'dim'   3 ...  % HWCN = 1234
    'indices'   {} % empty for normal concatenate operation, each value must be non-identical number
                   % indices specifies each bottom's dim concate to which indices of the whole top
};

    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update
        if isfield(l, 'cat_param')
            wp = nn.utils.vararginHelper(default_cat_param, l.cat_param);
        else
            wp = nn.utils.vararginHelper(default_cat_param, default_cat_param);
        end


        resource = {};

        assert(numel(l.bottom)>=1);
        assert(numel(l.top)==1);
        assert(numel(wp.dim) == 1 && wp.dim >= 1 && wp.dim <= 4);
        if ~isempty(wp.indices)
            assert(numel(wp.indices)==numel(l.bottom));
            assert(all(unique([wp.indices{:}])>0));
            sumSize = sum(cell2mat(bottomSizes'),1);
            assert(numel(unique([wp.indices{:}]))==sumSize(3));
        end

        topSizes = [1, 1, 1, 1];
        topSizes(1:numel(bottomSizes{1})) = bottomSizes{1}; % prevent matlab singleton dimension error
        otherDims = setdiff(1:4, wp.dim);
        for i=2:numel(bottomSizes)
            tmpSize = [1, 1, 1, 1];
            tmpSize(1:numel(bottomSizes{i})) = bottomSizes{i};
            if isequal(tmpSize(otherDims),  topSizes(otherDims))
                topSizes(wp.dim) = topSizes(wp.dim) + tmpSize(wp.dim);
            else
                error('Dimension mismatch.');
            end
        end

        topSizes = {topSizes};

        %return updated param
        param.cat_param = wp;
    end


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        if isempty(l.cat_param.indices)
            top   = {cat(3, bottom{:})};
        else
            D = numel(unique([l.cat_param.indices{:}]));
            dims = [1,1,1,1];
            dims0 = size(bottom{1});
            dims(1:numel(dims0)) = dims0;
            dims(l.cat_param.dim) = D;
            if opts.gpuMode
                top = {gpuArray.zeros(dims, 'single')};
            else
                top = {zeros(dims, 'single')};
            end
            switch l.cat_param.dim
                case 1
                    for i=1:numel(l.cat_param.indices)
                        top{1}(l.cat_param.indices{i},:,:,:) = bottom{i};
                    end
                case 2
                    for i=1:numel(l.cat_param.indices)
                        top{1}(:,l.cat_param.indices{i},:,:) = bottom{i};
                    end
                case 3
                    for i=1:numel(l.cat_param.indices)
                        top{1}(:,:,l.cat_param.indices{i},:) = bottom{i};
                    end
                case 4
                    for i=1:numel(l.cat_param.indices)
                        top{1}(:,:,:,l.cat_param.indices{i}) = bottom{i};
                    end
                otherwise
                    error('dim must be 1~4');
            end
        end
    end


    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff)
        bottom_diff = cell(1, numel(bottom));

        if isempty(l.cat_param.indices)
            sizeofbtm = cellfun(@(x) size(x, l.cat_param.dim), bottom);
            cumuSize = cumsum(sizeofbtm); %[5,3,6] => [5,8,14]
            cumuSize = [0, cumuSize];
            switch l.cat_param.dim
                case 1
                    for i=1:numel(bottom)
                        bottom_diff{i} = top_diff{1}((cumuSize(i)+1):cumuSize(i+1),:,:,:);
                    end
                case 2
                    for i=1:numel(bottom)
                        bottom_diff{i} = top_diff{1}(:,(cumuSize(i)+1):cumuSize(i+1),:,:);
                    end
                case 3
                    for i=1:numel(bottom)
                        bottom_diff{i} = top_diff{1}(:,:,(cumuSize(i)+1):cumuSize(i+1),:);
                    end
                case 4
                    for i=1:numel(bottom)
                        bottom_diff{i} = top_diff{1}(:,:,:,(cumuSize(i)+1):cumuSize(i+1));
                    end
                otherwise
                    error('dim must be 1~4');
            end
        else
            switch l.cat_param.dim
                case 1
                    for i=1:numel(bottom)
                        bottom_diff{i} = top_diff{1}(l.cat_param.indices{i},:,:,:);
                    end
                case 2
                    for i=1:numel(bottom)
                        bottom_diff{i} = top_diff{1}(:,l.cat_param.indices{i},:,:);
                    end
                case 3
                    for i=1:numel(bottom)
                        bottom_diff{i} = top_diff{1}(:,:,l.cat_param.indices{i},:);
                    end
                case 4
                    for i=1:numel(bottom)
                        bottom_diff{i} = top_diff{1}(:,:,:,l.cat_param.indices{i});
                    end
                otherwise
                    error('dim must be 1~4');
            end
        end
        

    end

end