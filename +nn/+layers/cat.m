function o = cat(varargin)
%CAT Concatenate N blobs into one blob

o.name         = 'Cat';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;

default_cat_param = {
    'dim'   3  % HWCN = 1234
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

        assert(numel(l.bottom)>=0);
        assert(numel(l.top)==1);
        assert(numel(wp.dim) == 1 && wp.dim >= 1 && wp.dim <= 4);

        topSizes = [1, 1, 1, 1];
        topSizes(1:numel(size(l.bottom{1}))) = size(l.bottom{1}); % prevent matlab singleton dimension error
        otherDims = setdiff(1:4, wp.dim);
        for i=2:numel(l.bottom)
            tmpSize = [1, 1, 1, 1];
            tmpSize(1:numel(size(l.bottom{i}))) = size(l.bottom{i});
            if isequal(tmpSize(otherDims),  topSizes(otherDims))
                topSizes(wp.dim) = topSizes(wp.dim) + tmpSize(wp.dim);
            else
                error('Dimension mismatch.');
            end
        end

        topSizes = {topSizes};

        %return updated param
        param = wp;
    end


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        top   = {cat(3, bottom{:})};
    end


    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        bottom_diff = cell(1, numel(bottom));
        switch l.cat_param.dim
            case 1
                for i=1:numel(bottom)
                    bottom_diff{i} = top_diff{1}(i,:,:,:);
                end
            case 2
                for i=1:numel(bottom)
                    bottom_diff{i} = top_diff{1}(:,i,:,:);
                end
            case 3
                for i=1:numel(bottom)
                    bottom_diff{i} = top_diff{1}(:,:,i,:);
                end
            case 4
                for i=1:numel(bottom)
                    bottom_diff{i} = top_diff{1}(:,:,:,i);
                end
            otherwise
                for i=1:numel(bottom)
                    bottom_diff{i} = top_diff{1}(:,:,i,:);
                end
        end

    end

end