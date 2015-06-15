function o = slice(varargin)
%SLICE Slice a blob into many blobs

o.name         = 'Slice';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;

default_slice_param = {
        'dim' 3 ...  % HWCN = 1234
    'indices' []
};

    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update
        if isfield(l, 'slice_param')
            wp = nn.utils.vararginHelper(default_slice_param, l.slice_param);
        else
            wp = nn.utils.vararginHelper(default_slice_param, default_slice_param);
        end


        resource = {};

        K = numel(wp.indices);

        assert(numel(l.bottom)==1);
        assert(numel(l.top)==K);


        topSizes = cell(1, K);
        d1 = size(bottom{1},1);
        d2 = size(bottom{1},2);
        d3 = size(bottom{1},3);
        d4 = size(bottom{1},4);
        switch wp.dim
            case 1
                for i=1:K
                    topSizes{K} = [wp.indices{i},d2,d3,d4];
                end
            case 2
                for i=1:K
                    topSizes{K} = [d1,wp.indices{i},d3,d4];
                end
            case 3
                for i=1:K
                    topSizes{K} = [d1,d2,wp.indices{i},d4];
                end
            case 4
                for i=1:K
                    topSizes{K} = [d1,d2,d3, wp.indices{i}];
                end
            otherwise
                for i=1:K
                    topSizes{K} = [d1,d2,wp.indices{i},d4];
                end
        end

        %return updated param
        param.cat_param = wp;
    end


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        K = size(l.cat_param.indices);
        switch l.cat_param.dim
            case 1
                for i=1:K
                    top{K} = bottom{1}(l.cat_param.indices{i},:,:,:);
                end
            case 2
                for i=1:K
                    top{K} = bottom{1}(:,l.cat_param.indices{i},:,:);
                end
            case 3
                for i=1:K
                    top{K} = bottom{1}(:,:,l.cat_param.indices{i},:);
                end
            case 4
                for i=1:K
                    top{K} = bottom{1}(:,:,:,l.cat_param.indices{i});
                end
            otherwise
                for i=1:K
                    top{K} = bottom{1}(:,:,l.cat_param.indices{i},:);
                end
        end
    end


    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        bottom_diff = {bottom{1}.*0}; %works for normal array and gpuArray
        K = size(l.cat_param.indices);
        switch l.cat_param.dim
            case 1
                for i=1:K
                    bottom_diff{1}(l.cat_param.indices{i},:,:,:) = top_diff{i};
                end
            case 2
                for i=1:K
                    bottom_diff{1}(:,l.cat_param.indices{i},:,:) = top_diff{i};
                end
            case 3
                for i=1:K
                    bottom_diff{1}(:,:,l.cat_param.indices{i},:) = top_diff{i};
                end
            case 4
                for i=1:K
                    bottom_diff{1}(:,:,:,l.cat_param.indices{i}) = top_diff{i};
                end
            otherwise
                for i=1:K
                    bottom_diff{1}(:,:,l.cat_param.indices{i},:) = top_diff{i};
                end
        end
    end

end