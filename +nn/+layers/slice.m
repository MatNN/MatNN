function o = slice(varargin)
%SLICE Slice a blob into many blobs
%  NOTICE
%  This layer can also extract desired portion of a blob,
%  not just slice it
%
%  EXAMPLE
%  no.newLayer({
%      'type'   'nn.layers.slice' ...
%      'name'   'sliceIt'         ...
%      'bottom' 'data'            ...
%      'top'    {'sliceRes1', 'sliceRes2', 'sliceRes3', 'sliceRes4'} ...
%      'slice_param' {
%          'dim' 3 ...
%          'indices' {1:3, 4:6, 7:9, 10}
%          }
%      });

o.name         = 'Slice';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;

default_slice_param = {
        'dim' 3 ...  % HWCN = 1234
    'indices' {}
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
        assert(numel(l.top)==numel(wp.indices));


        topSizes = cell(1, K);
        d1 = bottomSizes{1}(1);
        d2 = bottomSizes{1}(2);
        d3 = bottomSizes{1}(3);
        d4 = bottomSizes{1}(4);
        switch wp.dim
            case 1
                for i=1:K
                    topSizes{K} = [numel(wp.indices{i}),d2,d3,d4];
                end
            case 2
                for i=1:K
                    topSizes{K} = [d1,numel(wp.indices{i}),d3,d4];
                end
            case 3
                for i=1:K
                    topSizes{K} = [d1,d2,numel(wp.indices{i}),d4];
                end
            case 4
                for i=1:K
                    topSizes{K} = [d1,d2,d3, numel(wp.indices{i})];
                end
            otherwise
                for i=1:K
                    topSizes{K} = [d1,d2,numel(wp.indices{i}),d4];
                end
        end

        %return updated param
        param.cat_param = wp;
    end


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        K = numel(l.cat_param.indices);
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
        K = numel(l.cat_param.indices);
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