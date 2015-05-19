function o = euclideanLoss(architecture)
%EUCLIDEANLOSS
%
% NOTICE
%   label index starts from 0 (compatible with other NN tools)
%   you can specify begining index from parameter

if nargin == 0
    architecture = 'default';
end

o.name         = 'EuclideanLoss';
o.generateLoss = true;

% process architecture
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;

default_euclideanLoss_param = {
    'labelIndex_start' single(0)     ...
           'threshold' single(1)     ... % minimal area size = 1
    'per_channel_area' false         ... % this will divide each channel's distance with ground truth's area
};

d_ = [];
d2  = [];
areas = [];

    function [resource, topSizes, param] = setup(l, bottomSizes)
        resource = {};

        if isfield(l, 'euclideanLoss_param')
            wp = nn.utils.vararginHelper(default_euclideanLoss_param, l.euclideanLoss_param);
        else
            wp = nn.utils.vararginHelper(default_euclideanLoss_param, default_euclideanLoss_param);
        end
        param.euclideanLoss_param = wp;

        assert(numel(l.bottom)==2);
        assert(numel(l.top)==1);
        
        resSize1 = bottomSizes{1};
        ansSize = bottomSizes{2};

        assert( isequal(ansSize, resSize1) | (isequal(ansSize([1,2,4]), resSize1([1,2,4])) && ansSize(3)==1) );
        % Label size must be HxWxCxN or HxWx1xN

        topSizes = {[1, 1, 1, 1]};

    end


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        if size(bottom{2},3) == 1 && size(bottom{1},3) > 1 % if label = HxWx1xN
            d_ = bottom{1};
            for i = 1:size(bottom{1},3)
                d_(:,:,i,:) = bsxfun(@minus, d_(:,:,i,:),  bottom{2}.*(bottom{2}==i) );
            end
            d2 = d_.^2;
            if l.euclideanLoss_param.per_channel_area
                labels = sort(unique(bottom{2}));
                labels(labels < labelIndex_start) = [];
                areas  = arrayfun( @(x) sum(sum(bottom{2}==x,1),2), labels, 'UniformOutput', false ); %=cell array
                areas  = cat(3, areas{:});

                top{1} = 0.5 * sum(bsxfun(@rdivide, d2, areas));%/size(bottom{1},4);
            else
                top{1} = 0.5 * sum(d2(:));%/size(bottom{1},4);
            end
        else % if label = HxWxCxN
            d_ = bottom{1}-bottom{2};
            d2 = d_.^2;
            if l.euclideanLoss_param.per_channel_area
                areas  = single(min( sum(sum(bottom{2},1),2) , l.euclideanLoss_param.threshold) ); %dim=1x1xCxN

                top{1} = 0.5 * sum(bsxfun(@rdivide, d2, areas));%/size(bottom{1},4);
            else
                top{1} = 0.5 * sum(d2(:));%/size(bottom{1},4);
            end
        end
        
    end


    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        if l.euclideanLoss_param.per_channel_area
            bottom_diff{1} = top_diff .* bsxfun(@rdivide, d_, areas);%/size(bottom{1},4);
            bottom_diff{2} = -top_diff .* bsxfun(@rdivide, d_, areas);%/size(bottom{1},4);
        else
            bottom_diff{1} = top_diff .* d_;%/size(bottom{1},4);
            bottom_diff{2} = -top_diff .* d_;%/size(bottom{1},4);
        end
        
    end
end