function o = logisticLoss(networkParameter)
%LOGISTICLOSS 
%
% NOTICE
%   label index starts from 0 (compatible with other NN tools)
%   you can specify begining index from parameter

o.name         = 'LogisticLoss';
o.generateLoss = true;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;


default_logisticLoss_param = {
    'labelIndex_start' single(0)    ...
           'threshold' single(1e-4)
};

% Save Forward result for faster computation
resultBlob = [];
ind        = [];
N          = [];
ll         = [];

    function [resource, topSizes, param] = setup(l, bottomSizes)
        resource = {};

        if isfield(l, 'logisticLoss_param')
            wp = nn.utils.vararginHelper(default_logisticLoss_param, l.logisticLoss_param);
        else
            wp = nn.utils.vararginHelper(default_logisticLoss_param, default_logisticLoss_param);
        end
        param.logisticLoss_param = wp;


        assert(numel(l.bottom)==2);
        assert(numel(l.top)==1);
        

        resSize = bottomSizes{1};
        ansSize = bottomSizes{2};
        if ~isequal(resSize(4),prod(ansSize))
            if ~(isequal(resSize([1,2,4]), ansSize([1,2,4])) && ansSize(3) == 1) && ~(isequal(resSize(4), ansSize(4)) && isequal(ansSize(1:3),[1 1 1]))
                error('Label size must be Nx1, 1xN or HxWx1xN.');
            end
        end
        topSizes = {[1, 1, 1, 1]};

    end
    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        %resultBlob = max(bottom{1}, l.logisticLoss_param.threshold);
        resultBlob = bottom{1}+l.logisticLoss_param.threshold;

        resSize = nn.utils.size4D(resultBlob);
        labelSize = nn.utils.size4D(bottom{2});

        if resSize(4) == numel(bottom{2})
            label = reshape(bottom{2}, [1, 1, 1 resSize(4)]) ;
            label = repmat(label, [resSize(1), resSize(2)]) ;
        else
            if ~isequal(resSize([1,2,4]), labelSize([1,2,4]))
                error('Label size must be Nx1, 1xN or HxWx1xN.');
            else
                label = bottom{2};
            end
        end
        ll = label >= l.logisticLoss_param.labelIndex_start;
        %label = label(ll) - l.logisticLoss_param.labelIndex_start;
        N = resSize(1)*resSize(2)*resSize(4);
        %ind = find(ll)-1;
        %ind = 1 + mod(ind, N)  ...
        %        + N * label(:) ...
        %        + N*resSize(3) * floor(ind/N);

        if opts.gpuMode
            ind = gpuArray.false(resSize);
        else
            ind = false(resSize);
        end
        for i=1:resSize(3)
            ind(:,:,i,:) = label == i + (l.logisticLoss_param.labelIndex_start - 1);
        end

        top{1} = -sum(log(resultBlob(ind)))/N;

    end
    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff)
        dzdx = -bsxfun(@rdivide, bsxfun(@times, ll, top_diff{1}/N), resultBlob);

        % only ground truth label are correct, set others to zero
        outdzdx = dzdx*0; % faster than zeros(size(dzdx)); ?
        outdzdx(ind) = dzdx(ind);
        bottom_diff = {outdzdx,[]};

    end
end