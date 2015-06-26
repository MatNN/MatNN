function o = softmaxLoss(varargin)
%SOFTMAXLOSS 
%
% NOTICE
%   label index starts from 0 (compatible with other NN tools)
%   you can specify begining index from parameter

o.name         = 'SoftmaxLoss';
o.generateLoss = true;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;


default_softmaxLoss_param = {
     'labelIndex_start' single(0)    ...
            'threshold' single(1e-4) ...
    'ForceEliminateInf' false ... % CAUTIONS! Don't set to true in most cases, unless you are really sure other workaround is uesless
};

% Save Forward result for faster computation
ind        = [];
N          = [];

    function [resource, topSizes, param] = setup(l, bottomSizes)
        resource = {};

        if isfield(l, 'softmaxLoss_param')
            wp = nn.utils.vararginHelper(default_softmaxLoss_param, l.softmaxLoss_param);
        else
            wp = nn.utils.vararginHelper(default_softmaxLoss_param, default_softmaxLoss_param);
        end
        param.softmaxLoss_param = wp;


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
        %resultBlob = max(bottom{1}, l.softmaxLoss_param.threshold);
        resultBlob = bottom{1};
        resSize = size(resultBlob);
        resSize(4) = size(resultBlob,4);
        labelSize = size(bottom{2});
        labelSize(4) = size(bottom{2},4);
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
        ll = label >= l.softmaxLoss_param.labelIndex_start;
        label = label(ll) - l.softmaxLoss_param.labelIndex_start;
        N = resSize(1)*resSize(2);
        ind = find(ll)-1;%0:numel(label)-1
        ind = 1 + mod(ind, N)  ...
                + N * label(:) ...
                + N*resSize(3) * floor(ind/N);

        %compute logsumexp
        if l.softmaxLoss_param.ForceEliminateInf
            y = LogSumExp_noInf(resultBlob, 3, l.softmaxLoss_param.threshold);
        else
            y = LogSumExp(resultBlob, 3, l.softmaxLoss_param.threshold);
        end
        y = y(ll);
        top{1} = sum( y(:)-resultBlob(ind) )/N;
    end
    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        %compute derivative
        y = Exp(bottom{1}, 3)+l.softmaxLoss_param.threshold;
        y = bsxfun(@rdivide, y, sum(y,3));
        y(ind) = y(ind)-1;
        bottom_diff = { y*top_diff{1}/N , []};
    end
end

function y = LogSumExp_noInf(X, dim, thres)
    M = max(X, [], dim);
    m = min(X, [], dim);
    y = M + log(sum(exp( bsxfun(@minus, X, M) ), dim)+thres);
    ind = isinf(y);
    ind_po = ind & y > 0;
    ind_ne = ind & y < 0;
    y(ind_po) = M(ind_po);
    y(ind_ne) = m(ind_ne);
end

function y = LogSumExp(X, dim, thres)
    M = max(X, [], dim);
    y = M + log(sum(exp( bsxfun(@minus, X, M) ), dim)+thres);
end
function [y] = Exp(X, dim)
    M = max(X, [], dim);
    y = exp( bsxfun(@minus, X, M) );
end