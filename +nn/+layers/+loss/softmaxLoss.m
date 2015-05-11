function o = softmaxLoss()
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
     'labelIndex_start' single(0)     ...
            'threshold' single(1e-20) ...
    'ForceEliminateInf' false ... % CAUTIONS! Don't set to true in most cases, unless you are really sure other workaround is uesless
};

% Save Forward result for faster computation
resultBlob = [];
ind        = [];
N          = [];

    function [resource, topSizes, param] = setup(l, bottomSizes)
        resource = {};

        if isfield(l, 'softmaxLoss_param')
            wp = vllab.utils.vararginHelper(default_softmaxLoss_param, l.softmaxLoss_param);
        else
            wp = vllab.utils.vararginHelper(default_softmaxLoss_param, default_softmaxLoss_param);
        end
        param.softmaxLoss_param = wp;


        assert(numel(l.bottom)==2);
        assert(numel(l.top)==1);
        

        resSize = bottomSizes{1};
        ansSize = bottomSizes{2};
        if ~isequal(resSize(4),prod(ansSize))
            if ~(isequal(resSize([1,2,4]), ansSize([1,2,4]) && ansSize(3)) == 1) && ~(isequal(resSize(4), ansSize(4)) && isequal(ansSize(1:3),[1 1 1]))
                error('Label size must be Nx1, 1xN or HxWx1xN.');
            end
        end
        topSizes = {[1, 1, 1, 1]};

    end
    function [outputBlob, weightUpdate] = forward(opts, l, weights, blob)
        weightUpdate = {};
        %{-
        resultBlob = max(blob{1}, l.softmaxLoss_param.threshold);
        resSize = size(resultBlob);
        labelSize = size(blob{2});
        if resSize(4) == numel(blob{2})
            label = reshape(blob{2}, [1, 1, 1 resSize(4)]) ;
            label = repmat(label, [resSize(1), resSize(2)]) ;
        else
            if ~isequal(resSize([1,2,4]), labelSize([1,2,4]))
                error('Label size must be Nx1, 1xN or HxWx1xN.');
            else
                label = blob{2};
            end
        end
        label = label - l.softmaxLoss_param.labelIndex_start;
        N = resSize(1)*resSize(2);
        ind = 0:(numel(label) - 1);
        ind = 1 + mod(ind, N)     ...
                + N * label(:)' ...
                + N*resSize(3) * floor(ind/N);

        %compute logsumexp
        if l.softmaxLoss_param.ForceEliminateInf
            y = LogSumExp_noInf(resultBlob, 3);
        else
            y = LogSumExp(resultBlob, 3);
        end
        outputBlob = { sum( y(:)'-resultBlob(ind) )/N };
        %}
        %{
        y=vl_nnsoftmaxloss(blob{1},blob{2}+1);
        outputBlob = { y };
        %}
    end
    function [outputdzdx, outputdzdw] = backward(opts, l, weights, blob, dzdy)
        outputdzdw = {};

        %{-
        %compute derivative
        y = Exp(resultBlob, 3);
        y = bsxfun(@rdivide, y, sum(y,3));
        y(ind) = y(ind)-1;
        outputdzdx = { y*dzdy{1}/N , []};
        %}
        %{
        y=vl_nnsoftmaxloss(blob{1},blob{2}+1, dzdy{1});
        outputdzdx = { y,[] };
        %}
    end
end

function y = LogSumExp_noInf(X, dim)
    M = max(X, [], dim);
    m = min(X, [], dim);
    y = M + log(sum(exp( bsxfun(@minus, X, M) ),dim));
    ind = isinf(y);
    ind_po = ind & y > 0;
    ind_ne = ind & y < 0;
    y(ind_po) = M(ind_po);
    y(ind_ne) = m(ind_ne);
end

function y = LogSumExp(X, dim)
    M = max(X, [], dim);
    y = M + log(sum(exp( bsxfun(@minus, X, M) ), dim));
end
function [y] = Exp(X, dim)
    M = max(X, [], dim);
    y = exp( bsxfun(@minus, X, M) );
end