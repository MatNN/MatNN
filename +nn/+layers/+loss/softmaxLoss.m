function o = softmaxLoss(networkParameter)
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
            'threshold' single(realmin('single')) ...
    'ForceEliminateInf' false ... % CAUTIONS! Don't set to true in most cases, unless you are really sure other workaround is uesless
           'accumulate' true  ... % report per-batch loss (false) or avg loss (true), this does not affect backpropagation
          'loss_weight' 1 ... % a multiplier to the loss
};

% Save Forward result for faster computation
ind        = [];
N          = [];
ll         = [];
accumulateN = 0;
accumulateL = 0;

    function [resource, topSizes, param] = setup(l, bottomSizes)
        resource = {};

        if isfield(l, 'softmaxLoss_param')
            wp = nn.utils.vararginHelper(default_softmaxLoss_param, l.softmaxLoss_param);
        else
            wp = nn.utils.vararginHelper(default_softmaxLoss_param, default_softmaxLoss_param);
        end
        param.softmaxLoss_param = wp;


        assert(numel(l.bottom)>=2 && numel(l.bottom)<=3); % for label_weight
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
        ll = label >= l.softmaxLoss_param.labelIndex_start;
        N = resSize(1)*resSize(2)*resSize(4);


        % if opts.gpuMode
        %     ind = gpuArray.false(resSize);
        % else
        %     ind = false(resSize);
        % end
        ind = (1:numel(label))' -1;
        ind = ind(ll(:));
        ll  = ind+1;
        label = label(:);
        label = label(ll)-l.softmaxLoss_param.labelIndex_start;
        ind = mod(ind, resSize(1)*resSize(2)) + ...
              label*resSize(1)*resSize(2) + ...
              floor(ind/(resSize(1)*resSize(2)))*resSize(1)*resSize(2)*resSize(3) + ...
              1;

        y = Exp(resultBlob, 3);
        y = bsxfun(@rdivide, y, sum(y,3));
        y = y(ind);

        if l.softmaxLoss_param.accumulate
            if opts.currentIter == 1
                accumulateL = 0;
                accumulateN = 0;
            end
            % for label_weights
            if numel(l.bottom)==3
                accumulateL = accumulateL - sum(bottom{3}(ind) .* log(max(y,l.softmaxLoss_param.threshold)))/(resSize(1)*resSize(2));
            else
                accumulateL = accumulateL - sum(log(max(y,l.softmaxLoss_param.threshold)))/(resSize(1)*resSize(2));
            end
            
            accumulateN = accumulateN + resSize(4);
            top{1} = accumulateL/accumulateN;
        else
            % for label_weights
            if numel(l.bottom)==3
                top{1} = l.softmaxLoss_param.loss_weight* (-sum( bottom{3}(ind) .* log(max(y,l.softmaxLoss_param.threshold)))/N);
            else
                top{1} = l.softmaxLoss_param.loss_weight* (-sum(log(max(y,l.softmaxLoss_param.threshold)))/N);
            end
            
        end
        
        %top{1} = vl_nnloss(bottom{1},bottom{2},[],'loss', 'softmaxlog');
        
    end
    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff)
        %compute derivative
        y = Exp(bottom{1}, 3);
        y = bsxfun(@rdivide, y, max(sum(y,3), l.softmaxLoss_param.threshold));
        y(ind)  = y(ind)-1;
        bottom_diff = { bottom{3}.* l.softmaxLoss_param.loss_weight * (y.* top_diff{1})/N , []};
        ind        = [];
        N          = [];
        ll         = [];
        accumuateN = 0;
        accumuateL = 0;
        %bottom_diff{1} = vl_nnloss(bottom{1}, bottom{2},top_diff{1},'loss', 'softmaxlog');
        %bottom_diff{2} = [];
    end
end

function [y] = Exp(X, dim)
    M = max(X, [], dim);
    y = exp( bsxfun(@minus, X, M) );
end