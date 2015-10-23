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
            'threshold' realmin('single') ...
           'accumulate' true  ... % report per-batch loss (false) or avg loss (true), this does not affect backpropagation
          'loss_weight' 1 ... % a multiplier to the loss
};

% Save Forward result for faster computation
ind         = [];
N           = [];
ll          = [];
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


        if numel(networkParameter.gpus)>0
            accumulateN = gpuArray.zeros(1,1,'single');
            accumulateL = gpuArray.zeros(1,1,'single');
        end

    end

    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)

        % Reshape label
        resultBlob = bottom{1};
        resSize    = nn.utils.size4D(resultBlob);
        labelSize  = nn.utils.size4D(bottom{2});
        if resSize(4) == numel(bottom{2}) || resSize(1) == numel(bottom{2})
            label = reshape(bottom{2}, [1, 1, 1 resSize(4)]);
            label = repmat(label, [resSize(1), resSize(2)]);
        else
            if ~isequal(resSize([1,2,4]), labelSize([1,2,4]))
                error('Label size must be Nx1, 1xN or HxWx1xN.');
            else
                label = bottom{2};
            end
        end

        % Calc correspond indices
        ll    = label >= l.softmaxLoss_param.labelIndex_start;
        N     = resSize(1)*resSize(2)*resSize(4);
        ind   = (1:numel(label))' -1;
        ind   = ind(ll(:));
        ll    = ind+1;
        label = label(:);
        label = label(ll)-l.softmaxLoss_param.labelIndex_start; % DO NOT ADD 1, we calc zero-based ind.
        ind   = mod(ind, resSize(1)*resSize(2)) + ...
                label*resSize(1)*resSize(2) + ...
                floor(ind/(resSize(1)*resSize(2)))*resSize(1)*resSize(2)*resSize(3) + ...
                1; % ADD 1 to match matlab 1-based ind

        % Do softmax
        y = Exp(resultBlob, 3);
        y = bsxfun(@rdivide, y, sum(y,3));
        y = y(ind);

        % Compute log-loss, and accumulate loss if needed
        if l.softmaxLoss_param.accumulate
            if opts.currentIter == 1
                accumulateL = accumulateL*0;
                accumulateN = accumulateN*0;
            end
            % for label_weights
            if numel(l.bottom)==3
                accumulateL = accumulateL - sum(bottom{3}(ll) .* log(max(y,l.softmaxLoss_param.threshold)))/(resSize(1)*resSize(2));
            else
                accumulateL = accumulateL - sum(log(max(y,l.softmaxLoss_param.threshold)))/(resSize(1)*resSize(2));
            end
            
            accumulateN = accumulateN + resSize(4);
            top{1} = accumulateL/accumulateN;
        else
            % for label_weights
            if numel(l.bottom)==3
                top{1} = l.softmaxLoss_param.loss_weight* (-sum( bottom{3}(ll) .* log(max(y,l.softmaxLoss_param.threshold)))/N);
            else
                top{1} = l.softmaxLoss_param.loss_weight* (-sum(log(max(y,l.softmaxLoss_param.threshold)))/N);
            end
            
        end
        
    end

    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff)
        y = Exp(bottom{1}, 3);
        y = bsxfun(@rdivide, y, max(sum(y,3), l.softmaxLoss_param.threshold));
        y(ind)  = y(ind)-1;
        if numel(l.bottom)==3
            bottom_diff = { bsxfun(@times, bottom{3}, l.softmaxLoss_param.loss_weight * (y.* top_diff{1})/N) , [], []};
        else
            bottom_diff = { l.softmaxLoss_param.loss_weight * (y.* top_diff{1})/N , []};
        end
    end

end

function [y] = Exp(X, dim)
    M = max(X, [], dim);
    y = exp( bsxfun(@minus, X, M) );
end