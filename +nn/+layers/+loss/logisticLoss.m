function o = logisticLoss(varargin)
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
    'labelIndex_start' single(0)     ...
           'threshold' single(1e-20) ...
};

% Save Forward result for faster computation
resultBlob = [];
ind        = [];
N          = [];

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
            if ~(isequal(resSize([1,2,4]), ansSize([1,2,4]) && ansSize(3)) == 1) && ~(isequal(resSize(4), ansSize(4)) && isequal(ansSize(1:3),[1 1 1]))
                error('Label size must be Nx1, 1xN or HxWx1xN.');
            end
        end
        topSizes = {[1, 1, 1, 1]};

    end
    function [outputBlob, weights] = forward(opts, l, weights, blob)

        resultBlob = min(blob{1}, l.logisticLoss_param.threshold);
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
        label = label - l.logisticLoss_param.labelIndex_start;
        N = resSize(1)*resSize(2);
        ind = 0:(numel(label) - 1);
        ind = 1 + mod(ind, N)     ...
                + N * label(:)' ...
                + N*resSize(3) * floor(ind/(resSize(1)*resSize(2)));

        outputBlob = {- sum(log(resultBlob(ind)))/N };

    end
    function [mydzdx, mydzdw] = backward(opts, l, weights, blob, dzdy, mydzdw, mydzdwCumu)
        dzdx = -(1./resultBlob) * (dzdy{1}/N);

        % only ground truth label are correct, set others to zero
        outdzdx = dzdx*0; % faster than zeros(size(dzdx)); ?
        outdzdx(ind) = dzdx(ind);
        mydzdx = {outdzdx,[]};

    end
end