function o = contrastiveLoss()
%LOGISTICLOSS 
%
% NOTICE
%   label index starts from 0 (compatible with other NN tools)
%   you can specify begining index from parameter

o.name         = 'contrastiveLoss';
o.generateLoss = true;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;


default_contrastiveLoss_param = {
    'labelIndex_start' single(0)     ...
           'threshold' single(1e-20) ...
              'margin' single(1)     ...
};

% Save Forward result for faster computation
d_ = 0;
d = 0;

    function [resource, topSizes, param] = setup(l, bottomSizes)
        resource = {};

        if isfield(l, 'contrastiveLoss_param')
            wp = vllab.utils.vararginHelper(default_contrastiveLoss_param, l.contrastiveLoss_param);
        else
            wp = vllab.utils.vararginHelper(default_contrastiveLoss_param, default_contrastiveLoss_param);
        end
        param.contrastiveLoss_param = wp;

        assert(numel(l.bottom)==3);
        assert(numel(l.top)==1);
        
        resSize1 = bottomSizes{1};
        resSize2 = bottomSizes{1};
        ansSize = bottomSizes{3};

        assert(isequal(resSize1, resSize2));
        assert(isequal(resSize1(1:2), resSize2(1:2)) && isequal(resSize1(1:2), [1 1]));
        assert(isequal(ansSize(1:3), [1 1 1]));
        assert(isequal(ansSize(4), resSize1(4)));

        topSizes = {[1, 1, 1, 1]};

    end
    function [outputBlob, weightUpdate] = forward(opts, l, weights, blob)
        weightUpdate = {};
        d_ = blob{1}-blob{2};
        d = sum((d_).^2, 3);
        y = blob{3};
        E = 0.5 * sum(  y.*d + (1-y).*max(l.contrastiveLoss_param.margin - d, single(0))  );%/size(blob{1},4);
        outputBlob = {E};

    end
    function [outputdzdx, outputdzdw] = backward(opts, l, weights, blob, dzdy)
        outputdzdw = {};
        outputdzdx = cell(1,3);
        m_d = l.contrastiveLoss_param.margin - d;
        rightTerm = d_;
        rightTerm(:,:,:,m_d(:)<=0) = 0;
        y = blob{3};
        outputdzdx{1} = dzdy{1} * (bsxfun(@times, d_, y) - bsxfun(@times, rightTerm, 1-y));% / size(blob{1}, 4);
        outputdzdx{2} = -outputdzdx{1};
    end
end