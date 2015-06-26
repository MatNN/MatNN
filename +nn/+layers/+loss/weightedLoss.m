function o = weightedLoss(varargin)
%weightedLoss bottoms must be losses

o.name         = 'weightedLoss';
o.generateLoss = true;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;


default_weightedLoss_param = {
    'coef' [] ...
};

    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update
        resource = {};

        if isfield(l, 'weightedLoss_param')
            wp = nn.utils.vararginHelper(default_weightedLoss_param, l.weightedLoss_param);
        else
            wp = nn.utils.vararginHelper(default_weightedLoss_param, default_weightedLoss_param);
        end


        assert(numel(l.bottom)==numel(wp.coef), 'Coefficient number must be the same as bottoms');
        assert(numel(l.top)==1);
        for i=2:numel(l.bottom)
            assert(numel(bottomSizes{i-1})==numel(bottomSizes{i}));
        end

        topSizes = {[1,1,1,1]};

        %return updated param
        param.weightedLoss_param = wp;
    end


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        top{1} = top{1}.*0;
        top{1} = sum(top{1}(:));
        for i=1:numel(bottom)
            top{1} = top{1} + l.weightedLoss_param.coef(i) .* bottom{i};
        end
    end


    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        for i=1:numel(bottom)
            bottom_diff{i} = l.weightedLoss_param.coef(i);
        end
    end


end