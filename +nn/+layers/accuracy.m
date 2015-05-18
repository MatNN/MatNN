function o = accuracy(varargin)
%ACCURACY 
%
% NOTICE
%   label index starts from 0 (compatible with other NN tools)
%   you can specify begining index from parameter

o.name         = 'Accuracy';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;

default_accuracy_param = {
     'labelIndex_start' single(0)     ...
};

    function [resource, topSizes, param] = setup(l, bottomSizes)
        resource = {};

        if isfield(l, 'accuracy_param')
            wp = nn.utils.vararginHelper(default_accuracy_param, l.accuracy_param);
        else
            wp = nn.utils.vararginHelper(default_accuracy_param, default_accuracy_param);
        end
        param.accuracy_param = wp;


        assert(numel(l.bottom)==2);
        assert(numel(l.top)==1);
        

        resSize = bottomSizes{1};
        ansSize = bottomSizes{2};

        if ~isequal(resSize(4),prod(ansSize))
            if ~(isequal(resSize([1,2,4]), ansSize([1,2,4]) && ansSize(3)) == 1) && ~(isequal(resSize(4), ansSize(4)) && isequal(ansSize(1:3),[1 1 1]))
                error('Label size must be Nx1, 1xN, 1x1x1xN or HxWx1xN.');
            end
        end

        topSizes = {[1, 1, 1, 1]};

    end
    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        % if size(prediction) == HxWxCxN, divide cumulative acc by HxWxN, size(label) == HxWx1xN
        % if size(prediction) == 1x1xCxN, divide cumulative acc by N, size(label) == 1x1x1xN
        % if size(prediction) == 1x1x1xN, divide cumulative acc by N, size(label) == 1x1x1xN
        label = bottom{2} - l.accuracy_param.labelIndex_start;
        [~, argMax] = max(bottom{1}, [], 3);
        k = (argMax -1 + l.accuracy_param.labelIndex_start) == label;
        top{1} = sum(k(:))/(size(bottom{1},1)*size(bottom{1},2));%*size(bottom{1},4)); don't divide N here, because train.m will do it for us

    end
    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        bottom_diff = {[],[]};
    end
end
