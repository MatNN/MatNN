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
    'labelIndex_start' single(0) ...
        'meanClassAcc' false ...
             'dataNum' [] ... % set this if only you set 'perClassAcc' to true
};
counting = 0;
perClassArea = [];
perClassAcc  = [];

    function [resource, topSizes, param] = setup(l, bottomSizes)
        resource = {};

        if isfield(l, 'accuracy_param')
            wp = nn.utils.vararginHelper(default_accuracy_param, l.accuracy_param);
        else
            wp = nn.utils.vararginHelper(default_accuracy_param, default_accuracy_param);
        end
        param.accuracy_param = wp;

        resSize = bottomSizes{1};
        ansSize = bottomSizes{2};

        assert(numel(l.bottom)==2);
        if wp.meanClassAcc
            assert(numel(l.top)==2, 'Accuracy layer will generate two outputs if you set ''meanClassAcc'' to true.');
            assert(~isempty(wp.dataNum), 'Accuracy layer needs total data number if you set ''meanClassAcc'' to true.')
            perClassArea = zeros(1, resSize(3), 'single');
            perClassAcc  = zeros(1, resSize(3), 'single');
        else
            assert(numel(l.top)==1);
        end

        if ~isequal(resSize(4),prod(ansSize))
            if ~(isequal(resSize([1,2,4]), ansSize([1,2,4])) && ansSize(3) == 1) && ~(isequal(resSize(4), ansSize(4)) && isequal(ansSize(1:3),[1 1 1]))
                error('Label size must be Nx1, 1xN, 1x1x1xN or HxWx1xN.');
            end
        end

        topSizes = {[1, 1, 1, 1]};

    end
    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        % if size(prediction) == HxWxCxN, divide cumulative acc by HxWxN, size(label) == HxWx1xN
        % if size(prediction) == 1x1xCxN, divide cumulative acc by N, size(label) == 1x1x1xN
        % if size(prediction) == 1x1x1xN, divide cumulative acc by N, size(label) == 1x1x1xN

        if size(bottom{1},3) > 1
            [~, argMax] = max(bottom{1}, [], 3);
            argMax = argMax -1 + l.accuracy_param.labelIndex_start;
            argMax(bottom{2} < l.accuracy_param.labelIndex_start)    = 0; % Important: we just compare class ID >= "labelIndex_start", so class ID < "labelIndex_start" will be marked as correct.
            bottom{2}(bottom{2} < l.accuracy_param.labelIndex_start) = 0; % this may not an necessary step, but for potentially risks.
            k =  argMax == bottom{2};

            if l.accuracy_param.meanClassAcc
                if counting == 0 && opts.gpuMode
                    perClassArea = gpuArray(perClassArea);
                    perClassAcc = gpuArray(perClassAcc);
                end
                counting = counting+size(bottom{1},4);
                for i=1:size(bottom{1},3)
                    correctLabelInd = i-1+l.accuracy_param.labelIndex_start;
                    mask = bottom{2}==correctLabelInd;
                    perClassAcc(i) = perClassAcc(i)+sum(argMax(mask)==correctLabelInd);
                    perClassArea(i)  = perClassArea(i)+sum(mask(:));
                end
                if counting == l.accuracy_param.dataNum
                    %top{1} = sum(perClassAcc)/sum(perClassArea)*counting;
                    nonZeroArea = perClassArea~=0;
                    top{2} = mean(perClassAcc(nonZeroArea)./perClassArea(nonZeroArea)).*counting;% because train.m will divide N,so we multiply N first
                else
                    top{2} = 0;
                end
            end
        else
            k = bsxfun(@eq, bottom{1}, bottom{2});
        end
        if l.accuracy_param.meanClassAcc
            if counting == l.accuracy_param.dataNum
                top{1} = sum(perClassAcc)/sum(perClassArea)*counting;
                % reset var
                counting = 0;
                perClassAcc = perClassAcc.*0;
                perClassArea = perClassArea.*0;
            end
        else
            top{1} = sum(k(:))/sum(sum(sum(bottom{2} >= l.accuracy_param.labelIndex_start)))*size(bottom{1},4); %don't divide N here, because train.m will do it for us
        end
    end
    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        bottom_diff = {[],[]};
    end
end
