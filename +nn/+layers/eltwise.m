function o = eltwise(varargin)
%Eltwise Compute mean class accuracy for you

o.name         = 'Eltwise';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;


default_eltwise_param = {
    'operation' 'sum' ...
};

operator = @plus;
backwardFunc = @back_plus;

    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update
        resource = {};

        if isfield(l, 'eltwise_param')
            wp = nn.utils.vararginHelper(default_eltwise_param, l.eltwise_param);
        else
            wp = nn.utils.vararginHelper(default_eltwise_param, default_eltwise_param);
        end


        assert(numel(l.bottom)==2);
        assert(numel(l.top)==1);
        assert(numel(bottomSizes{1})==numel(bottomSizes{2}));

        switch lower(wp.operation)
            case 'sum'
                operator = @plus;
                backwardFunc = @back_plus;
            case 'prod'
                operator = @times;
                backwardFunc = @back_times;
            case 'max'
                operator = @max;
                backwardFunc = @back_max;
            case 'minus'
                operator = @minus;
                backwardFunc = @back_minus;
            otherwise
                error(['Not support operation:', wp.operation]);
        end

        topSizes = bottomSizes(1);

        %return updated param
        param.eltwise_param = wp;
    end


    function [outputBlob, weights] = forward(opts, l, weights, blob)
        outputBlob{1} = operator(blob{1}, blob{2});
    end


    function [mydzdx, mydzdw] = backward(opts, l, weights, blob, dzdy, mydzdw, mydzdwCumu)
        %numel(mydzdx) = numel(blob), numel(mydzdw) = numel(weights)
        [mydzdx{1}, mydzdx{2}] = backwardFunc(dzdy, blob{1}, blob{2});
        %mydzdw = {};
    end

    function [r1, r2] = back_plus(dzdy, ~, ~)
        r1 = dzdy;
        r2 = dzdy;
    end
    function [r1, r2] = back_times(dzdy, b1, b2)
        r1 = b2.*dzdy;
        r2 = b1.*dzdy;
    end
    function [r1, r2] = back_max(dzdy, b1, b2)
        r = max(b1,b2) == b1;
        r1 = dzdy.*r;
        r2 = dzdy.*(~r);
    end
    function [r1, r2] = back_minus(dzdy, ~, ~)
        r1 = dzdy;
        r2 = -dzdy;
    end

end