function o = sigmoid(varargin)
%CONVOLUTION Compute mean class accuracy for you

o.name         = 'Sigmoid';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;


    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update
        resource = {};

        assert(numel(l.bottom)==1);
        assert(numel(l.top)==1);

        topSizes = bottomSizes(1);


        %return updated param
        param = {};
    end


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        top{1} = 1./(1+exp(-bottom{1}));
    end


    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        sigmoid =  1./(1+exp(-bottom{1})) ;
        bottom_diff{1} = top_diff{1}.*(sigmoid.*(1-sigmoid));
        %weights_diff = {};
    end

end