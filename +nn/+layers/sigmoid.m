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


    function [outputBlob, weights] = forward(opts, l, weights, blob)
        outputBlob{1} = 1./(1+exp(-blob{1}));
    end


    function [mydzdx, mydzdw] = backward(opts, l, weights, blob, dzdy, mydzdw, mydzdwCumu)
        %numel(mydzdx) = numel(blob), numel(mydzdw) = numel(weights)
        sigmoid =  1./(1+exp(-blob{1})) ;
        mydzdx{1} = dzdy{1}.*(sigmoid.*(1-sigmoid));
        mydzdw = {};
    end

end