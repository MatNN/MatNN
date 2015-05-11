function o = sigmoid()
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


    function [outputBlob, weightUpdate] = forward(opts, l, weights, blob)
        outputBlob{1} = 1./(1+exp(-blob{1}));
        weightUpdate = {};
    end


    function [outputdzdx, outputdzdw] = backward(opts, l, weights, blob, dzdy)
        %numel(outputdzdx) = numel(blob), numel(outputdzdw) = numel(weights)
        sigmoid =  1./(1+exp(-blob{1})) ;
        outputdzdx{1} = dzdy{1}.*(sigmoid.*(1-sigmoid));
        outputdzdw = {};
    end

end