function o = relu(varargin)
%RELU Compute mean class accuracy for you

o.name         = 'ReLU';
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
        outputBlob{1} = max(blob{1}, 0);
        weightUpdate = {};
    end


    function [outputdzdx, outputdzdw] = backward(opts, l, weights, blob, dzdy)
        %numel(outputdzdx) = numel(blob), numel(outputdzdw) = numel(weights)
        outputdzdx{1} = (blob{1} > 0) .* dzdy{1};
        outputdzdw = {};
    end

end