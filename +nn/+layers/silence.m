function o = silence()
%RELU Compute mean class accuracy for you

o.name         = 'Silence';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;


    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update
        resource = {};

        assert(numel(l.bottom)==0);
        assert(numel(l.top)~=0);

        topSizes = {};


        %return updated param
        param = {};
    end


    function [outputBlob, weightUpdate] = forward(opts, l, weights, blob)
        outputBlob   = {};
        weightUpdate = {};
    end


    function [outputdzdx, outputdzdw] = backward(opts, l, weights, blob, dzdy)
        %numel(outputdzdx) = numel(blob), numel(outputdzdw) = numel(weights)
        
        if opts.gpu
            zero = gpuArray(0);
            for i=1:numel(blob)
                outputdzdx{1} = blob{1}*zero;
            end
        else
            for i=1:numel(blob)
                outputdzdx{1} = blob{1}*0;
            end
        end
        
        outputdzdw = {};
    end

end