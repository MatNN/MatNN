function o = silence(varargin)
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


    function [outputBlob, weights] = forward(opts, l, weights, blob)
        outputBlob   = {};
    end


    function [mydzdx, mydzdw] = backward(opts, l, weights, blob, dzdy, mydzdw, mydzdwCumu)
        %numel(mydzdx) = numel(blob), numel(mydzdw) = numel(weights)
        if opts.gpu
            zero = gpuArray(0);
            for i=1:numel(blob)
                mydzdx{1} = blob{1}*zero;
            end
        else
            for i=1:numel(blob)
                mydzdx{1} = blob{1}*0;
            end
        end

    end

end