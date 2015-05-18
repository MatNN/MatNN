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


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        top   = {};
    end


    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        if opts.gpu
            zero = gpuArray(0);
            for i=1:numel(bottom)
                bottom_diff{1} = bottom{1}*zero;
            end
        else
            for i=1:numel(bottom)
                bottom_diff{1} = bottom{1}*0;
            end
        end

    end

end