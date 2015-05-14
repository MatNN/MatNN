function o = dropout(varargin)
%DROPOUT ...

o.name         = 'Dropout';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;

 
default_dropout_param = {
            'name' {'', ''}  ...  %empty names means use autogenerated name
    'enable_terms' true      ...  % false = don't use dropout, but still generate mask
            'rate' 0.5       ...
    'learningRate' single(0) ...
     'weightDecay' single(0)
};

    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update
        resource = {};

        if isfield(l, 'dropout_param')
            wp = nn.utils.vararginHelper(default_dropout_param, l.dropout_param);
        else
            wp = nn.utils.vararginHelper(default_dropout_param, default_dropout_param);
        end

        
        assert(numel(l.bottom)==1);
        assert(numel(l.top)==1);

        
        topSizes = bottomSizes(1);
        if wp.enable_terms
            scale = single(1 / (1 - wp.rate)) ;
            resource.weight{1} = rand(topSizes{1},'single') >= wp.rate;
        end

        %return updated param
        param.dropout_param = wp;
    end


    function [outputBlob, weights] = forward(opts, l, weights, blob)

        if opts.disableDropout || ~l.dropout_param.enable_terms
            outputBlob{1} = blob{1};
        elseif opts.freezeDropout
            outputBlob{1} = blob{1}.*weights{1};
        else
            if isa(blob{1},'gpuArray')
                mask = single(1 / (1 - opts.rate)) * (gpuArray.rand(topSizes,'single') >= wp.rate);
                outputBlob{1} = blob{1} .* mask;
            else
                mask = single(1 / (1 - opts.rate)) * (rand(topSizes,'single') >= wp.rate);
                outputBlob{1} = blob{1} .* mask;
            end
            weights{1} = mask;
        end
    end


    function [mydzdx, mydzdw] = backward(opts, l, weights, blob, dzdy, mydzdw, mydzdwCumu)
        %numel(mydzdx) = numel(blob), numel(mydzdw) = numel(weights)
        if opts.disableDropout || ~l.dropout_param.enable_terms
            mydzdx{1} = dzdy{1};
        else
            mydzdx{1} = dzdy{1} .* weights{1};
        end
    end


end