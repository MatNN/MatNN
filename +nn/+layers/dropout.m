function o = dropout(varargin)
%DROPOUT ...

o.name         = 'Dropout';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;

 
default_dropout_param = {
            'name' {''}      ...  %empty names means use autogenerated name
    'enable_terms' true      ...
            'rate' 0.5       ...
};

    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight, or .misc
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
            resource.misc{1} = rand(topSizes{1},'single') >= wp.rate;
        end

        %return updated param
        miscParam = wp;
        miscParam.enable_terms = true;
        miscParam.learningRate = 0;
        miscParam.weightDecay  = 0;
        param.dropout_param = wp;
        param.misc_param = miscParam;
    end


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)

        if opts.disableDropout || ~l.dropout_param.enable_terms
            top{1} = bottom{1};
        elseif opts.freezeDropout
            top{1} = bottom{1}.*misc{1};
        else
            if isa(bottom{1},'gpuArray')
                mask = single(1 / (1 - opts.rate)) * (gpuArray.rand(topSizes,'single') >= wp.rate);
                top{1} = bottom{1} .* mask;
            else
                mask = single(1 / (1 - opts.rate)) * (rand(topSizes,'single') >= wp.rate);
                top{1} = bottom{1} .* mask;
            end
            misc{1} = mask;
        end
    end


    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        %numel(bottom_diff) = numel(bottom), numel(weights_diff) = numel(weights)
        if opts.disableDropout || ~l.dropout_param.enable_terms
            bottom_diff{1} = top_diff{1};
        else
            bottom_diff{1} = top_diff{1} .* weights{1};
        end
    end


end