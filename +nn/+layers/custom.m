function o = custom(networkParameter)
%CUSTOM 
%
% LIMITATIONS
%  1. NO Weights/ misc support
%  2. input and output size must be the same
%  3. your custom forward function must accepts the following inputs:
%     function(bottom)
%     and the genrated output must be "top"
%  4. your custom forward function must accepts the following inputs:
%     function(bottom, top, top_diff)
%     and the genrated output must be "bottom_diff"
%  5. you SHOULD NOT assign forward_func/backward_func a nested function.
%     It causes overflow when saving a network

o.name         = 'CUSTOM';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;

default_custom_param = {
      'forward_func'  [] ...
      'backward_func' [] ...
      'output_size' [] ...
};

    function [resource, topSizes, param] = setup(l, bottomSizes)
        resource = {};

        if isfield(l, 'custom_param')
            wp = nn.utils.vararginHelper(default_custom_param, l.custom_param);
        else
            error('You must sepcify custom forward and backward function');
        end

        assert(~isempty(wp.forward_func));
        assert(~isempty(wp.backward_func));
        assert(numel(l.top) == numel(wp.output_size));

        topSizes = wp.output_size;

        param.custom_param = wp;

    end
    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        top = l.custom_param.forward_func(bottom);

    end
    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff)
        bottom_diff = l.custom_param.backward_func(bottom, top, top_diff);
    end
end