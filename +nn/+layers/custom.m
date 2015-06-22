function o = custom(varargin)
%CUSTOM 
%
% LIMITATIONS
%  1. NO Weights/ misc support
%  2. input and output size must be the same
%  3. only one input and output
%  4. your custom forward function must accepts the following inputs:
%     function(bottom)
%     and the genrated output must be "top"
%  5. your custom forward function must accepts the following inputs:
%     function(bottom, top, top_diff)
%     and the genrated output must be "bottom_diff"

o.name         = 'CUSTOM';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;

default_custom_param = {
      'forward'  [] ...
      'backward' []
};

    function [resource, topSizes, param] = setup(l, bottomSizes)
        resource = {};

        if isfield(l, 'custom_param')
            wp = nn.utils.vararginHelper(default_custom_param, l.custom_param);
        else
            error('You must sepcify custom forward and backward function');
        end

        assert(numel(l.bottom)==1);
        assert(numel(l.top)==1);
        assert(~isempty(wp.forward));
        assert(~isempty(wp.backward));

        topSizes = bottomSizes;

        param.custom_param = wp;

    end
    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        top = {l.custom_param.forward(bottom{1})};

    end
    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff, weights_diff_isCumulate)
        bottom_diff = {l.custom_param.backward(bottom{1}, top{1}, top_diff{1})};
    end
end