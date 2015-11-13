classdef Custom < nn.layers.template.BaseLayer

    % Default parameters
    properties (SetAccess = protected, Transient)
        default_customData_param = {
           'dataProvider' [] ... % function handle
             'batch_size' 1  ...
            'output_size' {} ... % a cell array of top sizes.
                'shuffle' false ...
        };
    end

    methods
        % CPU Forward
        function out = f(obj, varargin)
            error('You should call your function instead of calling this.');
        end
        % CPU Backward
        function in_diff = b(obj, varargin)
            in_diff = [];
        end

        % Forward function for training/testing routines
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            [top, weights, misc] = obj.params.customData.dataProvider(opts, obj.params, weights, misc, bottom, top);
        end
        % Backward function for training/testing routines
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            bottom_diff = {[]};
        end

        % Calc Output sizes
        function outSizes = outputSizes(obj, opts, inSizes)
            if iscell(inSizes)
                outSizes = obj.params.customData.output_size;
            else
                error('output_size must be a cell array.');
            end
        end

        % Setup function for training/testing routines
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==0, 'Custom data layer does not accept inputs.');
            assert(numel(baseProperties.top)>=1);
        end

    end

    

end





function o = customData(networkParameter)
%CUSTOMDATA for advanced users only

o.name         = 'CustomData';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;
o.outputSize   = [];


default_customData_param = {
   'dataProvider' [] ... % function handle
     'batch_size' 1  ...
    'output_size' {} ... % a function handle or a cell array of top sizes.
        'shuffle' false ...
};

    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update


        if isfield(l, 'customData_param')
            wp = nn.utils.vararginHelper(default_customData_param, l.customData_param);
        else
            wp = nn.utils.vararginHelper(default_customData_param, default_customData_param);
        end

        assert(numel(l.bottom)==0, [o.name, ' layer does not accept inputs.']);

        topSizes = wp.output_size;
        resource = {};

        %return updated param
        param.customData_param = wp;
    end


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        [top, weights, misc] = l.customData_param.dataProvider(networkParameter, opts, l, weights, misc, bottom, top);
    end


    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff)
        bottom_diff = {};
    end

end
