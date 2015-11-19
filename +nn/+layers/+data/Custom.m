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
