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
