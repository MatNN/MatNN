function o = batchNormalization(networkParameter)
%BATCHNORMALIZATION

o.name         = 'BatchNormalization';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;


default_weight_param = {
            'name' {'', ''} ...
       'generator' {@nn.generator.constant, @nn.generator.constant} ...
       'generator_param' {{'value', 1}, {'value', 0}} ...
    'learningRate' single([1 1]) ...
     'weightDecay' single([0 0])
};

    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update

        if isfield(l, 'weight_param')
            wp = nn.utils.vararginHelper(default_weight_param, l.weight_param);
        else
            wp = nn.utils.vararginHelper(default_weight_param, default_weight_param);
        end

        assert(numel(l.bottom)==1);
        assert(numel(l.top)==1);
        topSizes = bottomSizes;

        N = bottomSizes{1}(3);

        resource.weight{1} = wp.generator{1}([1, 1, N, 1], wp.generator_param{1});
        resource.weight{2} = wp.generator{2}([1, 1, N, 1], wp.generator_param{2});

        %return updated param
        param.weight_param = wp;
    end


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        if ~opts.layerSettings.enableBnorm
            top{1} = bottom{1};
        else
            top{1} = vl_nnbnorm(bottom{1}, weights{1}, weights{2});
        end
    end


    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff)
        if ~opts.layerSettings.enableBnorm
            bottom_diff{1} = top_diff{1};
            return;
        end

        [ bottom_diff{1}, weights_diff{1}, weights_diff{2} ]= ...
                         vl_nnbnorm(bottom{1}, weights{1}, weights{2}, top_diff{1});
    end

end
