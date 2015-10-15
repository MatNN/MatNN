function o = reshape(networkParameter)
%RELU Rectified linear unit

o.name         = 'Reshape';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;

default_reshape_param = {
    'output_size'   {[],1,1,0} % 0=current size
};
origShape = [];
    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update

        if isfield(l, 'reshape_param')
            wp = nn.utils.vararginHelper(default_reshape_param, l.reshape_param);
        else
            wp = nn.utils.vararginHelper(default_reshape_param, default_reshape_param);
        end


        resource = {};
        assert(numel(l.bottom)==1);
        assert(numel(l.top)==1);
        if sum(cellfun('isempty', wp.output_size)) >=2
            error('there should be 1 unknown output size');
        end


        % replace 0
        os = wp.output_size;
        for i=1:4
            if os{i} == 0
                os{i} = bottomSizes{1}(i);
            end
        end
        % test reshape size
        tmpData = false(bottomSizes{1});
        tmpData = reshape(tmpData, os{:});


        topSizes = {nn.utils.size4D(tmpData)};
        %return updated param
        param = {};
    end

    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        origShape = nn.utils.size4D(bottom{1});
        os = l.reshape_param.output_size;
        for i=1:4
            if os{i} == 0
                os{i} = origShape(i);
            end
        end
        top{1} = reshape(bottom{1}, os{:});
    end

    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff)
        bottom_diff{1} = reshape(top_diff{1}, origShape);
    end

end