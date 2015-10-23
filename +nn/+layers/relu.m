function o = relu(networkParameter)
%RELU Rectified linear unit

o.name         = 'ReLU';
o.generateLoss = false;
o.setup        = @setup;
if numel(networkParameter.gpus) > 0
    o.forward  = @forwardGPU;
    o.backward = @backwardGPU;
    zero = gpuArray.zeros(1, 'single');
else
    o.forward  = @forward;
    o.backward = @backward;
end


    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update
        resource = {};
        assert(numel(l.bottom)==1);
        assert(numel(l.top)==1);
        topSizes = bottomSizes(1);
        %return updated param
        param = {};
    end

    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        top = {max(bottom{1}, 0)};
    end

    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff)
        bottom_diff = {(bottom{1} > 0) .* top_diff{1}};
    end

    function [top, weights, misc] = forwardGPU(opts, l, weights, misc, bottom, top)
        top{1} = arrayfun(@fG, bottom{1}, zero);
    end

    function [bottom_diff, weights_diff, misc] = backwardGPU(opts, l, weights, misc, bottom, top, top_diff, weights_diff)
        bottom_diff{1} = arrayfun(@bG, bottom{1}, top_diff{1}, zero);
    end

end

% GPU kernels
function a = fG(a, z)
    a = max(a, z);
end
function b = bG(b, tf, z)
    if b > z
        b = tf;
    else
        b = z;
    end
end