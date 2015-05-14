function o = pooling(varargin)
%POOLING Compute mean class accuracy for you

o.name         = 'Pooling';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;


default_pooling_param = {
         'method' 'max' ...
    'kernel_size' [1 1] ...
            'pad' 0     ...
         'stride' [1 1] ...
};

    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update
        resource = {};

        if isfield(l, 'pooling_param')
            wp = nn.utils.vararginHelper(default_pooling_param, l.pooling_param);
        else
            wp = nn.utils.vararginHelper(default_pooling_param, default_pooling_param);
        end


        assert(numel(l.bottom)==1);
        assert(numel(l.top)==1);


        if numel(wp.kernel_size) == 1
            wp.kernel_size = [wp.kernel_size, wp.kernel_size];
        end
        if numel(wp.stride) == 1
            wp.stride = [wp.stride, wp.stride];
        end
        if numel(wp.pad) == 1
            wp.pad = [wp.pad, wp.pad, wp.pad, wp.pad];
        end


        btmSize = bottomSizes{1};
        topSizes = {[ceil([(btmSize(1)+2*wp.pad(1)-wp.kernel_size(1))/wp.stride(1)+1, (btmSize(2)+2*wp.pad(2)-wp.kernel_size(2))/wp.stride(2)+1]), btmSize(3), btmSize(4)]};


        %return updated param
        param.pooling_param = wp;
    end


    function [outputBlob, weightUpdate] = forward(opts, l, weights, blob)
        outputBlob{1} = vl_nnpool(blob{1}, l.pooling_param.kernel_size, 'pad', l.pooling_param.pad, 'stride', l.pooling_param.stride, 'method', l.pooling_param.method);
        weightUpdate = {};
    end


    function [mydzdx, mydzdw] = backward(opts, l, weights, blob, dzdy, mydzdw, mydzdwCumu)
        %numel(mydzdx) = numel(blob), numel(mydzdw) = numel(weights)
        mydzdx{1} = vl_nnpool(blob{1}, l.pooling_param.kernel_size, dzdy{1}, 'pad', l.pooling_param.pad, 'stride', l.pooling_param.stride, 'method', l.pooling_param.method);
        %mydzdw = {};
    end

end