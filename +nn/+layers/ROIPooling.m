function o = ROIPooling(networkParameter)
%ROIPOOLING only support max pooling
% input 
%    1 -> bottom data
%    2 -> roi data, format: each column: [N_ind, x,y,X,Y]'

o.name         = 'ROI Pooling';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;


default_ROIPooling_param = {
      'output_size' single([6 6]) ...
    'spatial_scale' single(1/16)
};
af = [];
ab = [];
argmax_data = [];
    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update
        resource = {};

        if isfield(l, 'ROIPooling_param')
            wp = nn.utils.vararginHelper(default_ROIPooling_param, l.ROIPooling_param);
        else
            wp = nn.utils.vararginHelper(default_ROIPooling_param, default_ROIPooling_param);
        end


        assert(numel(l.bottom)==2); % [1] = bottom, [2] = roi data
        assert(numel(l.top)==1);
        assert(wp.spatial_scale > 0);
        assert(all(wp.output_size > 0));

        % check the number of roi data boxes of each image are equal to N
        %assert();

        if numel(networkParameter.gpus)>0
            ptxp = which('ROIPooling.ptx');
            cup  = which('ROIPooling.cu');
            af = nn.utils.gpu.createHandle(prod(bottomSizes{1}), ptxp, cup, 'ROIPoolForward');
            ab = nn.utils.gpu.createHandle(prod(bottomSizes{1}), ptxp, cup, 'ROIPoolBackward');
        else
            error('Affine Layer runs on GPU (currently).');
        end

        topSizes = @(x){[wp.output_size(1) wp.output_size(2) x{1}(3) x{1}(4)]};

        %return updated param
        param.ROIPooling_param = wp;
    end


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        if opts.gpuMode
            s = nn.utils.size4D(bottom{1});
            len = prod(s);
            top{1} = gpuArray.zeros(l.ROIPooling_param.output_size(1), l.ROIPooling_param.output_size(2), s(3), s(4), 'single');
            argmax_data = top{1};
            top{1} = feval(af, len, bottom{1}, l.ROIPooling_param.spatial_scale, s(3), s(1), s(2), l.ROIPooling_param.output_size(1), l.ROIPooling_param.output_size(2), bottom{2}, top{1}, argmax_data);
        end
    end


    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff)
        if opts.gpuMode
            s = nn.utils.size4D(bottom{1});
            len = prod(s);
            bottom_diff{1} = bottom{1}.*0;
            bottom_diff{1} = feval(af, len, top_diff{1}, argmax_data, size(bottom{2},1), l.ROIPooling_param.spatial_scale, s(3), s(1), s(2), l.ROIPooling_param.output_size(1), l.ROIPooling_param.output_size(2), bottom_diff{1}, bottom{2});
        end
    end

end