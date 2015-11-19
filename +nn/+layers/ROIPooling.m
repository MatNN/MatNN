classdef ROIPooling < nn.layers.template.BaseLayer
% ROIPooling
% [1] = bottom, [2] = roi data
    properties (SetAccess = protected, Transient)
        default_roiPooling_param = {
              'output_size' single([6 6]) ...
            'spatial_scale' single(1/16)
        };
    end
    properties (Access = protected, Transient)
        argmaxData;
    end
    properties (Access = protected)
        forwardHandle;
        backwardHandle;
    end

    methods
        function v = propertyDevice(obj)
            v = obj.propertyDevice@nn.layers.template.BaseLayer();
            v.forwardHandle = 1;
            v.backwardHandle  = 1;
        end

        function out = f(~, in, kernel, pad, stride, method) %#ok
            error('Not implemented yet.');
        end
        function in_diff = b(~, in, out_diff, kernel, pad, stride, method) %#ok
            error('Not implemented yet.');
        end
        function [out, argmax_data] = gf(~, in, rois, output_size, spatial_scale)
            s = nn.utils.size4D(in);
            len = prod(s);
            out = gpuArray.zeros(output_size(1), output_size(2), s(3), s(4), 'single');
            argmax_data = out;
            [out, argmax_data] = feval(obj.forwardHandle, len, in, spatial_scale, s(3), s(1), s(2), output_size(1), output_size(2), rois, out, argmax_data);
        end
        function in_diff = gb(~, in, rois, out_diff, output_size, spatial_scale, argmax_data)
            s = nn.utils.size4D(in);
            len = prod(s);
            in_diff = in.*0;
            in_diff = feval(obj.backwardHandle, len, out_diff, argmax_data, size(rois,1), spatial_scale, s(3), s(1), s(2), output_size(1), output_size(2), in_diff, rois);
        end
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            p = obj.params.roiPooling;
            if opts.gpuMode
                [top{1}, obj.argmaxData] = obj.gf(bottom{1}, bottom{2}, p.output_size, p.spatial_scale);
            else
                error('Not implemented yet.');
            end
        end
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            p = obj.params.roiPooling;
            if opts.gpuMode
                bottom_diff{1} = obj.gb(bottom{1}, bottom{2}, top_diff{1}, p.output_size, p.spatial_scale, obj.argmaxData);
            else
                error('Not implemented yet.');
            end
            
        end
        function outSizes = outputSizes(obj, opts, inSizes)
            p = obj.params.roiPooling;
            outSizes = {[p.output_size(1) p.output_size(2) inSizes{1}(3) inSizes{1}(4)]};
        end
        function setParams(obj, baseProperties)
            obj.setParams@nn.layers.template.BaseLayer(baseProperties);
            p = obj.params.roiPooling;
            assert(p.spatial_scale > 0);
            assert(all(p.output_size > 0));
        end
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==2);
            assert(numel(baseProperties.top)==1);
            obj.createGPUFun(inSizes{1});
        end
        function createGPUFun(obj, sampleSize)
            mf = fileparts(mfilename('fullpath'));
            ptxp = fullfile(mf, 'private', 'ROIPooling.ptx');
            cup = fullfile(mf, 'private', 'ROIPooling.cu');
            obj.forwardHandle = nn.utils.gpu.createHandle(prod(sampleSize), ptxp, cup, 'ROIPoolForward');
            obj.backwardHandle = nn.utils.gpu.createHandle(prod(sampleSize), ptxp, cup, 'ROIPoolBackward');
        end   
    end
end