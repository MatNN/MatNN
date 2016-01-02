classdef BilinearInterpolation < nn.layers.template.BaseLayer

    % Default parameters
    properties (SetAccess = protected, Transient)
        default_bilinear_param = {
            'showDebugWindow' false ...
        };
    end

    % intermediate savings (computed values, recomputed every time)
    properties (SetAccess = {?nn.BaseObject}, GetAccess = public)
        forwardHandle;
        backwardHandle;
        count = 0;
    end


    methods
        function v = propertyDevice(obj)
            v = obj.propertyDevice@nn.layers.template.BaseLayer();
            v.count = 0;
            v.forwardHandle = 'createForwardHandle';
            v.backwardHandle  = 'createBackwardHandle';
        end

        function out = f(obj, in, pos)
            s = nn.utils.size4D(in);
            outSize = nn.utils.size4D(pos);
            outSize(3) = s(3);
            out = gpuArray.zeros(outSize, 'single');

            obj.forwardHandle.GridSize = ceil( prod(s)/obj.MaxThreadsPerBlock );
            out = feval(obj.forwardHandle, in, s, pos, out, outSize);
        end
        function [in_diff, pos_diff] = b(obj, in, pos, out, out_diff)
            s   = nn.utils.size4D(out_diff);
            os  = nn.utils.size4D(out);
            len = prod(s);
            in_diff = in.*single(0);
            pos_diff = pos.*single(0);
            obj.backwardHandle.GridSize = ceil( len/obj.MaxThreadsPerBlock );
            [in_diff, pos_diff] = feval(obj.backwardHandle, in, s, pos, out, os, out_diff, in_diff, pos_diff);
            
        end
        function out = gf(obj, varargin)
            out = obj.f(varargin{:});
        end
        function in_diff = gb(obj, varargin)
            in_diff = obj.b(varargin{:});
        end

        function forward(obj)
            net = obj.net;
            data = net.data;
            if net.opts.gpu
                data.val{obj.top} = obj.f(data.val{obj.bottom});
                if obj.params.bilinear.showDebugWindow
                    if mod(obj.count, 20) == 0
                        subplot(1,2,1), imshow(gather(data.val{obj.bottom(1)}(:,:,:,1)), []);
                        subplot(1,2,2), imshow(gather(data.val{obj.top}(:,:,:,1)), []);
                        drawnow;
                        obj.count = 0;
                    end
                    obj.count = obj.count+1;
                end
            else
                error('BilinearInterpolation Layer : only support gpu mode currently.');
            end
            data.forwardCount(obj.bottom, obj.top);
        end
        function backward(obj)
            net = obj.net;
            data = net.data;
            if net.opts.gpu
                [bottom_diff1, bottom_diff2] = obj.b(data.val{obj.bottom}, data.val{obj.top}, data.diff{obj.top});
            else
                error('BilinearInterpolation Layer : only support gpu mode currently.');
            end
            data.backwardCount(obj.bottom, obj.top, bottom_diff1, bottom_diff2);
        end
        
        function outSizes = outputSizes(obj, inSizes)
            assert(inSizes{2}(3) == 2);
            outSizes = inSizes(1);
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.bottom)==2);
            assert(numel(obj.top)==1);
            obj.createGPUFun(inSizes{2});
        end
        function createGPUFun(obj, sampleSize)
            obj.forwardHandle = obj.createForwardHandle();
            obj.backwardHandle = obj.createBackwardHandle();
        end
        function h = createForwardHandle(obj, varargin)
            if isempty(varargin) || strcmpi(varargin{1}, 'GPU')
                mf = fileparts(mfilename('fullpath'));
                ptxp = fullfile(mf, 'private', 'bilinearInterpolation.ptx');
            cup = fullfile(mf, 'private', 'bilinearInterpolation.cu');
                h = nn.utils.gpu.createHandle(1, ptxp, cup, 'BilinearInterpolationForward');
            elseif strcmpi(varargin{1}, 'CPU')
                h = [];
            end
        end
        function h = createBackwardHandle(obj, varargin)
            if isempty(varargin) || strcmpi(varargin{1}, 'GPU')
                mf = fileparts(mfilename('fullpath'));
                ptxp = fullfile(mf, 'private', 'bilinearInterpolation.ptx');
            cup = fullfile(mf, 'private', 'bilinearInterpolation.cu');
                h = nn.utils.gpu.createHandle(1, ptxp, cup, 'BilinearInterpolationBackward');
            elseif strcmpi(varargin{1}, 'CPU')
                h = [];
            end
        end
    end
end
