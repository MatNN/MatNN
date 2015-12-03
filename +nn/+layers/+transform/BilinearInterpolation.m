classdef BilinearInterpolation < nn.layers.template.BaseLayer

    % Default parameters
    properties (SetAccess = protected, Transient)
        default_bilinear_param = {
            'showDebugWindow' false ...
        };
    end

    % intermediate savings (computed values, recomputed every time)
    properties (SetAccess = {?nn.layers.template.BaseLayer}, GetAccess = public)
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

        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            if opts.gpuMode
                data.val{l.top} = obj.f(data.val{l.bottom(1)}, data.val{l.bottom(2)});
                if obj.params.bilinear.showDebugWindow
                    if mod(obj.count, 20) == 0
                        subplot(1,2,1), imshow(gather(data.val{l.bottom(1)}(:,:,:,1)), []);
                        subplot(1,2,2), imshow(gather(data.val{l.top}(:,:,:,1)), []);
                        drawnow;
                        obj.count = 0;
                    end
                    obj.count = obj.count+1;
                end
            else
                error('BilinearInterpolation Layer : only support gpu mode currently.');
            end
        end
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            if opts.gpuMode
                [bottom_diff{1}, bottom_diff{2}] = obj.b(data.val{l.bottom(1)}, data.val{l.bottom(2)}, data.val{l.top}, data.diff{l.top});
            else
                error('BilinearInterpolation Layer : only support gpu mode currently.');
            end
            data = nn.utils.accumulateData(opts, data, l, bottom_diff{:});
        end
        
        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            assert(inSizes{2}(3) == 2);
            outSizes = inSizes(1);
        end
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.bottom)==2);
            assert(numel(l.top)==1);
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
                h = nn.utils.gpu.createHandle(prod(sampleSize), ptxp, cup, 'BilinearInterpolationForward');
            elseif strcmpi(varargin{1}, 'CPU')
                h = [];
            end
        end
        function h = createBackwardHandle(obj, varargin)
            if isempty(varargin) || strcmpi(varargin{1}, 'GPU')
                mf = fileparts(mfilename('fullpath'));
                ptxp = fullfile(mf, 'private', 'bilinearInterpolation.ptx');
            cup = fullfile(mf, 'private', 'bilinearInterpolation.cu');
                h = nn.utils.gpu.createHandle(prod(sampleSize), ptxp, cup, 'BilinearInterpolationBackward');
            elseif strcmpi(varargin{1}, 'CPU')
                h = [];
            end
        end
    end
end
