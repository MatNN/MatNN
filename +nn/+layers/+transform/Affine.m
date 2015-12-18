classdef Affine < nn.layers.template.BaseLayer

    % Default parameters
    properties (SetAccess = protected, Transient)
        default_affine_param = {
            'sampler' 'bilinear' ... % currently only support bilinear interpolation
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

        function out = f(obj, in, affineMatrix)
            out = in.*single(0);
            s = nn.utils.size4D(in);
            len = prod(s);
            obj.forwardHandle.GridSize = ceil( len/obj.MaxThreadsPerBlock );
            out = feval(obj.forwardHandle, in, s, affineMatrix, len, out);
        end
        function [in_diff, affine_diff] = b(obj, in, affineMatrix, out, out_diff)
            s   = nn.utils.size4D(out_diff);
            len = prod(s);
            in_diff = in.*single(0);
            affine_diff = affineMatrix.*single(0);
            obj.backwardHandle.GridSize = ceil( len/obj.MaxThreadsPerBlock );
            [in_diff, affine_diff] = feval(obj.backwardHandle, in, s, affineMatrix, len, out, out_diff, in_diff, affine_diff);
            
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
                if obj.params.affine.showDebugWindow
                    if mod(obj.count, 20) == 0
                        s = nn.utils.size4D(data.val{obj.bottom(1)});
                        t = gather(data.val{obj.bottom(2)}(:,:,:,1));
                        o1 = trans(t, [1,1]      , s(1:2));
                        o2 = trans(t, [s(1),1]   , s(1:2));
                        o3 = trans(t, [s(1),s(2)], s(1:2));
                        o4 = trans(t, [1,s(2)]   , s(1:2));
                        ox = [o1(2),o2(2),o3(2),o4(2)];
                        oy = [o1(1),o2(1),o3(1),o4(1)];

                        subplot(1,2,1), imshow(gather(data.val{obj.bottom(1)}(:,:,:,1)), []), ...
                                        line(ox(1:2), oy(1:2), 'LineWidth',4,'Color','r'), ...
                                        line(ox(2:3), oy(2:3), 'LineWidth',4,'Color','g'), ...
                                        line(ox(3:4), oy(3:4), 'LineWidth',4,'Color','b'), ...
                                        line(ox([4,1]), oy([4,1]), 'LineWidth',4,'Color','y');
                        set(gca,'Clipping','off');
                        subplot(1,2,2), imshow(gather(data.val{obj.top}(:,:,:,1)), []);
                        drawnow;
                        obj.count = 0;
                    end
                    obj.count = obj.count+1;
                end
            else
                error('Affine Layer : only support gpu mode currently.');
            end
            function o = trans(a,o,s) % p = [y,x], s = [h,w]
                p = 2*((o./s)-0.5);
                o(2) = a(1)*p(2) + a(3)*p(1) + a(5);
                o(1) = a(2)*p(2) + a(4)*p(1) + a(6);
                o = (o./2+0.5).*s;
            end
            data.forwardCount(obj.bottom, obj.top);
        end
        function backward(obj)
            net = obj.net;
            data = net.data;
            if net.opts.gpu
                [bottom_diff1, bottom_diff2] = obj.b(data.val{obj.bottom}, data.val{obj.top}, data.diff{obj.top});
            else
                error('Affine Layer : only support gpu mode currently.');
            end
            data.backwardCount(obj.bottom, obj.top, bottom_diff1, bottom_diff2);
        end
        
        function outSizes = outputSizes(obj, inSizes)
            assert(inSizes{2}(1) == 1 && inSizes{2}(2) == 1 && inSizes{2}(3) == 6 && inSizes{2}(4) == inSizes{1}(4));
            outSizes = inSizes(1);
        end
        function setParams(obj)
            obj.setParams@nn.layers.template.BaseLayer();
            p = obj.params.affine;
            assert(strcmpi(p.sampler, 'bilinear'));
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.bottom)==2);
            assert(numel(obj.top)==1);
            obj.createGPUFun(inSizes{1});
        end
        function createGPUFun(obj, sampleSize)
            obj.forwardHandle = obj.createForwardHandle();
            obj.backwardHandle = obj.createBackwardHandle();
        end
        function h = createForwardHandle(obj, varargin)
            if isempty(varargin) || strcmpi(varargin{1}, 'GPU')
                mf = fileparts(mfilename('fullpath'));
                ptxp = fullfile(mf, 'private', 'affine.ptx');
                cup = fullfile(mf, 'private', 'affine.cu');
                h = nn.utils.gpu.createHandle(1, ptxp, cup, 'AffineForward');
            elseif strcmpi(varargin{1}, 'CPU')
                h = [];
            end
        end
        function h = createBackwardHandle(obj, varargin)
            if isempty(varargin) || strcmpi(varargin{1}, 'GPU')
                mf = fileparts(mfilename('fullpath'));
                ptxp = fullfile(mf, 'private', 'affine.ptx');
                cup = fullfile(mf, 'private', 'affine.cu');
                h = nn.utils.gpu.createHandle(1, ptxp, cup, 'AffineBackward');
            elseif strcmpi(varargin{1}, 'CPU')
                h = [];
            end
        end
    end
end
