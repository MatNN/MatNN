classdef RandDistort < handle
% RandDistort
% top1 = distorted data
% top2 = affine matrix

    % Default parameters
    properties (SetAccess = protected, Transient)
        default_randDistort_param = {
            'angle'  [-50, 50] ...
            'scaleX' [0.3,  1] ...
            'scaleY' [0.3,  1] ...
            'scaleEQ'   false  ... % set to true if you want scaleX = scaleY
            'shiftX' [0,  0.3] ...
            'shiftY' [0,  0.3] ...
            'extend' [28,  28] ... % must >=0
        };
    end

    % intermediate savings (computed values, recomputed every time)
    properties (Access = protected)
        forwardHandle;
    end
    properties (Constant)
        MaxThreadsPerBlock = 256;
    end


    methods
        function v = propertyDevice(obj)
            v = obj.propertyDevice@nn.layers.template.BaseLayer();
            v.forwardHandle = 1;
        end
        function varargout = f(obj, in, angles, scaleX, scaleY, scaleEQ, shiftX, shiftY, extend)
            error('not implemented yet.');
        end
        function varargout = gf(obj, in, angles, scaleX, scaleY, scaleEQ, shiftX, shiftY, extend)
             out = in.*0;
            s = nn.utils.size4D(in);
            len = prod(s);
            ral = gpuArray.rand(s(4),1, 'single')*(angles(2)-angles(1))+angles(1);
            if scaleEQ
                r = gpuArray.rand(s(4),1, 'single');
                rsx = r*(scaleX(2)-scaleX(1))+scaleX(1);
                rsy = r*(scaleY(2)-scaleY(1))+scaleY(1);
            else
                rsx = gpuArray.rand(s(4),1, 'single')*(scaleX(2)-scaleX(1))+scaleX(1);
                rsy = gpuArray.rand(s(4),1, 'single')*(scaleY(2)-scaleY(1))+scaleY(1);
            end

            rix = gpuArray.rand(s(4),1, 'single')*(shiftX(2)-shiftX(1))+shiftX(1);
            riy = gpuArray.rand(s(4),1, 'single')*(shiftY(2)-shiftY(1))+shiftY(1);
            w = gpuArray.zeros(1,1,6,s(4), 'single');
            %rix = rix-(rsx-1)./2;
            %riy = riy-(rsy-1)./2;

            w(1,1,1,:) = cosd(ral).*rsx;
            w(1,1,2,:) = sind(ral).*rsy;
            w(1,1,3,:) = -sind(ral).*rsx;
            w(1,1,4,:) = cosd(ral).*rsy;
            
            w(1,1,5,:) = rix;
            w(1,1,6,:) = riy;

            obj.forwardHandle.GridSize = ceil(s/obj.MaxThreadsPerBlock);
            if all(extend==0)
                out = feval(obj.forwardHandle, in, s, w, len, out);
            else
                tmpp = feval(obj.forwardHandle, in, s, w, len, out);
                out = gpuArray.zeros(extend(1)+s(1),extend(2)+s(2),1,s(4),'single');
                randPosx = randi(extend(1)+1,1,s(4));
                randPosy = randi(extend(2)+1,1,s(4));
                for i=1:s(4)
                    out(randPosy(i):(randPosy(i)+s(1)-1), randPosx(i):(randPosx(i)+s(2)-1),1,i) = tmpp(:,:,1,i);
                end
            end
            varargout{1} = out;
            %top{1}(randperm(len, floor(len*0.05))) = randi(255,1,floor(len*0.05));
            if numel(nargout)==2
                ww = reshape(w,2,3,[]);
                w = gpuArray.zeros(3,3,s(4),'single');
                w(1:2,1:3,:) = ww;
                w(3,3,:) = 1.0;
                w = pagefun(@inv,w);
                w = w(1:2,1:3,:);
                w = reshape(w,1,1,6,[]);
                varargout{2} = w;
            end
        end
        function [in_diff] = b(obj, varargin)
            in_diff = [];
        end

        % Forward function for training/testing routines
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            p = obj.params.randDistort;
            if opts.gpuMode
                if numel(top==1)
                    top{1} = obj.gf(bottom{1}, p.angle, p.scaleX, p.scaleY, p.scaleEQ, p.shiftX, p.shiftY, p.extend);
                elseif numel(top==2)
                    [top{1}, top{2}] = obj.gf(bottom{1}, p.angle, p.scaleX, p.scaleY, p.scaleEQ, p.shiftX, p.shiftY, p.extend);
                else
                    error('top number mismatch.');
                end
            else
                error('RandDistort Layer : only support gpu mode currently.');
            end
        end
        % Backward function for training/testing routines
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            bottom_diff = {};
        end
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==1);
            assert(numel(baseProperties.top)>=1 && numel(baseProperties.top)<=2);
            obj.createGPUFun();
        end
        function createGPUFun(obj, sampleSize)
            mf = fileparts(mfilename('fullpath'));
            ptxp = fullfile(mf, 'private', 'affine.ptx');
            cup = fullfile(mf, 'private', 'affine.cu');
            obj.forwardHandle = nn.utils.gpu.createHandle(prod(sampleSize), ptxp, cup, 'AffineForward');
        end   

end


% function o = randomDistortion(networkParameter)
% %RANDOMDISTORTION No backpropagation!!!
% %  top 2 = affine parameters

% o.name         = 'RandomDistortion';
% o.generateLoss = false;
% o.setup        = @setup;
% o.forward      = @forward;
% o.backward     = @backward;
% o.outputSize   = [];

% default_distort_param = {
%     'angle'  [-50, 50] ...
%     'scaleX' [0.3,  1] ...
%     'scaleY' [0.3,  1] ...
%     'scaleEQ'   false  ... % set to true if you want scaleX = scaleY
%     'shiftX' [0,  0.3] ...
%     'shiftY' [0,  0.3] ...
%     'extend' [28,  28] ... % must >=0
% };

%     function [resource, topSizes, param] = setup(l, bottomSizes)
%         % resource only have .weight
%         % if you have other outputs you want to save or share
%         % you can set its learning rate to zero to prevent update

%         if isfield(l, 'distort_param')
%             wp = nn.utils.vararginHelper(default_distort_param, l.distort_param);
%         else
%             wp = nn.utils.vararginHelper(default_distort_param, default_distort_param);
%         end


%         assert(numel(l.bottom)==1);
%         assert(numel(l.top)<=2);
%         assert(all(wp.extend>=0));
%         resource={};
%         if numel(l.top)==1
%             topSizes = @(x) {[wp.extend(1)+x{1}(1), wp.extend(2)+x{1}(2),x{1}(3), x{1}(4)]};
%         else
%             topSizes = @(x) {[wp.extend(1)+x{1}(1), wp.extend(2)+x{1}(2),x{1}(3), x{1}(4)], [1,1,6,x{1}(4)]};
%         end


%         ptxp = which('affine.ptx');
%         cup  = which('affine.cu');
%         wp.forward = nn.utils.gpu.createHandle(prod(bottomSizes{1}), ptxp, cup, 'AffineForward');
%         param.distort_param = wp;
%     end


%     function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
%         top{1} = bottom{1}.*0;
%         s = nn.utils.size4D(bottom{1});
%         len = prod(s);
%         p = l.distort_param;

%         if opts.gpuMode
%             ral = gpuArray.rand(s(4),1, 'single')*(p.angle(2)-p.angle(1))+p.angle(1);
%             if p.scaleEQ
%                 r = gpuArray.rand(s(4),1, 'single');
%                 rsx = r*(p.scaleX(2)-p.scaleX(1))+p.scaleX(1);
%                 rsy = r*(p.scaleY(2)-p.scaleY(1))+p.scaleY(1);
%             else
%                 rsx = gpuArray.rand(s(4),1, 'single')*(p.scaleX(2)-p.scaleX(1))+p.scaleX(1);
%                 rsy = gpuArray.rand(s(4),1, 'single')*(p.scaleY(2)-p.scaleY(1))+p.scaleY(1);
%             end

%             rix = gpuArray.rand(s(4),1, 'single')*(p.shiftX(2)-p.shiftX(1))+p.shiftX(1);
%             riy = gpuArray.rand(s(4),1, 'single')*(p.shiftY(2)-p.shiftY(1))+p.shiftY(1);
%             w = gpuArray.zeros(1,1,6,s(4), 'single');
%         else
%             ral = rand(s(4),1, 'single')*(p.angle(2)-p.angle(1))+p.angle(1);
%             if p.scaleEQ
%                 r = rand(s(4),1, 'single');
%                 rsx = r*(p.scaleX(2)-p.scaleX(1))+p.scaleX(1);
%                 rsy = r*(p.scaleY(2)-p.scaleY(1))+p.scaleY(1);
%             else
%                 rsx = rand(s(4),1, 'single')*(p.scaleX(2)-p.scaleX(1))+p.scaleX(1);
%                 rsy = rand(s(4),1, 'single')*(p.scaleY(2)-p.scaleY(1))+p.scaleY(1);
%             end
%             rix = rand(s(4),1, 'single')*(p.shiftX(2)-p.shiftX(1))+p.shiftX(1);
%             riy = rand(s(4),1, 'single')*(p.shiftY(2)-p.shiftY(1))+p.shiftY(1);
%             w = zeros(1,1,6,s(4), 'single');
%         end
%         %rix = rix-(rsx-1)./2;
%         %riy = riy-(rsy-1)./2;

%         w(1,1,1,:) = cosd(ral).*rsx;
%         w(1,1,2,:) = sind(ral).*rsy;
%         w(1,1,3,:) = -sind(ral).*rsx;
%         w(1,1,4,:) = cosd(ral).*rsy;
%         %w(1,1,5,:) = cosd(ral).*rsx.*rix - sind(ral).*rsy.*riy;
%         %w(1,1,6,:) = sind(ral).*rsx.*rix + cos(ral).*rsy.*riy;
        
%         w(1,1,5,:) = rix;
%         w(1,1,6,:) = riy;

%         if all(p.extend==0)
%             top{1} = feval(p.forward, bottom{1}, s, w, len, top{1});
%         else
%             tmpp = feval(p.forward, bottom{1}, s, w, len, top{1});
%             if opts.gpuMode
%                 top{1} = gpuArray.zeros(p.extend(1)+s(1),p.extend(2)+s(2),1,s(4),'single');
%             else
%                 top{1} = zeros(p.extend(1)+s(1),p.extend(2)+s(2),1,s(4),'single');
%             end
%             randPosx = randi(p.extend(1)+1,1,s(4));
%             randPosy = randi(p.extend(2)+1,1,s(4));
%             for i=1:s(4)
%                 top{1}(randPosy(i):(randPosy(i)+s(1)-1), randPosx(i):(randPosx(i)+s(2)-1),1,i) = tmpp(:,:,1,i);
%             end
%         end
        
%         %top{1}(randperm(len, floor(len*0.05))) = randi(255,1,floor(len*0.05));
%         if numel(l.top)==2 %output affine parameters
%             ww = reshape(w,2,3,[]);
%             if opts.gpuMode
%                 w = gpuArray.zeros(3,3,s(4),'single');
%                 w(1:2,1:3,:) = ww;
%                 w(3,3,:) = 1.0;
%                 w = pagefun(@inv,w);
%                 w = w(1:2,1:3,:);
%                 w = reshape(w,1,1,6,[]);
%                 top{2} = w;
%             end
%         end
%     end


%     function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff)
%         %numel(bottom_diff) = numel(bottom), numel(weights_diff) = numel(weights)
%         bottom_diff = {[]};
%     end

% end
