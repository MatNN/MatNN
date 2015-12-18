classdef EuclideanLoss < nn.layers.template.LossLayer

    properties (SetAccess = {?nn.BaseObject}, GetAccess = public)
        threshold = realmin('single');
        batchSize = 1;
        N         = [];
        accumulateN = single(0);
        accumulateL = single(0);
    end
    
    properties (Access = protected, Transient)
        distance;
    end

    methods
        function v = propertyDevice(obj)
            v = obj.propertyDevice@nn.layers.template.LossLayer();
            v.threshold = 2;
            v.batchSize = 0;
            v.N           = 2;
            v.accumulateN = 2;
            v.accumulateL = 2;
        end
        function loss = f(obj, in, label, varargin) % varargin{1} = label_weight
            %reshape
            resSize    = nn.utils.size4D(in);
            labelSize  = nn.utils.size4D(label);
            if resSize(3)>1 && labelSize(3)==1
                d = in;
                for i = 1:resSize(3)
                    d(:,:,i,:) = d(:,:,i,:) - (label==i);
                end
                d2 = d.^2;
                obj.distance = d;
            elseif resSize(3) == labelSize(3)
                obj.distance = in-label;
                d2 = obj.distance.^2;
            else
                error('size mismatch.');
            end
            if numel(varargin)==1
                obj.N = sum(varargin{1}(:));
                loss = 0.5 * sum( varargin{1}.*d2(:) )/obj.N;
            else
                obj.N = resSize(1)*resSize(2)*resSize(4);
                loss = 0.5 * sum(d2(:))/obj.N;
            end
            obj.batchSize = resSize(4);

        end

        % must call .f() first
        function [in_diff, label_diff] = b(obj, out_diff, varargin)
            in_diff = out_diff .* obj.distance ./ obj.N;
            label_diff = -in_diff;
        end

        function forward(obj)
            lst = obj.params.loss.labelIndex_start;
            net = obj.net;
            data = net.data;
            if numel(obj.bottom) == 3
                loss = obj.params.loss.loss_weight * obj.f(data.val{obj.bottom(1)}, data.val{obj.bottom(2)}-lst+1, data.val{obj.bottom(3)});
            else
                loss = obj.params.loss.loss_weight * obj.f(data.val{obj.bottom(1)}, data.val{obj.bottom(2)}-lst+1);
            end
            
            if obj.params.loss.accumulate
                if net.opts.currentIter == 1
                    obj.accumulateL = obj.accumulateL*0;
                    obj.accumulateN = obj.accumulateN*0;
                end
                obj.accumulateL = obj.accumulateL + loss*obj.batchSize;
                obj.accumulateN = obj.accumulateN + obj.batchSize;
                loss = obj.accumulateL/obj.accumulateN;
            end
            data.val{obj.top} = loss;
            data.forwardCount(obj.bottom, obj.top);
        end
        function backward(obj)
            p = obj.params.loss;
            net = obj.net;
            data = net.data;
            if numel(obj.bottom) == 3
                [bd1,bd2] = obj.b(data.diff{obj.top}, data.val{obj.bottom(3)});
            else
                [bd1,bd2] = obj.b(data.diff{obj.top});
            end

            bd1 = bd1 * p.loss_weight;
            bd2 = bd2 * p.loss_weight;
            
            if ~isa(bd1,'gpuArray') && net.opts.gpu
                bd1 = gpuArray(bd1);
                bd2 = gpuArray(bd2);
            end
            if numel(obj.bottom) == 3
                data.backwardCount(obj.bottom,  obj.top, bd1, bd2, []);
            else
                data.backwardCount(obj.bottom,  obj.top, bd1, bd2);
            end
        end

        function outSizes = outputSizes(obj, inSizes)
            resSize = inSizes{1};
            ansSize = inSizes{2};
            assert( isequal(ansSize, resSize) | (isequal(ansSize([1,2,4]), resSize([1,2,4])) && ansSize(3)==1) );
            % Label size must be HxWxCxN or HxWx1xN
            outSizes = {[1,1,1,1]};
        end
        function setParams(obj)
            obj.setParams@nn.layers.template.BaseLayer();
            obj.threshold = obj.params.loss.threshold;
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.LossLayer(inSizes);
            assert(numel(obj.bottom)>=2 && numel(obj.bottom)<=3);
            assert(numel(obj.top)==1);
            if obj.net.opts.gpu
                obj.accumulateN = gpuArray.zeros(1,1,'single');
                obj.accumulateL = gpuArray.zeros(1,1,'single');
            end
        end
    end
end