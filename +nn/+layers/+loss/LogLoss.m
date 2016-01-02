classdef LogLoss < nn.layers.template.LossLayer

    properties (SetAccess = {?nn.BaseObject}, GetAccess = public)
        threshold   = realmin('single');
        batchSize   = 1;
        ind         = [];
        N           = [];
        ll          = [];
        accumulateN = single(0);
        accumulateL = single(0);
    end

    methods
        function v = propertyDevice(obj)
            v = obj.propertyDevice@nn.layers.template.LossLayer();
            v.threshold = 2;
            v.batchSize = 0;
            v.ind         = 2;
            v.N           = 2;
            v.ll          = 2;
            v.accumulateN = 2;
            v.accumulateL = 2;
        end
        function loss = f(obj, in, label, varargin) % varargin{1} = label_weight
            %reshape
            resSize    = nn.utils.size4D(in);
            labelSize  = nn.utils.size4D(label);
            if resSize(4) == numel(label) || resSize(1) == numel(label)
                label = reshape(label, [1, 1, 1 resSize(4)]);
                label = repmat(label, [resSize(1), resSize(2)]);
            else
                if ~isequal(resSize([1,2,4]), labelSize([1,2,4]))
                    error('Label size must be Nx1, 1xN or HxWx1xN.');
                end
            end

            obj.calc_internalPara(resSize, label);

            if numel(varargin)==1
                label_weight = varargin{1}(obj.ll);
                obj.N = sum(label_weight(:));
                loss = -sum(log(max(in(obj.ind), obj.threshold)))/obj.N;
            else
                obj.N = resSize(1)*resSize(2)*resSize(4);
                loss = -sum(log(max(in(obj.ind), obj.threshold)))/obj.N;
            end
            obj.batchSize = resSize(4);
        end

        % must call .f() first
        function in_diff = b(obj, in, label, out_diff, varargin)
            dzdx = -bsxfun(@rdivide, bsxfun(@times, label >= obj.params.loss.labelIndex_start, out_diff/obj.N), in);
            % only ground truth label are correct, set others to zero
            in_diff = dzdx*0; % faster than zeros(size(dzdx)); ?
            in_diff(obj.ind) = in_diff(obj.ind);
            if numel(varargin)==1
                in_diff = bsxfun(@times, varargin{1}, in_diff);
            else
                in_diff = (y.*out_diff)/obj.N;
            end
        end

        function calc_internalPara(obj, resSize, label)
            % Calc correspond indices
            labelQ  = label >= obj.params.loss.labelIndex_start;
            index   = (1:numel(label))' -1;
            index   = index(labelQ(:));
            %index   = find(labelQ(:))-1;
            labelQ  = index+1;
            label   = label(:);
            label   = label(labelQ)-obj.params.loss.labelIndex_start; % DO NOT ADD 1, we calc zero-based ind.
            index   = mod(index, resSize(1)*resSize(2)) + ...
                      label*resSize(1)*resSize(2) + ...
                      floor(index/(resSize(1)*resSize(2)))*resSize(1)*resSize(2)*resSize(3) + ...
                      1; % ADD 1 to match matlab 1-based ind
            obj.ind = index;
            obj.ll  = labelQ;
        end

        function forward(obj)
            data = obj.net.data;
            loss = obj.params.loss.loss_weight * obj.f(data.val{obj.bottom});
            
            if obj.params.loss.accumulate
                if opts.currentIter == 1
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
                bd = p.loss_weight * obj.b(data.val{obj.bottom(1)}, data.val{obj.bottom(2)}, data.diff{obj.top}, data.val{obj.bottom(3)});
            else
                bd = p.loss_weight * obj.b(data.val{obj.bottom(1)}, data.val{obj.bottom(2)}, data.diff{obj.top});
            end
            if ~isa(bd,'gpuArray') && net.opts.gpu
                bd = gpuArray(bd);
            end
            if numel(obj.bottom) == 3
                data.backwardCount(obj.bottom,  obj.top, bd, [], []);
            else
                data.backwardCount(obj.bottom,  obj.top, bd, []);
            end
        end

        function outSizes = outputSizes(obj, inSizes)
            resSize = inSizes{1};
            ansSize = inSizes{2};
            if ~isequal(resSize(4),prod(ansSize))
                if ~(isequal(resSize([1,2,4]), ansSize([1,2,4])) && ansSize(3) == 1) && ~(isequal(resSize(4), ansSize(4)) && isequal(ansSize(1:3),[1 1 1]))
                    error('Label size must be Nx1, 1xN or HxWx1xN.');
                end
            end
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
