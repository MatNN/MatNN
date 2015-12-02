classdef SoftMaxLoss < nn.layers.template.LossLayer

    properties (Access = {?nn.layers.template.BaseLayer, ?nn.layers.template.LossLayer})
        threshold = realmin('single');
        batchSize = 1;
        ind         = [];
        N           = [];
        ll          = [];
        accumulateN = single(0);
        accumulateL = single(0);
        forwardHandle;
        backwardHandle;
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
            v.forwardHandle = 'createForwardHandle';
            v.backwardHandle = 'createBackwardHandle';
        end
        
        function loss = f(obj, in, label, varargin) % varargin{1} = label_weight
            %reshape
            resSize    = nn.utils.size4D(in);
            labelSize  = nn.utils.size4D(label);
            if resSize(4) == numel(label)
                label = reshape(label, [1, 1, 1 resSize(4)]);
                label = repmat(label, [resSize(1), resSize(2)]);
            else
                if ~isequal(resSize([1,2,4]), labelSize([1,2,4]))
                    error('Label size must be Nx1, 1xN or HxWx1xN.');
                end
            end

            [ind_, ll_] = calc_internalPara(resSize, label, obj.params.loss.labelIndex_start);

            % Do softmax
            y = exp( bsxfun(@minus, in, max(in, [], 3)) );
            y = bsxfun(@rdivide, y, sum(y,3));
            y = y(ind_);

            if numel(varargin)==1
                label_weight = varargin{1}(ll_);
                obj.N = sum(label_weight(:));
                loss = -sum( label_weight .* log(max(y,obj.threshold)))/obj.N;
            else
                obj.N = resSize(1)*resSize(2)*resSize(4);
                loss = -sum(log(max(y,obj.threshold)))/obj.N;
            end
            obj.batchSize = resSize(4);
            obj.ind = ind_;
            obj.ll = ll_;

        end
        function loss = gf(obj, in, label, varargin) % varargin{1} = label_weight
            resSize    = nn.utils.size4D(in);
            labelSize  = nn.utils.size4D(label);

            nnn = resSize(1)*resSize(2)*resSize(4);
            obj.forwardHandle.GridSize = ceil( nnn/obj.MaxThreadsPerBlock );
            loss = single(0);

            if numel(varargin)==1
                loss = feval(obj.forwardHandle, in, resSize, label, varargin{1}, labelSize, obj.threshold, obj.params.loss.labelIndex_start, true, loss)/nnn;
            else
                loss = feval(obj.forwardHandle, in, resSize, label, single(1), labelSize, obj.threshold, obj.params.loss.labelIndex_start, false, loss)/nnn;
            end
            obj.batchSize = resSize(4);

        end
        % must call .f() first
        function in_diff = b(obj, in, out_diff, varargin)
            y = exp( bsxfun(@minus, in, max(in, [], 3)) );
            y = bsxfun(@rdivide, y, max(sum(y,3), obj.threshold));
            y(obj.ind)  = y(obj.ind)-1;
            if numel(varargin)==1
                in_diff = bsxfun(@times, varargin{1}, (y.*out_diff)/obj.N );
            else
                in_diff = (y.*out_diff)/obj.N;
            end
        end
        function in_diff = gb(obj, in, label, out_diff, varargin)
            resSize    = nn.utils.size4D(in);
            labelSize  = nn.utils.size4D(label);
            obj.backwardHandle.GridSize = ceil( numel(in)/obj.MaxThreadsPerBlock );
            in_diff = in.*single(0);
            if numel(varargin)==1
                in_diff = feval(obj.backwardHandle, in, resSize, label, varargin{1}, labelSize, out_diff, obj.threshold, obj.params.loss.labelIndex_start, true, in_diff);
            else
                in_diff = feval(obj.backwardHandle, in, resSize, label, single(1), labelSize, out_diff, obj.threshold, obj.params.loss.labelIndex_start, false, in_diff);
            end
        end

        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            if opts.gpuMode
                loss = obj.params.loss.loss_weight * obj.gf(data.val{l.bottom});
            else
                loss = obj.params.loss.loss_weight * obj.f(data.val{l.bottom});
            end
            
            if obj.params.loss.accumulate
                if opts.currentIter == 1
                    obj.accumulateL = obj.accumulateL*0;
                    obj.accumulateN = obj.accumulateN*0;
                end
                obj.accumulateL = obj.accumulateL + loss*obj.batchSize;
                obj.accumulateN = obj.accumulateN + obj.batchSize;
                loss = obj.accumulateL/obj.accumulateN;
            end
            data.val{l.top} = loss;
        end
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            p = obj.params.loss;
            if opts.gpuMode
                if numel(l.bottom) == 3
                    bd = p.loss_weight * obj.gb(data.val{l.bottom(1)}, data.val{l.bottom(2)}, data.diff{l.top}, data.val{l.bottom(3)});
                else
                    bd = p.loss_weight * obj.gb(data.val{l.bottom(1)}, data.val{l.bottom(2)}, data.diff{l.top});
                end
            else
                if numel(l.bottom) == 3
                    bd = p.loss_weight * obj.b(data.val{l.bottom(1)}, data.diff{l.top}, data.val{l.bottom(3)});
                else
                    bd = p.loss_weight * obj.b(data.val{l.bottom(1)}, data.diff{l.top});
                end
            end
            
            if ~isa(bd,'gpuArray') && opts.gpuMode
                bd = gpuArray(bd);
            end
            if numel(l.bottom) == 3
                data = nn.utils.accumulateData(opts, data, l, bd, [], []);
            else
                data = nn.utils.accumulateData(opts, data, l, bd, []);
            end
        end

        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            resSize = inSizes{1};
            ansSize = inSizes{2};
            if ~isequal(resSize(4),prod(ansSize))
                if ~(isequal(resSize([1,2,4]), ansSize([1,2,4])) && ansSize(3) == 1) && ~(isequal(resSize(4), ansSize(4)) && isequal(ansSize(1:3),[1 1 1]))
                    error('Label size must be Nx1, 1xN or HxWx1xN.');
                end
            end
            outSizes = {[1,1,1,1]};
        end
        function setParams(obj, l)
            obj.setParams@nn.layers.template.BaseLayer(l);
            obj.threshold = obj.params.loss.threshold;
        end
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.LossLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.bottom)>=2 && numel(l.bottom)<=3);
            assert(numel(l.top)==1);
            if opts.gpuMode
                obj.accumulateN = gpuArray.zeros(1,1,'single');
                obj.accumulateL = gpuArray.zeros(1,1,'single');
                obj.createGPUFun(inSizes{1});
            end
        end

        function createGPUFun(obj, sampleSize)
            obj.forwardHandle = obj.createForwardHandle();
            obj.backwardHandle = obj.createBackwardHandle();
        end
        function h = createForwardHandle(obj, varargin)
            if isempty(varargin) || strcmpi(varargin{1}, 'GPU')
                mf = fileparts(mfilename('fullpath'));
                ptxp = fullfile(mf, 'private', 'softmaxloss.ptx');
                cup = fullfile(mf, 'private', 'softmaxloss.cu');
                h = nn.utils.gpu.createHandle(1, ptxp, cup, 'SoftMaxLossForward');
            elseif strcmpi(varargin{1}, 'CPU')
                h = [];
            end
        end
        function h = createBackwardHandle(obj, varargin)
            if isempty(varargin) || strcmpi(varargin{1}, 'GPU')
                mf = fileparts(mfilename('fullpath'));
                ptxp = fullfile(mf, 'private', 'softmaxloss.ptx');
                cup = fullfile(mf, 'private', 'softmaxloss.cu');
                h = nn.utils.gpu.createHandle(1, ptxp, cup, 'SoftMaxLossBackward');
            elseif strcmpi(varargin{1}, 'CPU')
                h = [];
            end
        end

    end
    

end

function [index, labelQ] = calc_internalPara(resSize, label, labelIndex_start)
    % Calc correspond indices
    labelQ  = label >= labelIndex_start;
    index   = (1:numel(label))' -1;
    index   = index(labelQ(:));
    %index   = find(labelQ(:))-1;
    labelQ  = index+1;
    label   = label(:);
    label   = label(labelQ)-labelIndex_start; % DO NOT ADD 1, we calc zero-based ind.
    index   = mod(index, resSize(1)*resSize(2)) + ...
              label*resSize(1)*resSize(2) + ...
              floor(index/(resSize(1)*resSize(2)))*resSize(1)*resSize(2)*resSize(3) + ...
              1; % ADD 1 to match matlab 1-based ind
end