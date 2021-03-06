classdef ContrastiveLoss < nn.layers.template.LossLayer

    properties (SetAccess = protected, Transient)
        default_contrastiveLoss_param = { 'margin' single(1) };
    end

    properties (SetAccess = {?nn.BaseObject}, GetAccess = public)
        threshold = realmin('single');
        batchSize = 1;
        N         = [];
        accumulateN = single(0);
        accumulateL = single(0);
    end

    methods (Access = protected)
        function modifyDefaultParams(obj)
            obj.default_loss_param = {
                  'threshold' single(1e-4) ...
                 'accumulate' true  ... % report per-batch loss (false) or avg loss (true), this does not affect backpropagation
                'loss_weight' single(1) ... % a multiplier to the loss
            };
        end
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
        function loss = f(obj, in1, in2, y, margin)
            resSize    = nn.utils.size4D(in1);
            y = reshape(y, 1,1,1,resSize(4));
            
            d_ = in1-in2;
            d = sum((d_).^2, 3);
            obj.N = resSize(1)*resSize(2)*resSize(4);
            %E = 0.5 * sum(  y.*d + (1-y).*max(l.contrastiveLoss_param.margin - d, single(0))  )/size(bottom{1},4);
            loss = 0.5 * sum(  y.*d + (1-y).*max(margin-sqrt(d), single(0)).^2  )/obj.N;

            obj.batchSize = resSize(4);
        end

        % must call .f() first
        function [in1_diff, in2_diff] = b(obj, in1, in2, y, margin, out_diff)
            d_ = in1-in2;
            d = sum((d_).^2, 3);
            y = reshape(y, 1,1,1,numel(y));

            rightTerm = sqrt(d);
            m_d = margin - rightTerm;
            rightTerm = m_d ./ (rightTerm + obj.threshold);
            rightTerm = bsxfun(@times, d_, rightTerm .* (m_d>0));
            in1_diff = out_diff * (bsxfun(@times, d_+rightTerm, y) - rightTerm) / obj.N;
            in2_diff = -in1_diff;
        end
        function forward(obj)
            net = obj.net;
            data = net.data;
            loss = obj.params.loss.loss_weight * obj.f(data.val{obj.bottom}, obj.params.contrastiveLoss.margin);
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
            net = obj.net;
            data = net.data;
            p = obj.params.loss;
            [bd1,bd2] = obj.b(data.val{obj.bottom}, obj.params.contrastiveLoss.margin, data.diff{obj.top});

            bd1 = bd1 * p.loss_weight;
            bd2 = bd2 * p.loss_weight;
            
            if ~isa(bd1,'gpuArray') && net.opts.gpu
                bd1 = gpuArray(bd1);
                bd2 = gpuArray(bd2);
            end
            %nn.utils.accumulateDiff(data, obj.bottom(1:2),  obj.top, bd1, bd2, []);
            data.backwardCount(obj.bottom, obj.top, bd1, bd2, []);
        end
        function outSizes = outputSizes(~, inSizes)
            assert( isequal(inSizes{1}, inSizes{2}) );
            assert( inSizes{1}(4) == prod(inSizes{3}) );
            % similarity input size must be 1x1x1xN, or 1xN or Nx1
            outSizes = {[1,1,1,1]};
        end
        function setParams(obj)
            obj.setParams@nn.layers.template.BaseLayer();
            obj.threshold = obj.params.loss.threshold;
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.LossLayer(inSizes);
            assert(numel(obj.bottom)==3);
            assert(numel(obj.top)==1);
            if obj.net.opts.gpu
                obj.accumulateN = gpuArray.zeros(1,1,'single');
                obj.accumulateL = gpuArray.zeros(1,1,'single');
            end
        end
    end
end