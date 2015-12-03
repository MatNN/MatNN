classdef SmoothL1Loss < nn.layers.template.LossLayer


    properties (Access = {?nn.layers.template.BaseLayer, ?nn.layers.template.LossLayer})
        threshold = realmin('single');
        batchSize = 1;
        N           = [];
        df          = [];
        accumulateN = single(0);
        accumulateL = single(0);
    end
    

    methods
        function v = propertyDevice(obj)
            v = obj.propertyDevice@nn.layers.template.LossLayer();
            v.threshold = 2;
            v.batchSize = 0;
            v.N           = 2;
            v.df          = 2;
            v.accumulateN = 2;
            v.accumulateL = 2;
        end
        function loss = f(obj, in1, in2, varargin) % varargin{1} = label_weight
            bSize = nn.utils.size4D(in1);
            obj.N = bSize(1)*bSize(2)*bSize(4);

            if numel(varargin)==1
                obj.df = (in1-in2).*varargin{1};
            else
                obj.df = in1-in2;
            end
            loss = obj.df;
            ind  = abs(obj.df) < 1;
            loss(ind)  = (loss(ind).^2)/2;
            loss(~ind) = abs(loss(~ind))-0.5;
            loss = sum(abs(loss(:)))/obj.N;

            obj.batchSize = bSize(4);

        end

        % must call .f() first
        function [in1_diff, in2_diff] = b(obj, out_diff)
            der = obj.df;
            ind = abs(obj.df)>=1;
            der(ind) = sign(der(ind));
            der = der .* (out_diff/obj.N);
            in1_diff = der;
            in2_diff = -der;
        end
        function forward(obj, nnObj, l, opts, data, net)
            loss = obj.params.loss.loss_weight * obj.f(data.val{l.bottom});
            
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
        function backward(obj, nnObj, l, opts, data, net)
            p = obj.params.loss;
            if numel(l.bottom) == 3
                [bd1, bd2] = obj.b(data.diff{l.top}, data.val{l.bottom(3)});
            else
                [bd1, bd2] = obj.b(data.diff{l.top});
            end
            bd1 = bd1*p.loss_weight;
            bd2 = bd2*p.loss_weight;

            if ~isa(bd1,'gpuArray') && opts.gpuMode
                bd1 = gpuArray(bd1);
                bd2 = gpuArray(bd2);
            end
            if numel(l.bottom) == 3
                nn.utils.accumulateData(opts, data, l, bd1, bd2, []);
            else
                nn.utils.accumulateData(opts, data, l, bd1, bd2);
            end
        end
        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            assert(isequal(inSizes{1},inSizes{2}), 'bottom1 and bottom2 must have the same sizes.');
            if numel(inSizes)==3
                assert(isequal(inSizes{1},inSizes{3}), 'bottom1 and bottom3 must have the same sizes.');
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
            end
        end
    end
end