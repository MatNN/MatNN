classdef LogLoss < nn.layers.template.LossLayer

    properties (Access = {?nn.layers.template.BaseLayer, ?nn.layers.template.LossLayer})
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

        % Forward function for training/testing routines
        function [data, net] = forward(obj, nnObj, l, opts, data, net)
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
        % Backward function for training/testing routines
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            p = obj.params.loss;
            if numel(l.bottom) == 3
                bd = p.loss_weight * obj.b(data.val{l.bottom(1)}, data.val{l.bottom(2)}, data.diff{l.top}, data.val{l.bottom(3)});
            else
                bd = p.loss_weight * obj.b(data.val{l.bottom(1)}, data.val{l.bottom(2)}, data.diff{l.top});
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

        % Calc Output sizes
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
            end
        end

    end
    

end










% function o = logisticLoss(networkParameter)
% %LOGISTICLOSS 
% %
% % NOTICE
% %   label index starts from 0 (compatible with other NN tools)
% %   you can specify begining index from parameter

% o.name         = 'LogisticLoss';
% o.generateLoss = true;
% o.setup        = @setup;
% o.forward      = @forward;
% o.backward     = @backward;


% default_logisticLoss_param = {
%     'labelIndex_start' single(0)    ...
%            'threshold' single(1e-4)
% };

% % Save Forward result for faster computation
% resultBlob = [];
% ind        = [];
% N          = [];
% ll         = [];

%     function [resource, topSizes, param] = setup(l, bottomSizes)
%         resource = {};

%         if isfield(l, 'logisticLoss_param')
%             wp = nn.utils.vararginHelper(default_logisticLoss_param, l.logisticLoss_param);
%         else
%             wp = nn.utils.vararginHelper(default_logisticLoss_param, default_logisticLoss_param);
%         end
%         param.logisticLoss_param = wp;


%         assert(numel(l.bottom)==2);
%         assert(numel(l.top)==1);
        

%         resSize = bottomSizes{1};
%         ansSize = bottomSizes{2};
%         if ~isequal(resSize(4),prod(ansSize))
%             if ~(isequal(resSize([1,2,4]), ansSize([1,2,4])) && ansSize(3) == 1) && ~(isequal(resSize(4), ansSize(4)) && isequal(ansSize(1:3),[1 1 1]))
%                 error('Label size must be Nx1, 1xN or HxWx1xN.');
%             end
%         end
%         topSizes = {[1, 1, 1, 1]};

%     end
%     function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
%         %resultBlob = max(bottom{1}, l.logisticLoss_param.threshold);
%         resultBlob = bottom{1}+l.logisticLoss_param.threshold;

%         resSize = nn.utils.size4D(resultBlob);
%         labelSize = nn.utils.size4D(bottom{2});

%         if resSize(4) == numel(bottom{2})
%             label = reshape(bottom{2}, [1, 1, 1 resSize(4)]) ;
%             label = repmat(label, [resSize(1), resSize(2)]) ;
%         else
%             if ~isequal(resSize([1,2,4]), labelSize([1,2,4]))
%                 error('Label size must be Nx1, 1xN or HxWx1xN.');
%             else
%                 label = bottom{2};
%             end
%         end
%         ll = label >= l.logisticLoss_param.labelIndex_start;
%         %label = label(ll) - l.logisticLoss_param.labelIndex_start;
%         N = resSize(1)*resSize(2)*resSize(4);
%         %ind = find(ll)-1;
%         %ind = 1 + mod(ind, N)  ...
%         %        + N * label(:) ...
%         %        + N*resSize(3) * floor(ind/N);

%         if opts.gpuMode
%             ind = gpuArray.false(resSize);
%         else
%             ind = false(resSize);
%         end
%         for i=1:resSize(3)
%             ind(:,:,i,:) = label == i + (l.logisticLoss_param.labelIndex_start - 1);
%         end

%         top{1} = -sum(log(resultBlob(ind)))/N;

%     end
%     function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff)
%         dzdx = -bsxfun(@rdivide, bsxfun(@times, ll, top_diff{1}/N), resultBlob);

%         % only ground truth label are correct, set others to zero
%         outdzdx = dzdx*0; % faster than zeros(size(dzdx)); ?
%         outdzdx(ind) = dzdx(ind);
%         bottom_diff = {outdzdx,[]};

%     end
% end