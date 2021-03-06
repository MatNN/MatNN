classdef Accuracy < nn.layers.template.BaseLayer

    % Default parameters
    properties (SetAccess = protected, Transient)
        default_accuracy_param = {
            'labelIndex_start' single(0) ...
                'meanClassAcc' false ... % true: compute avg.acc and mean class acc.; false: per-batch acc
                  'accumulate' false ...
        };
        % 'accumulate' property is NOT available when you call .f() .b() directly.
    end

    % variables (not computed every time, eg. once at launch)
    properties (Access = {?nn.BaseObject})
        counting = single(0);
        perClassArea = single(0);
        perClassAcc  = single(0);
        accumulateAcc = single(0);
        accumulate = false;
    end
    
    methods
        function v = propertyDevice(obj)
            v = obj.propertyDevice@nn.layers.template.BaseLayer();
            v.counting = 0;
            v.perClassArea = 2;
            v.perClassAcc  = 2;
            v.accumulateAcc = 2;
            v.accumulate = 0;
        end

        function varargout = f(obj, in, label, labelIndex_start, doMeanClass, varargin) %varargin{1} = currentIter
            resSize = nn.utils.size4D(in);
            if resSize(4) == numel(label) && size(label,4) ~= resSize(4)
                label = reshape(label,1,1,1,[]);
            end

            if resSize(3) > 1
                [~, argMax] = max(in, [], 3);
                argMax = argMax -1 + labelIndex_start;
                argMax(label < labelIndex_start) = 0; % Important: we just compare class ID >= "labelIndex_start", so class ID < "labelIndex_start" will be marked as correct.
                label(label  < labelIndex_start) = 0; % this may not an necessary step, but for potentially risks.
                k = argMax == label;

                if doMeanClass
                    if obj.accumulate
                        if numel(varargin) == 1 && varargin{1} == 1
                            obj.perClassAcc  = obj.perClassAcc.*single(0);
                            obj.perClassArea = obj.perClassArea.*single(0);
                            obj.counting = obj.counting*single(0);
                        end
                        obj.counting = obj.counting + resSize(4);
                        for i=1:resSize(3)
                            correctLabelInd = i-1+labelIndex_start;
                            mask = label==correctLabelInd;
                            obj.perClassAcc(i)  = obj.perClassAcc(i)+sum(argMax(mask)==correctLabelInd);
                            obj.perClassArea(i) = obj.perClassArea(i)+sum(mask(:));
                        end
                    else
                        obj.perClassAcc = obj.perClassAcc.*0;
                        obj.perClassArea = obj.perClassArea.*0;
                        for i=1:resSize(3)
                            correctLabelInd = i-1+labelIndex_start;
                            mask = label==correctLabelInd;
                            obj.perClassAcc(i)  = sum(argMax(mask)==correctLabelInd);
                            obj.perClassArea(i) = sum(mask(:));
                        end
                    end
                    nonZeroArea = obj.perClassArea~=0;
                    varargout{2} = mean(obj.perClassAcc(nonZeroArea)./obj.perClassArea(nonZeroArea));
                end
            else
                k = bsxfun(@eq, in, label);
            end
            if doMeanClass
                varargout{1} = sum(obj.perClassAcc)/sum(obj.perClassArea);
            else
                if obj.accumulate
                    if numel(varargin) == 1 && varargin{1} == 1
                        obj.accumulateAcc = obj.accumulateAcc.*0;
                        obj.counting = obj.counting*0;
                    end
                    obj.counting = obj.counting+resSize(4);
                    obj.accumulateAcc = obj.accumulateAcc+(sum(k(:))/sum(sum(sum(label >= labelIndex_start))))*resSize(4);
                    varargout{1} = obj.accumulateAcc/obj.counting;
                else
                    varargout{1} = sum(k(:))/sum(sum(sum(label >= labelIndex_start)));%*size(bottom{1},4); %don't divide N here, because train.m will do it for us
                end
                
            end
        end
        function in_diff = b(obj, in, out, out_diff)
            in_diff = [];
        end
        function varargout = gf(obj, in, label, doMeanClass, varargin)
            if doMeanClass
                [varargout{1}, varargout{2}] = obj.f(in, label, doMeanClass,varargin{:});
            else
                varargout{1} = obj.f(in, label, doMeanClass,varargin{:});
            end
        end

        function forward(obj)
            p = obj.params.accuracy;
            net = obj.net;
            data = net.data;
            if p.meanClassAcc
                [a, b] = obj.f(data.val{obj.bottom(1)}, data.val{obj.bottom(2)}, p.labelIndex_start, p.meanClassAcc, net.opts.currentIter);
                data.val(obj.top) = {single(a), single(b)};
            else
                data.val{obj.top} = single(obj.f(data.val{obj.bottom(1)}, data.val{obj.bottom(2)}, p.labelIndex_start, p.meanClassAcc, net.opts.currentIter));
            end
            data.forwardCount(obj.bottom, obj.top);
        end
        function backward(obj)
            obj.net.data.backwardCount(obj.bottom,  obj.top, []);
        end

        function outSizes = outputSizes(obj, inSizes)
            resSize = inSizes{1};
            ansSize = inSizes{2};
            if ~isequal(resSize(4),prod(ansSize))
                if ~(isequal(resSize([1,2,4]), ansSize([1,2,4])) && ansSize(3) == 1) && ~(isequal(resSize(4), ansSize(4)) && isequal(ansSize(1:3),[1 1 1]))
                    error('Label size must be Nx1, 1xN, 1x1x1xN or HxWx1xN.');
                end
            end
            obj.perClassArea = zeros(1, resSize(3), 'single');
            obj.perClassAcc  = zeros(1, resSize(3), 'single');
            outSizes = {[1,1,1,1]};
        end
        function setParams(obj)
            obj.setParams@nn.layers.template.BaseLayer();
            obj.accumulate = obj.params.accuracy.accumulate;
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.bottom)==2);
            if obj.params.accuracy.meanClassAcc
                assert(numel(obj.top)==2, 'Accuracy layer will generate two outputs if you set ''meanClassAcc'' to true.');
            else
                assert(numel(obj.top)==1);
            end

            if obj.net.opts.gpu
                obj.perClassArea = gpuArray(obj.perClassArea);
                obj.perClassAcc  = gpuArray(obj.perClassAcc);
            end
        end

    end
    

end