function runPhase(obj, currentFace, currentRepeatTimes, globalIterNum, currentIter)
    numGpus  = numel(obj.gpus);
    outputBlobID = obj.data.outId.(currentFace);

    accumulateOutBlobs = cell(size(outputBlobID));
    accumulateOutBlobsNum = numel(accumulateOutBlobs);

    optface = obj.pha_opt.(currentFace);

    % Create options for forwardbackward
    optface.disableDropout  = optface.learningRate == 0;
    optface.outputBlobCount = cellfun(@numel, obj.data.connectId.(currentFace));
    optface.name = currentFace;
    optface.gpuMode = numGpus >= 1;
    optface.gpus = obj.gpus;

    % Find initial weight learning rate ~= 0 to update them
    needToUpdatedWeightsInd = find(~obj.net.weightsIsMisc & ~cellfun(@isempty,obj.net.weights));
    if ~isempty(optface.backpropToLayer)
        nw = [];
        for ww = numel(obj.net.phase.(currentFace)):-1:1
            l = obj.net.phase.(currentFace){ww};
            if isfield(l, 'weights')
                nw = [nw, l.weights]; %#ok
            end
            if strcmp(optface.backpropToLayer, l.name)
                break;
            end
        end
        [~,wind] = setdiff(needToUpdatedWeightsInd,nw);
        needToUpdatedWeightsInd(wind) = [];
    end

    if numGpus > 0
        dzdy = gpuArray.ones(1, 'single');
    else
        dzdy = single(1);
    end

    % Calculate total iteration number, current phase total iteration number
    currentPhaseTotalIter = (currentRepeatTimes-1)*optface.numToNext+currentIter;
    obj.clearData();
    count = 1;count_per_display = 1;
    phaseTime = tic;
    pstart = tic;
    net = obj.net;
    data = obj.data;
    obj.net = {};
    obj.data = {};
    net.weightsDiffCount = net.weightsDiffCount*int32(0);
    for t = currentIter:optface.numToNext
        % set learning rate
        learningRate = optface.learningRatePolicy(globalIterNum, currentPhaseTotalIter, optface.learningRate, optface.learningRateGamma, optface.learningRatePower, optface.learningRateSteps);

        % set currentIter
        optface.currentIter = t;

        for s=1:optface.iter_size
            % evaluate CNN
            optface.accumulate = s > 1;
            optface.freezeDropout = s > 1;
            [data,net] = obj.fb(data, net, currentFace, optface, dzdy);
            %obj.fb(net, dzdy, res, optface, currentFace, numGpus >= 1, userRequest);

            % accumulate backprop errors
            % assume all output blobs are loss-like blobs
            for ac = 1:accumulateOutBlobsNum
                if isempty(accumulateOutBlobs{ac})
                    accumulateOutBlobs{ac} = sum(data.val{outputBlobID(ac)}(:));
                else
                    accumulateOutBlobs{ac} = accumulateOutBlobs{ac} + sum(data.val{outputBlobID(ac)}(:));
                end
            end
        end
        net.weightsDiffCount = net.weightsDiffCount*int32(0);

        if optface.learningRate ~= 0
            if numGpus == 0
                net = obj.updateWeightCPU(net, learningRate, optface.weightDecay, optface.momentum, optface.iter_size, needToUpdatedWeightsInd);
                %net = solver.solve(optface, learningRate, net, res, needToUpdatedWeightsInd);
            elseif numGpus == 1
                net = obj.updateWeightGPU(net, learningRate, optface.weightDecay, optface.momentum, optface.iter_size, needToUpdatedWeightsInd);
            else
                labBarrier();
                %accumulate weight gradients from other labs
                %res.dzdw = gop(@(a,b) cellfun(@plus, a,b, 'UniformOutput', false), res.dzdw);
                for nz=1:numel(net.weightsDiff)
                    net.weightsDiff{nz} = gop(@plus, net.weightsDiff{nz});
                end
                %net = solver.solve(optface, learningRate, net, res, needToUpdatedWeightsInd);
                net = obj.updateWeightGPU(net, learningRate, optface.weightDecay, optface.momentum, optface.iter_size, needToUpdatedWeightsInd);
            end
        end


        % Print learning statistics
        if mod(count, optface.displayIter) == 0 || (count == 1 && optface.showFirstIter) || t==optface.numToNext
            if optface.learningRate ~= 0
                preStr = [datestr(now, '[mmdd HH:MM:SS.FFF '), sprintf('Lab%d—%s] F%d/G%d lr(%g) ', labindex, currentFace,currentPhaseTotalIter, globalIterNum, learningRate)];
            else
                preStr = [datestr(now, '[mmdd HH:MM:SS.FFF '), sprintf('Lab%d—%s] F%d/G%d ', labindex, currentFace,currentPhaseTotalIter, globalIterNum)];
            end
            
            for ac = 1:accumulateOutBlobsNum
                if isinf(accumulateOutBlobs{ac})
                    fprintf('\n');
                    error('A blob output = Inf');
                elseif ~isempty(accumulateOutBlobs{ac})
                    fprintf(preStr);
                    fprintf('%s(%.6g) ', data.names{outputBlobID(ac)}, accumulateOutBlobs{ac}./(optface.iter_size*count_per_display)); % this is a per-batch avg., not output avg.
                end
                if ac~=numel(accumulateOutBlobs)
                    fprintf('\n');
                end
            end

            fprintf('%.2fs(%.2f iter/s)\n', toc(phaseTime), count_per_display/toc(phaseTime));
            phaseTime = tic;
            accumulateOutBlobs = cell(size(outputBlobID));
            count_per_display = 0;
        end

        % Save model
        if ~isempty(optface.numToSave) && mod(count, optface.numToSave) == 0
            if numGpus > 1
                labBarrier();
            end
            obj.net = net;
            obj.data = data;
            net = {};
            data = {};
            if numGpus > 1
                labBarrier();
            end
            if labindex == 1 % only one worker can save the model
                obj.save(obj.saveFilePath(globalIterNum));
            end
            if numGpus > 1
                labBarrier();
            end
            net = obj.net;
            data = obj.data;
            obj.net = {};
            obj.data = {};
            
        end

        count = count+1;
        count_per_display = count_per_display+1;
        globalIterNum = globalIterNum+1;
        currentPhaseTotalIter = currentPhaseTotalIter+1;

    end
    obj.net = net;
    obj.data = data;
    obj.globalIter = globalIterNum;
    toc(pstart);
    pause(1);
end