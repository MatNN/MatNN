function runFlow(obj, flowOpts, flowLayerIDs, currentRepeatTimes, globalIterNum, currentIter)
%RUNFLOW
%
% GUIDELINE
% 1. DO NOT perform add/remove vars or layers statements in layers 
%    which are execuated by this function.
%    eg. addlayers in your layer.forward(). You should .addlayer in setup()
%
data = obj.data;

numGpus  = numel(obj.gpu);
[involvedDataID, inoutIds, outputIDs, ~] = obj.findDataID(data, obj.layers(flowLayerIDs));

accumulateOutBlobs = cell(size(outputIDs));
accumulateOutBlobsNum = numel(accumulateOutBlobs);

% Create options
flowOpts.disableDropout = flowOpts.lr == 0;
flowOpts.inParallel = obj.inParallel;
flowOpts.gpu = obj.gpu;

% Find initial weight learning rate ~= 0 to update them
needToUpdatedWeightsInd = find(data.lr>0 & involvedDataID);

if numGpus > 0
    dzdy = gpuArray.ones(1, 'single');
else
    dzdy = single(1);
end

% Calculate total iteration number, current flow total iteration number
currentFlowTotalIter = (currentRepeatTimes-1)*flowOpts.iter+currentIter;
flowTime = tic;
pstart = tic;
%data.count(involvedDataID) = data.count(involvedDataID)*0;
data.clearCount(involvedDataID)
if obj.clearDataOnStart
    data.clear(inoutIds);
end

for i=needToUpdatedWeightsInd
    if isempty(obj.data.momentum{i})
        data.momentum{i} = data.val{i}.*single(0);
    end
end

% Running time constant variables
% -------------------------------
count = 1;
count_per_display = 1;
ss = 1:flowOpts.iter_size;
sb = 1:accumulateOutBlobsNum;
weightsNUMEL = [];
if flowOpts.gpu
    gFun = obj.solverGPUFun;
end

% Saving stat
origPreserve = data.preserve(outputIDs);
origCM = data.conserveMemory;
data.preserve(outputIDs) = true;
data.conserveMemory = flowOpts.conserveMemory;
% -------------------------------


for t = currentIter:flowOpts.iter
    % set learning rate
    learningRate = flowOpts.lrPolicy(globalIterNum, currentFlowTotalIter, flowOpts.lr, flowOpts.lrGamma, flowOpts.lrPower, flowOpts.lrSteps);

    % set currentIter
    flowOpts.currentIter = t;
    for s=ss
        % evaluate CNN
        flowOpts.accumulate = s > 1;
        flowOpts.freezeDropout = s > 1;
        obj.opts = flowOpts; % important

        % forward
        %data.clearCount(inoutIds);

        for i = flowLayerIDs
            obj.layers{i}.forward();
        end
        % backward
        if flowOpts.lr ~= 0
            data.diff(outputIDs) = {dzdy};
            for i = flowLayerIDs(end:-1:1)
                obj.layers{i}.backward();
            end
        end

        % accumulate backprop errors
        % assume all output blobs are loss-like blobs
        for ac = sb
            if isempty(accumulateOutBlobs{ac})
                accumulateOutBlobs{ac} = data.val{outputIDs(ac)};
            else
                accumulateOutBlobs{ac} = accumulateOutBlobs{ac} + data.val{outputIDs(ac)};
            end
        end
    end
    %data.clearCount(needToUpdatedWeightsInd);

    if flowOpts.lr ~= 0
        if flowOpts.inParallel
            for nz=1:numel(data.diff(needToUpdatedWeightsInd))
                data.diff{needToUpdatedWeightsInd(nz)} = gplus(data.diff{needToUpdatedWeightsInd(nz)});
            end
        end
        if numGpus == 0
            obj.updateWeightCPU(data, learningRate, flowOpts.decay, flowOpts.momentum, flowOpts.iter_size, needToUpdatedWeightsInd);
        else
            if ~isempty(needToUpdatedWeightsInd)
                if isempty(weightsNUMEL)
                    weightsNUMEL = zeros(size(data.val),'single');
                    for i=needToUpdatedWeightsInd
                        weightsNUMEL(i) = numel(data.val{i});
                    end
                    gFun.GridSize = ceil( max(weightsNUMEL)/obj.MaxThreadsPerBlock );
                end
                obj.updateWeightGPU(data, learningRate, flowOpts.decay, flowOpts.momentum, flowOpts.iter_size, needToUpdatedWeightsInd, gFun, weightsNUMEL);
            else
                %warning('No need to update weights.');
            end
        end
    end

    % Print learning statistics
    if mod(count, flowOpts.displayIter) == 0 || (count == 1 && flowOpts.showFirstIter) || t==flowOpts.iter
        if obj.showDate
            dStr = datestr(now, '[mmdd HH:MM:SS.FFF ');
        else
            dStr = '';
        end
        if flowOpts.lr ~= 0
            preStr = [dStr, sprintf('Lab%d] F%d/G%d lr(%g) ', labindex, currentFlowTotalIter, globalIterNum, learningRate)];
        else
            preStr = [dStr, sprintf('Lab%d] F%d/G%d ', labindex, currentFlowTotalIter, globalIterNum)];
        end
        
        for ac = 1:accumulateOutBlobsNum
            if ~isempty(accumulateOutBlobs{ac})
                fprintf(preStr);
                fprintf('%s(%.6g) ', data.names{outputIDs(ac)}, accumulateOutBlobs{ac}./(flowOpts.iter_size*count_per_display)); % this is a per-batch avg., not output avg.
            end
            if ac~=numel(accumulateOutBlobs)
                fprintf('\n');
            end
        end

        flowTime = toc(flowTime);
        fprintf('%.2fs(%.2f iter/s)\n', flowTime, count_per_display/flowTime);
        flowTime = tic;
        accumulateOutBlobs = cell(size(outputIDs));
        count_per_display = 0;
    end

    % Save model
    if ~isempty(flowOpts.numToSave) && mod(count, flowOpts.numToSave) == 0
        % only one worker can save the model
        if numGpus>1, labBarrier(); end
        if labindex==1, obj.save(obj.saveFilePath(globalIterNum)); end
        if numGpus>1, labBarrier(); end
    end

    count = count+1;
    count_per_display = count_per_display+1;
    globalIterNum = globalIterNum+1;
    currentFlowTotalIter = currentFlowTotalIter+1;
end

% Restore Stat
% Saving stat
data.preserve(outputIDs) = origPreserve;
data.conserveMemory = origCM;



obj.globalIter = globalIterNum;
toc(pstart);

end
