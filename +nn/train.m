function [net, batchStructTrain, batchStructVal] = train(netObj, batchStructTrain, batchStructVal, opts_user)
%TRAIN  (Based on the example code of Matconvnet)
%
%  NOTE
%    provided 'batchStructVal' will fetch all samples and test them one by one
%

opts.numEpochs = [];
opts.numInterations = [];
opts.numToTest = 1; %runs how many Epochs or iterations to test
opts.numToSave = 10; % note, this value must be dvidable by opts.numToTest
opts.displayIter = 10; %Show info every opts.displayIter iterations
opts.batchSize = 256;
opts.numSubBatches = 1;
opts.gpus = [];
opts.computeMode = 'default';

opts.learningRate = 0.001;
opts.learningRateGamma = 0.1;
opts.learningRatePower = 0.75;
opts.learningRateSteps = 1000;
opts.learningRatePolicy = @(currentBatchNumber, lr, gamma, power, steps) lr*(gamma^floor(currentBatchNumber/steps));
opts.weightDecay = 0.0005;
opts.momentum = 0.9;
opts.solver   = @nn.solvers.StochasticGradientDescent;

opts.continue = [] ; % if you specify the saving's iteration/epoch number, then you can load it
opts.expDir = fullfile('data','exp');
opts.conserveMemory = false;
opts.sync = false;
opts.prefetch = false;

opts.plotDiagnostics = false;
opts = vl_argparse(opts, opts_user);

if isempty(batchStructTrain) && isempty(batchStructVal), error('Must specify Train/Val batch struct!!'); end
if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir); end


% opts.numEpochs and opts.numInterations cannot be set at the same time
if isempty(opts.numEpochs) + isempty(opts.numInterations) == 1
    if ~isempty(opts.numEpochs)
        opts.epit = 'Epoch'; %epit means epoch_iteration
    else
        opts.epit = 'Iter';
    end
else
    error('You must set opts.numEpochs OR opts.numInterations.');
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(batchStructTrain) ;

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
    if isempty(gcp('nocreate')),
        parpool('local',numGpus) ;
        spmd, gpuDevice(opts.gpus(labindex)), end
    end
elseif numGpus == 1
    gpuDevice(opts.gpus)
end

if numGpus == 0 && ~strcmp(opts.computeMode, 'default')
    opts.computeMode = 'default';
    warning('Set computeMode to ''default''.');
end
netObj.computeMode(opts.computeMode);
net = netObj.getNet();
disp(['Set to <strong>', opts.computeMode, '</strong> mode']);

% -------------------------------------------------------------------------
%                                           Find train/valid phase layer ID
% -------------------------------------------------------------------------

visitLayerID_train = [];
visitLayerID_valid = [];
outputBlobID_train = [];
outputBlobID_valid = [];
for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'phase')
        if strcmpi(net.layers{i}.phase, 'train')
            visitLayerID_train = [visitLayerID_train, i]; %#ok
            cb = net.blobConnectId(net.layers{i}.top);
            for c = 1:numel(cb)
                if isempty(cb{c})
                    outputBlobID_train = [outputBlobID_train, net.layers{i}.top(c)]; %#ok
                end
            end
        elseif strcmpi(net.layers{i}.phase, 'valid')
            visitLayerID_valid = [visitLayerID_valid, i]; %#ok
            cb = net.blobConnectId(net.layers{i}.top);
            for c = 1:numel(cb)
                if isempty(cb{c})
                    outputBlobID_valid = [outputBlobID_valid, net.layers{i}.top(c)]; %#ok
                end
            end
        else
            error(['Unknown layer phase:', net.layers{i}.phase]);
        end
    else
        cb = net.blobConnectId(net.layers{i}.top);
        for c = 1:numel(cb)
            if isempty(cb{c})
                outputBlobID_train = [outputBlobID_train, net.layers{i}.top(c)]; %#ok
                outputBlobID_valid = [outputBlobID_valid, net.layers{i}.top(c)]; %#ok
            end
        end
        visitLayerID_train = [visitLayerID_train, i]; %#ok
        visitLayerID_valid = [visitLayerID_valid, i]; %#ok
    end
end

% initialize some variable for this function
trainPa.branchedTopID = find(cellfun(@numel, net.blobConnectId) > 1);


% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

% is epoch training or iterations
if strcmpi(opts.epit, 'epoch')
    runTimes = opts.numEpochs;
else
    runTimes = opts.numInterations;
end

%Load exist tmp savings
modelPath = @(ep) fullfile(opts.expDir, sprintf('%s-%s%d.mat', net.name, lower(opts.epit), ep));
if ~isempty(opts.continue)
    if exist(modelPath(opts.continue),'file')
        if opts.continue == runTimes
            fprintf('Load completed training network: %s\n', modelPath(opts.continue)) ;
        else
            fprintf('Resuming by loading %s%d\n', lower(opts.epit), opts.continue) ;
        end
        tmp = load(modelPath(opts.continue), 'net') ;
        
        % Consider that loaded net is the same as our net, so just use their
        % weights
        %netObj.setBaseNet(tmp.net);
        %net = netObj.getNet();

        net.weights = tmp.net.weights;
        net.momentum = tmp.net.momentum;
        clearvars tmp;
    else
        error('Can''t found savings.');
    end
end

startInd = 1;
if ~isempty(opts.continue)
    startInd = opts.continue+1;
end

if opts.numToTest == 0 || numel(opts.numToTest) == 0
    opts.numToTest = opts.numToSave;
end


%saveTimes = numel(startInd:opts.numToTest:(runTimes-1))+1;

% Save current time
startTime = tic;

% setup solver
opts.solver = opts.solver(opts.computeMode, net);

%rngState = rng;
% start training from the last position
lastSavePoint = floor((startInd-1)/opts.numToSave);
for i = startInd:opts.numToTest:(runTimes-1)
    % move CNN to GPU as needed
    if numGpus == 1
        net = nn.utils.movenet(net, 'gpu') ;
    elseif numGpus > 1
        spmd(numGpus)
            net_ = nn.utils.movenet(net, 'gpu') ;
        end
    end

    %iter/epoch range
    epitRange = i:min(i+opts.numToTest-1, runTimes);

    %Generate randomSeed use current rng settings
    %randomSeed = randi(65536,1)-1;

    % train one epoch (or achieved opts.numToTest) and validate
    % process_runs msut accept startIndex and endIndex of iter/epoch
    if numGpus <= 1
        %generate random seed
        %rng(randomSeed);
        [net, batchStructTrain] = process_runs(true, opts, numGpus, net, batchStructTrain, visitLayerID_train, outputBlobID_train, epitRange) ;
        [net, ~] = process_runs(false, opts, numGpus, net, batchStructVal, visitLayerID_valid, outputBlobID_valid, epitRange) ;
    else
        spmd(numGpus)
            %generate random seed
            %rng(randomSeed);
            [net_, batchStructTrain] = process_runs(true, opts, numGpus, net, batchStructTrain, visitLayerID_train, outputBlobID_train, epitRange) ;
            [net, ~] = process_runs(false, opts, numGpus, net, batchStructVal, visitLayerID_valid, outputBlobID_valid, epitRange) ;
        end
    end

    % save model file

    if floor(epitRange(end)/opts.numToSave)~=lastSavePoint || epitRange(end)==runTimes
        if numGpus > 1
            spmd(numGpus)
                net_ = nn.utils.movenet(net_, 'cpu') ;
            end
            net = net_{1} ;
        else
            net = nn.utils.movenet(net, 'cpu') ;
        end
        fprintf('Saving network model to %s ... \n',modelPath(epitRange(end)));
        if ~evaluateMode, save(modelPath(epitRange(end)), 'net'); end
    end
    lastSavePoint = floor(epitRange(end)/opts.numToSave);
end

%restores rng state
%rng(rngState);

% Report training/evaluation time
startTime = toc(startTime);
fprintf('<strong>Total running time:</strong> %.2fs\n', startTime);

% ---- main function's end
end


% -------------------------------------------------------------------------
function  [net, batchStruct] = process_runs(training, opts, numGpus, net, batchStruct, visitLayerID, outputBlobID, epitRange)
% -------------------------------------------------------------------------

    if isempty(batchStruct)
        return;
    end

    if numGpus >= 1
        one = gpuArray(single(1));
    else
        one = single(1);
    end

    if isfield(batchStruct, 'batchNumber') && strcmpi(opts.epit, 'epoch')
        globalBatchNum = batchStruct.batchNumber*(epitRange(1)-1)+1;
        [X,Y] = meshgrid(epitRange,1:batchStruct.batchNumber);
        epitRange = [reshape(X,1,[]); reshape(Y,1,[])]; % this creates a vector like: [1 1 1, 2 2 2, 3 3 3, ... ; 1 2 3 .., 1 2 3 .., 1 2 3 ..]
        rangeNumber = batchStruct.batchNumber;

    else
        rangeNumber = numel(epitRange);
        epitRange = [epitRange; 1:rangeNumber];
        globalBatchNum = epitRange(1);
    end

    if training
        mode = 'training';
    else
        mode = 'validation';
        opts.numToTest = 0;
        
        %opts.batchSize = 1;
        %opts.numSubBatches = 1;
        rangeNumber = batchStruct.batchNumber;
        epitRange = 1:batchStruct.N:batchStruct.m;
        epitRange = [ones(1,numel(epitRange));epitRange];
        globalBatchNum = epitRange(1);
        opts.displayIter = rangeNumber;
    end

    %create options for forwardbackward
    optFB.accumulate = false;
    optFB.conserveMemory = opts.conserveMemory;
    optFB.sync = opts.sync;
    optFB.disableDropout = ~training;
    optFB.freezeDropout = false;
    optFB.visitLayerID = visitLayerID;
    optFB.outputBlobCount = cellfun(@numel, net.blobConnectId);
    optFB.gpuMode = numGpus >= 1;
    optFB.doder = training;

    % find initial weightLR ~= 0 to update them
    needToUpdatedWeightsInd = find(~net.weightsIsMisc & ~cellfun(@isempty,net.weights));


    accumulateOutBlobs = zeros(size(outputBlobID));
    res = [];
    count = 1;
    cumuTrainedDataNumber = 0;
    batchTime = tic ;
    for t = epitRange
        % set learning rate
        learningRate = opts.learningRatePolicy(globalBatchNum, opts.learningRate, opts.learningRateGamma, opts.learningRatePower, opts.learningRateSteps) ;

        % get batch data
        [data, dataN, batchStruct] = nn.batch.fetch(batchStruct, opts.numSubBatches);
        if opts.prefetch
            [~, ~, batchStruct] = nn.batch.fetch(batchStruct, opts.numSubBatches);
        end

        %current subbatch number
        numSubBatches = numel(data);

        % run subbatches
        for s=1:numSubBatches
            % evaluate CNN
            if training, dzdy = one; else, dzdy = []; end
            optFB.accumulate = s ~= 1;
            res = nn.forwardbackward(net, data{s}, dzdy, res, optFB);

            % accumulate training errors
            % assume all output blobs are loss-like blobs
            for ac = 1:numel(accumulateOutBlobs)
                blobRes = double(gather( res.blob{outputBlobID(ac)} ));
                accumulateOutBlobs(ac) = accumulateOutBlobs(ac) + sum(blobRes(:));
            end
        end
        res.dzdwVisited = res.dzdwVisited & false;

        cumuTrainedDataNumber = cumuTrainedDataNumber+dataN;
        if training
            if numGpus <= 1
                net = opts.solver.solve(opts, learningRate, dataN, net, res, needToUpdatedWeightsInd);
            else
                labBarrier();
                %accumulate weights from other labs
                res.dzdw = gop(@(a,b) cellfun(@plus, a,b, 'UniformOutput', false), res.dzdw);
                net = opts.solver.solve(opts, learningRate, dataN, net, res, needToUpdatedWeightsInd);
            end
            % print learning statistics
            
            if mod(count, opts.displayIter) == 0 || count == 1
                preStr = sprintf('LabNo.%d - %s: %s %d (%d/%d), lr = %g ... ', labindex, mode, opts.epit, t(1), t(2), rangeNumber, learningRate); % eg. training iter 1600 (2/rangeNumber), lr = 0.001 ... %     training epoch 1 (2/batchNumber), lr = 0.001
                batchTime = toc(batchTime) ;
                speed = cumuTrainedDataNumber/batchTime;
                
                for ac = 1:numel(accumulateOutBlobs)
                    if isinf(accumulateOutBlobs(ac))
                        error('A blob output = Inf');
                    elseif ~isempty(accumulateOutBlobs(ac))
                        fprintf(preStr);
                        fprintf('blob(''%s'') = %.6g ', net.blobNames{outputBlobID(ac)}, accumulateOutBlobs(ac)/cumuTrainedDataNumber);
                    end
                    if ac~=numel(accumulateOutBlobs)
                        fprintf('\n');
                    end
                end

                fprintf('%.2fs (%.1f data/s), ', batchTime, speed);
                fprintf('\n') ;
                batchTime = tic;
                accumulateOutBlobs = zeros(size(outputBlobID));
                cumuTrainedDataNumber = 0;
            end
        else
            if count == 1
                disp('Validating...');
            end
            if mod(count, opts.displayIter) == 0
                preStr = sprintf('LabNo.%d - %s: %s %d (%d/%d), ', labindex, mode, opts.epit, t(1), cumuTrainedDataNumber, batchStruct.m);
                batchTime = toc(batchTime) ;
                speed = cumuTrainedDataNumber/batchTime;
                
                for ac = 1:numel(accumulateOutBlobs)
                    if isinf(accumulateOutBlobs(ac))
                        error('A blob output = Inf');
                    elseif ~isempty(accumulateOutBlobs(ac))
                        fprintf(preStr);
                        fprintf('blob(''%s'') = %.6g ', net.blobNames{outputBlobID(ac)}, accumulateOutBlobs(ac)/cumuTrainedDataNumber);
                    end
                    if ac~=numel(accumulateOutBlobs)
                        fprintf('\n');
                    end
                end

                fprintf('%.2fs (%.1f data/s), ', batchTime, speed);
                fprintf('\n') ;
                batchTime = tic;
                accumulateOutBlobs = zeros(size(outputBlobID));
                cumuTrainedDataNumber = 0;
            end
        end

        

        % update batchStruct
        if numel(accumulateOutBlobs) == 1
            batchStruct.lastErrorRateOfData(batchStruct.lastBatchIndices) = accumulateOutBlobs(1)/cumuTrainedDataNumber;
        end

        count = count+1;
        globalBatchNum = globalBatchNum+1;

    end

end
